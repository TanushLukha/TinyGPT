"""M3: a GPT — a decoder-only Transformer — written from scratch.

    ids -> token embedding (+ position embedding) -> n_layers × Block -> final norm -> logits
    Block:  x = x + Attention(Norm(x))
            x = x + MLP(Norm(x))            "pre-norm": normalise the input of each branch,
                                            keep the residual path x itself untouched

Two recipes behind one class (`style`):
  * "gpt2"  — learned absolute positions, LayerNorm, GELU MLP, biases     (GPT-2, 2019)
  * "llama" — rotary positions (RoPE), RMSNorm, SwiGLU MLP, no biases     (Llama, 2023)
and two implementations of attention with identical maths (`attn`):
  * "manual" — scores, causal mask, softmax and weighted sum written out (and attention maps can
               be recorded for plotting)
  * "sdpa"   — torch's fused scaled_dot_product_attention, which never stores the T × T matrix
tests/test_gpt.py checks that the two give the same outputs and gradients.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sampling import sample_next


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    block_size: int = 256       # longest context seen in training (= window length)
    d_model: int = 384          # width of the residual stream
    n_layers: int = 6
    n_heads: int = 6            # head size = d_model / n_heads = 64
    dropout: float = 0.1
    style: str = "gpt2"         # "gpt2" or "llama"
    attn: str = "sdpa"          # "sdpa" (fused) or "manual" (written out)
    tie_weights: bool = True    # output layer = token embedding matrix
    rope_base: float = 10000.0  # RoPE only: wavelength scale of the rotations


class RMSNorm(nn.Module):
    """LayerNorm without the mean subtraction and bias: x / rms(x) * g. Cheaper, works as well."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_tables(head_dim: int, max_len: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of the rotation angles: position p turns dimension pair i by p * base^(-2i/head_dim)."""
    inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)    # (head_dim / 2,)
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)                 # (max_len, head_dim / 2)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each pair of dimensions (0,1), (2,3), ... of x (B, H, T, head_dim) by its position's angle.

    Rotating queries and keys this way makes q_m · k_n depend only on m − n: attention sees
    *relative* positions, with no position parameters at all.
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        bias = cfg.style == "gpt2"
        self.n_heads, self.head_dim = cfg.n_heads, cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=bias)    # queries, keys, values at once
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=bias)       # mixes the heads' outputs
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.impl, self.rope = cfg.attn, cfg.style == "llama"
        self.record = False                    # set True to keep the last attention weights (plots)
        self.last_attn: torch.Tensor | None = None
        T = cfg.block_size
        self.register_buffer("causal", torch.tril(torch.ones(T, T, dtype=torch.bool)), persistent=False)

    def forward(self, x: torch.Tensor, cos=None, sin=None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v))  # (B, H, T, hd)
        if self.rope:
            q, k = apply_rope(q, cos[:T], sin[:T]), apply_rope(k, cos[:T], sin[:T])
        p = self.dropout if self.training else 0.0
        if self.impl == "manual" or self.record:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B, H, T, T): query t vs key s
            att = att.masked_fill(~self.causal[:T, :T], float("-inf"))   # no looking at later tokens
            att = F.softmax(att, dim=-1)                                 # each row sums to 1
            if self.record:
                self.last_attn = att.detach()
            y = F.dropout(att, p, self.training) @ v                     # weighted sum of values
        else:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)                 # concatenate the heads
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """Per-token feed-forward network. gpt2: d → 4d → GELU → d.
    llama (SwiGLU): two d → h projections, one gated by the other, then h → d; h = 8d/3 keeps the
    parameter count equal to the GELU version (3 matrices of d×h instead of 2 of d×4d)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        d = cfg.d_model
        self.swiglu = cfg.style == "llama"
        if self.swiglu:
            h = 64 * math.ceil(8 * d / 3 / 64)
            self.gate, self.up = nn.Linear(d, h, bias=False), nn.Linear(d, h, bias=False)
            self.proj = nn.Linear(h, d, bias=False)
        else:
            self.fc, self.proj = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.gate(x)) * self.up(x) if self.swiglu else F.gelu(self.fc(x))
        return self.drop(self.proj(h))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        Norm = RMSNorm if cfg.style == "llama" else nn.LayerNorm
        self.ln1, self.attn = Norm(cfg.d_model), CausalSelfAttention(cfg)
        self.ln2, self.mlp = Norm(cfg.d_model), MLP(cfg)

    def forward(self, x: torch.Tensor, cos=None, sin=None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)     # tokens exchange information
        return x + self.mlp(self.ln2(x))             # each token processes what it gathered


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.style not in ("gpt2", "llama") or cfg.attn not in ("sdpa", "manual"):
            raise ValueError(f"unknown style/attn: {cfg.style}/{cfg.attn}")
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model) if cfg.style == "gpt2" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = RMSNorm(cfg.d_model) if cfg.style == "llama" else nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.style == "llama":
            cos, sin = rope_tables(cfg.d_model // cfg.n_heads, cfg.block_size, cfg.rope_base)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.rope_cos = self.rope_sin = None
        self.apply(self._init)
        for name, p in self.named_parameters():      # GPT-2: shrink the layers that write into the
            if name.endswith("proj.weight"):          # residual stream, so its variance doesn't grow
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))   # with depth
        if cfg.tie_weights:
            self.head.weight = self.wte.weight

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())       # a tied matrix is counted once
        if non_embedding:
            n -= self.wte.weight.numel() + (self.wpe.weight.numel() if self.wpe is not None else 0)
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence of {T} tokens > block_size {self.cfg.block_size}")
        x = self.wte(idx)
        if self.wpe is not None:
            x = x + self.wpe(torch.arange(T, device=idx.device))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, eot_id: int | None = None,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """Continue (1, T) prompt ids. Without a cache, every new token re-reads the whole context
        (the last block_size tokens) — M4 adds a KV-cache to avoid that."""
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])
            nxt = sample_next(logits[:, -1].cpu(), temperature, top_k, generator).to(idx.device)
            idx = torch.cat([idx, nxt], dim=1)
            if eot_id is not None and bool((nxt == eot_id).all()):
                break
        return idx
