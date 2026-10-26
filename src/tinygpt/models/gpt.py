"""M3: a GPT — a decoder-only Transformer — written from scratch.

    ids -> token embedding (+ position embedding) -> n_layers × Block -> final norm -> logits
    Block:  x = x + Attention(Norm(x))
            x = x + MLP(Norm(x))            "pre-norm": normalise the input of each branch,
                                            keep the residual path x itself untouched

Two recipes behind one class (`style`):
  * "gpt2"  — learned absolute positions, LayerNorm, GELU MLP, biases     (GPT-2, 2019)
  * "llama" — rotary positions (RoPE), RMSNorm, SwiGLU MLP, no biases     (Llama, 2023)
`pos`, `norm` and `mlp` can override single components (e.g. GPT-2 with RoPE, for ablations).
Two implementations of attention with identical maths (`attn`):
  * "manual" — scores, causal mask, softmax and weighted sum written out (and attention maps can
               be recorded for plotting)
  * "sdpa"   — torch's fused scaled_dot_product_attention
tests/test_gpt.py checks that the two give the same outputs and gradients.

M4 adds a KV-cache for generation: the keys and values of tokens already processed never change,
so each new token computes only its own and attends to the stored ones (tests/test_kvcache.py
checks that cached and uncached generation agree).

M6 lets one batch hold prompts of different lengths, for serving: shorter prompts are padded on the
left, padding is masked out of attention, and every row counts positions from its own first real
token — so each row computes what it would compute alone (tests/test_kvcache.py checks it).
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
    style: str = "gpt2"         # "gpt2" or "llama": sets the four components below
    pos: str = ""               # "learned" or "rope"           ("" = the style's choice)
    norm: str = ""              # "layernorm" or "rmsnorm"
    mlp: str = ""               # "gelu" or "swiglu"
    attn: str = "sdpa"          # "sdpa" (fused) or "manual" (written out)
    tie_weights: bool = True    # output layer = token embedding matrix
    rope_base: float = 10000.0  # RoPE only: wavelength scale of the rotations

    def __post_init__(self):
        llama = self.style == "llama"
        self.pos = self.pos or ("rope" if llama else "learned")
        self.norm = self.norm or ("rmsnorm" if llama else "layernorm")
        self.mlp = self.mlp or ("swiglu" if llama else "gelu")
        allowed = {"style": ("gpt2", "llama"), "pos": ("learned", "rope"), "norm": ("layernorm", "rmsnorm"),
                   "mlp": ("gelu", "swiglu"), "attn": ("sdpa", "manual")}
        for name, ok in allowed.items():
            if getattr(self, name) not in ok:
                raise ValueError(f"{name} must be one of {ok}, got {getattr(self, name)!r}")


class KVCache:
    """Keys and values of every token processed so far, for every layer.

    Buffers are allocated once for the whole context (block_size) and filled in place: appending by
    concatenation would copy the whole cache at every step.

    `pad` (B,): how many padding tokens start each row (left padding, M6); None = no padding.
    """

    def __init__(self, n_layers: int, batch: int, n_heads: int, max_len: int, head_dim: int, device, dtype,
                 pad: torch.Tensor | None = None):
        shape = (n_layers, batch, n_heads, max_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_len, self.pos = max_len, 0          # pos = number of tokens stored
        self.pad = None if pad is None else torch.as_tensor(pad, dtype=torch.long, device=device)

    def store(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Write the new tokens' keys/values (B, H, T, hd) after the stored ones; return all of them."""
        end = self.pos + k.size(2)
        self.k[layer, :, :, self.pos:end] = k
        self.v[layer, :, :, self.pos:end] = v
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2


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


def padding_mask(pad: torch.Tensor, pos: int, T: int) -> torch.Tensor:
    """Attention mask (B, 1, T, pos + T) for left-padded rows, True = may attend: the query at
    position pos + i sees the keys up to itself that are not padding.

    A padding query sees only itself. With nothing to attend to, its softmax would be 0/0: NaN in
    the hand-written attention (the fused kernel returns zeros there), and the NaN would spread —
    real tokens give padding keys weight 0, but 0 × NaN is still NaN.
    """
    L = pos + T
    keys = torch.arange(L, device=pad.device)
    queries = torch.arange(pos, L, device=pad.device)
    causal = keys[None, :] <= queries[:, None]                     # (T, L)
    real = keys[None, :] >= pad[:, None]                           # (B, L): not a padding token
    itself = keys[None, :] == queries[:, None]                     # (T, L)
    return (causal & (real[:, None, :] | itself))[:, None]         # (B, 1, T, L)


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
        self.impl, self.rope = cfg.attn, cfg.pos == "rope"
        self.record = False                    # set True to keep the last attention weights (plots)
        self.last_attn: torch.Tensor | None = None
        T = cfg.block_size
        self.register_buffer("causal", torch.tril(torch.ones(T, T, dtype=torch.bool)), persistent=False)

    def forward(self, x: torch.Tensor, cos=None, sin=None, cache: KVCache | None = None,
                layer: int = 0, mask: torch.Tensor | None = None) -> torch.Tensor:
        """cos, sin: RoPE angles of x's positions, (T, hd/2) — or (B, 1, T, hd/2) for padded rows.
        mask: None = plain causal attention; (B, 1, T, L) for padded rows (see padding_mask)."""
        B, T, C = x.shape
        pos = cache.pos if cache is not None else 0     # absolute position of x's first token
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v))  # (B, H, T, hd)
        if self.rope:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.store(layer, k, v)              # (B, H, pos + T, hd): old tokens + new ones
        L = k.size(2)
        padded = mask is not None
        if not padded:
            mask = self.causal[pos:pos + T, :L]          # query at position pos+i sees keys 0..pos+i
        p = self.dropout if self.training else 0.0
        if self.impl == "manual" or self.record:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B, H, T, L): query t vs key s
            att = att.masked_fill(~mask, float("-inf"))                  # no looking at later tokens
            att = F.softmax(att, dim=-1)                                 # each row sums to 1
            if self.record:
                self.last_attn = att.detach()
            y = F.dropout(att, p, self.training) @ v                     # weighted sum of values
        elif padded:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
        elif T == L:                                     # no cache, or the first chunk: plain causal
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
        elif T == 1:                                     # one new token: it may see everything stored
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
        y = y.transpose(1, 2).contiguous().view(B, T, C)                 # concatenate the heads
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """Per-token feed-forward network. gpt2: d → 4d → GELU → d.
    llama (SwiGLU): two d → h projections, one gated by the other, then h → d; h = 8d/3 keeps the
    parameter count equal to the GELU version (3 matrices of d×h instead of 2 of d×4d)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        d = cfg.d_model
        self.swiglu = cfg.mlp == "swiglu"
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
        Norm = RMSNorm if cfg.norm == "rmsnorm" else nn.LayerNorm
        self.ln1, self.attn = Norm(cfg.d_model), CausalSelfAttention(cfg)
        self.ln2, self.mlp = Norm(cfg.d_model), MLP(cfg)

    def forward(self, x: torch.Tensor, cos=None, sin=None, cache: KVCache | None = None,
                layer: int = 0, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin, cache, layer, mask)   # tokens exchange information
        return x + self.mlp(self.ln2(x))                           # each token processes what it gathered


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model) if cfg.pos == "learned" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = RMSNorm(cfg.d_model) if cfg.norm == "rmsnorm" else nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.pos == "rope":
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

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, cache: KVCache | None = None):
        """idx (B, T) -> logits (B, T, V). With a cache, idx are the tokens *after* those already
        stored, positions continue from cache.pos, and the cache is extended. With a padded cache
        (cache.pad), row b counts positions from its first real token and its padding is masked."""
        B, T = idx.shape
        pos = cache.pos if cache is not None else 0
        if pos + T > self.cfg.block_size:
            raise ValueError(f"{pos} + {T} tokens > block_size {self.cfg.block_size}")
        pad = cache.pad if cache is not None else None
        positions = torch.arange(pos, pos + T, device=idx.device)          # (T,)
        mask = None
        if pad is not None:
            mask = padding_mask(pad, pos, T)
            positions = (positions - pad[:, None]).clamp(min=0)              # (B, T): as if unpadded
        x = self.wte(idx)
        if self.wpe is not None:
            x = x + self.wpe(positions)
        x = self.drop(x)
        cos = sin = None
        if self.rope_cos is not None:
            cos, sin = self.rope_cos[positions], self.rope_sin[positions]    # (T, hd/2) or (B, T, hd/2)
            if pad is not None:
                cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)                # (B, 1, T, hd/2): every head
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, cache, i, mask)
        if cache is not None:
            cache.pos += T
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def new_cache(self, batch: int = 1, pad: torch.Tensor | None = None) -> KVCache:
        p = self.wte.weight
        return KVCache(self.cfg.n_layers, batch, self.cfg.n_heads, self.cfg.block_size,
                       self.cfg.d_model // self.cfg.n_heads, p.device, p.dtype, pad=pad)

    @torch.no_grad()
    def iter_tokens(self, idx: torch.Tensor, max_new_tokens: int, eot_id: int | None = None,
                    generator: torch.Generator | None = None, use_cache: bool = True, temperature: float = 1.0,
                    top_k: int | None = None, pad: torch.Tensor | None = None, **sampling):
        """Continue the (B, T) prompt `idx`, yielding each new (B, 1) token as soon as it is chosen.

        Without a cache, every step re-reads the last block_size tokens. With one, the prompt is read
        once ("prefill") and every step processes only the newest token ("decode"). When the context
        window is full, the cache restarts from the last half of the text: positions can't go
        past block_size, so its keys and values must be recomputed at new positions.

        `pad` (B,): row b of idx starts with pad[b] padding tokens — a batch of prompts of different
        lengths, padded on the left (M6). It needs the cache, and the text must fit in one window
        (a restart would have to re-pad every row).
        """
        self.eval()
        if pad is not None:
            if not use_cache:
                raise ValueError("a padded batch needs the KV-cache")
            if idx.size(1) + max_new_tokens - 1 > self.cfg.block_size:
                raise ValueError(f"a padded batch must fit in one window: {idx.size(1)} prompt tokens + "
                                 f"{max_new_tokens} new > block_size {self.cfg.block_size} + 1")
        history = idx.cpu()                                # for the repetition penalty
        cache = self.new_cache(idx.size(0), pad=pad) if use_cache else None
        x = idx[:, -self.cfg.block_size:]
        for _ in range(max_new_tokens):
            logits, _ = self(x, cache=cache)
            nxt = sample_next(logits[:, -1].cpu(), temperature, top_k, generator, history=history, **sampling)
            history = torch.cat([history, nxt], dim=1)
            yield nxt.to(idx.device)
            if eot_id is not None and bool((nxt == eot_id).all()):
                return
            if cache is None:
                x = history[:, -self.cfg.block_size:].to(idx.device)
            elif cache.pos < self.cfg.block_size:
                x = nxt.to(idx.device)                     # just the new token
            else:
                cache = self.new_cache(idx.size(0))        # window full: re-read the last half
                x = history[:, -(self.cfg.block_size // 2):].to(idx.device)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, **kwargs) -> torch.Tensor:
        """Prompt (B, T) -> prompt + continuation (B, T + n). kwargs: see iter_tokens / sample_next."""
        return torch.cat([idx, *self.iter_tokens(idx, max_new_tokens, **kwargs)], dim=1)
