"""M2's neural baseline: an LSTM language model.

Why an LSTM? It was the strongest language-model architecture before the Transformer, so it is
the honest baseline for M3's GPT: same tokens, same context length, same optimiser and the same
evaluation; only the architecture changes.

Two implementations of the recurrent layer:
  * nn.LSTM     - PyTorch's fused kernel (cuDNN on NVIDIA, Metal on Apple). Used for training.
  * ScratchLSTM - the same equations written out by hand so every gate is visible. Its parameters
                  have nn.LSTM's names and shapes, so a checkpoint loads into either, and
                  tests/test_lstm.py checks that both give the same outputs and gradients.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sampling import sample_next


class ScratchLSTM(nn.Module):
    """Multi-layer, batch-first LSTM written gate by gate.

    For each layer and time step t, from the input x_t and the previous (h, c):
        i = sigmoid(W_ii x_t + W_hi h + b_i)    input gate:  how much of the candidate to write
        f = sigmoid(W_if x_t + W_hf h + b_f)    forget gate: how much of the old cell to keep
        g = tanh   (W_ig x_t + W_hg h + b_g)    candidate values
        o = sigmoid(W_io x_t + W_ho h + b_o)    output gate: how much of the cell to show
        c = f * c + i * g                       the cell is updated by ADDITION, which is what
        h = o * tanh(c)                         lets gradients survive many time steps
    The four gates are stacked in one matrix in the order i, f, g, o (nn.LSTM's layout).
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0,
                 batch_first: bool = True):
        super().__init__()
        if not batch_first:
            raise ValueError("ScratchLSTM only supports batch_first=True")
        self.hidden_size, self.num_layers, self.dropout = hidden_size, num_layers, dropout
        H = hidden_size
        for layer in range(num_layers):
            d_in = input_size if layer == 0 else H
            self.register_parameter(f"weight_ih_l{layer}", nn.Parameter(torch.empty(4 * H, d_in)))
            self.register_parameter(f"weight_hh_l{layer}", nn.Parameter(torch.empty(4 * H, H)))
            self.register_parameter(f"bias_ih_l{layer}", nn.Parameter(torch.empty(4 * H)))
            self.register_parameter(f"bias_hh_l{layer}", nn.Parameter(torch.empty(4 * H)))
        bound = 1 / math.sqrt(H)                 # nn.LSTM's default init: U(-1/sqrt(H), 1/sqrt(H))
        for p in self.parameters():
            nn.init.uniform_(p, -bound, bound)

    def forward(self, x: torch.Tensor, state=None):
        B, _, _ = x.shape
        L, H = self.num_layers, self.hidden_size
        h0, c0 = state if state is not None else (x.new_zeros(L, B, H), x.new_zeros(L, B, H))
        h_last, c_last = [], []
        for layer in range(L):
            w_ih = getattr(self, f"weight_ih_l{layer}")
            w_hh = getattr(self, f"weight_hh_l{layer}")
            b = getattr(self, f"bias_ih_l{layer}") + getattr(self, f"bias_hh_l{layer}")
            xw = x @ w_ih.T + b                  # input part of every time step in ONE matmul
            h, c = h0[layer], c0[layer]
            outs = []
            for t in range(x.size(1)):           # the recurrent part is sequential: step t needs h_{t-1}
                i, f, g, o = (xw[:, t] + h @ w_hh.T).chunk(4, dim=-1)
                c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
                h = torch.sigmoid(o) * torch.tanh(c)
                outs.append(h)
            x = torch.stack(outs, dim=1)
            if self.dropout and layer < L - 1:   # like nn.LSTM: dropout between layers, not after the last
                x = F.dropout(x, self.dropout, self.training)
            h_last.append(h)
            c_last.append(c)
        return x, (torch.stack(h_last), torch.stack(c_last))


@dataclass
class LSTMConfig:
    vocab_size: int = 8192
    d_model: int = 384          # embedding size = hidden size, which weight tying requires
    n_layers: int = 2
    dropout: float = 0.1
    init_range: float = 0.1     # embedding ~ U(-0.1, 0.1); GPT-2's N(0, 0.02) stalls this model (see M2 doc)
    forget_bias: float = 1.0    # start with the forget gate mostly open, so memories survive early on
    tie_weights: bool = True    # output layer = embedding matrix (saves 3.1M parameters)
    scratch: bool = False       # ScratchLSTM instead of nn.LSTM: same maths, much slower


class LSTMLM(nn.Module):
    """tokens -> embedding -> n_layers of LSTM -> linear layer -> next-token logits."""

    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        rnn = ScratchLSTM if cfg.scratch else nn.LSTM
        self.rnn = rnn(cfg.d_model, cfg.d_model, num_layers=cfg.n_layers,
                       dropout=cfg.dropout if cfg.n_layers > 1 else 0.0, batch_first=True)
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        nn.init.uniform_(self.emb.weight, -cfg.init_range, cfg.init_range)
        if cfg.tie_weights:
            self.head.weight = self.emb.weight
        else:
            nn.init.uniform_(self.head.weight, -cfg.init_range, cfg.init_range)
        H = cfg.d_model
        with torch.no_grad():                    # forget gate = rows H..2H of the stacked gates
            for layer in range(cfg.n_layers):
                getattr(self.rnn, f"bias_ih_l{layer}")[H:2 * H].fill_(cfg.forget_bias)
                getattr(self.rnn, f"bias_hh_l{layer}")[H:2 * H].zero_()

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())       # a tied matrix is counted once
        return n - self.emb.weight.numel() if non_embedding else n

    def step(self, idx: torch.Tensor, state=None):
        """(B, T) ids -> logits (B, T, V) and the final (h, c), which a later call can continue from."""
        out, state = self.rnn(self.drop(self.emb(idx)), state)
        return self.head(self.drop(out)), state

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits, _ = self.step(idx)               # every window starts from a zero state
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def iter_tokens(self, idx: torch.Tensor, max_new_tokens: int, eot_id: int | None = None,
                    generator: torch.Generator | None = None, use_cache: bool = True, temperature: float = 1.0,
                    top_k: int | None = None, **sampling):
        """Continue the (B, T) prompt, yielding each new (B, 1) token. The prompt is read once; after
        that each token costs one LSTM step, because (h, c) already summarises everything before it
        — the LSTM's state is its own "cache" (use_cache is accepted for a common interface)."""
        self.eval()
        history = idx.cpu()
        logits, state = self.step(idx)
        for _ in range(max_new_tokens):
            nxt = sample_next(logits[:, -1].cpu(), temperature, top_k, generator, history=history, **sampling)
            history = torch.cat([history, nxt], dim=1)
            yield nxt.to(idx.device)
            if eot_id is not None and bool((nxt == eot_id).all()):
                return
            logits, state = self.step(nxt.to(idx.device), state)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, **kwargs) -> torch.Tensor:
        """Prompt (B, T) -> prompt + continuation. kwargs: see iter_tokens / sampling.sample_next."""
        return torch.cat([idx, *self.iter_tokens(idx, max_new_tokens, **kwargs)], dim=1)
