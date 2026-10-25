"""Scoring language models in a way that does not depend on the tokenizer.

Cross-entropy per TOKEN can't compare models with different tokenizers: a tokenizer with longer
tokens has fewer, harder predictions. Bits per BYTE divides the total information by the size
of the text itself, so a character model, a byte model and our BPE models all land on one scale.

    bits per byte = (sum over predicted tokens of -ln p(token)) / (ln 2 * number of UTF-8 bytes)

Every model in this repo (n-grams, LSTM, GPT) is scored on exactly the same predictions: in each
source's slice of val.bin, every token except the first is predicted once.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import HINDI, HINGLISH, TokenData

LN2 = math.log(2)


def bits_per_byte(total_nats: float, n_bytes: int) -> float:
    return total_nats / (LN2 * n_bytes)


def summarize(per_source: dict[str, tuple[float, int]], data: TokenData) -> dict:
    """{source: (total nats, tokens predicted)} -> loss per token and bits per byte, by language."""
    groups = {"hindi": HINDI, "hinglish": HINGLISH, "all": tuple(per_source)}
    out = {}
    for name, sources in groups.items():
        present = [s for s in sources if s in per_source]
        if not present:
            continue
        nats = sum(per_source[s][0] for s in present)
        tokens = sum(per_source[s][1] for s in present)
        out[name] = {"loss": nats / tokens, "bpb": bits_per_byte(nats, data.n_bytes(present)),
                     "tokens": tokens}
    return out


def score_sources(nll: Callable[[int, int], tuple[float, int]], data: TokenData) -> dict:
    """Run nll(start, end) -> (nats, tokens) on every source's slice and summarize."""
    return summarize({src: nll(s, e) for src, (s, e) in data.ranges.items()}, data)


@torch.no_grad()
def total_nll(model, data: TokenData, block_size: int, device, start: int = 0, end: int | None = None,
              batch_size: int = 32) -> tuple[float, int]:
    """Sum of -ln p over every token in [start+1, end), and how many tokens that was."""
    model.eval()
    nats, count = 0.0, 0
    for x, y in data.windows(block_size, start, end, batch_size):
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="sum")
        nats += loss.item()
        count += y.numel()
    return nats, count


@torch.no_grad()
def token_logprobs(model, data: TokenData, block_size: int, device, start: int = 0, end: int | None = None,
                   batch_size: int = 32) -> np.ndarray:
    """ln p of every token in [start+1, end), in order (the same predictions total_nll sums up)."""
    model.eval()
    out = []
    for x, y in data.windows(block_size, start, end, batch_size):
        logits, _ = model(x.to(device))
        logp = torch.log_softmax(logits.float(), dim=-1).gather(-1, y.to(device).unsqueeze(-1))
        out.append(logp.reshape(-1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def evaluate(model, data: TokenData, block_size: int, device, batch_size: int = 32) -> dict:
    """Exact validation scores of a neural model: every val token, no sampling."""
    return score_sources(lambda s, e: total_nll(model, data, block_size, device, s, e, batch_size), data)
