"""Count-based n-gram language models: the simplest possible baselines.

An n-gram model predicts a token from the n-1 tokens before it, using counts from training:

    p_ML(w | h) = c(h, w) / c(h)          "of the times h was seen, how often did w follow?"

Most long contexts h are never seen in training, and most (h, w) pairs have count 0, so the raw
estimate gives probability 0 to perfectly normal text (infinite loss). Smoothing fixes that:

  add-k:  p(w | h) = (c(h, w) + k) / (c(h) + k V)      pretend every token followed h k extra times
  JM:     p_n(w | h) = lam_n p_ML(w | h) + (1 - lam_n) p_{n-1}(w | shorter h)    (Jelinek-Mercer)
          p_0(w) = 1 / V, and if h was never seen, p_n = p_{n-1}

Counts are stored sparsely: every n-gram is coded as one int64 (t0 V^(n-1) + ... + t_{n-1}),
the codes are sorted, and a lookup is a binary search (np.searchsorted).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def ngram_codes(tokens: np.ndarray, n: int, V: int) -> np.ndarray:
    """Code every length-n window of `tokens` as one int64: t0*V^(n-1) + t1*V^(n-2) + ... + t_{n-1}."""
    if V ** n >= 2 ** 63:
        raise ValueError(f"order {n} with vocab {V} overflows int64")
    m = len(tokens) - n + 1
    code = np.zeros(max(m, 0), dtype=np.int64)
    for j in range(n):
        code = code * V + tokens[j:j + m].astype(np.int64)
    return code


def _lookup(keys: np.ndarray, table_keys: np.ndarray, table_vals: np.ndarray) -> np.ndarray:
    if len(table_keys) == 0:
        return np.zeros(len(keys), dtype=np.int64)
    idx = np.minimum(np.searchsorted(table_keys, keys), len(table_keys) - 1)
    return np.where(table_keys[idx] == keys, table_vals[idx], 0)


class _Table:
    """Counts of every n-gram seen in training, and of every context (its first n-1 tokens)."""

    def __init__(self, codes: np.ndarray, V: int):
        self.codes, self.counts = np.unique(codes, return_counts=True)
        self.ctx_codes, first = np.unique(self.codes // V, return_index=True)
        self.ctx_counts = np.add.reduceat(self.counts, first)

    def __len__(self) -> int:
        return len(self.codes)


@dataclass(frozen=True)
class Smoothing:
    order: int                       # 1 = unigram, 2 = bigram, 3 = trigram, ...
    kind: str = "jm"                 # "addk" or "jm"
    k: float = 1.0                   # add-k only
    lambdas: tuple = ()              # jm only: one weight per order 1..order

    def describe(self) -> str:
        if self.kind == "addk":
            return f"add-k, k={self.k:g}"
        return "JM, λ=(" + ", ".join(f"{x:g}" for x in self.lambdas) + ")"


class NGramLM:
    def __init__(self, order: int, vocab_size: int, eot_id: int):
        self.order, self.V, self.eot = order, vocab_size, eot_id
        self.tables: list[_Table] = []

    def fit(self, tokens: np.ndarray) -> "NGramLM":
        self.tables = [_Table(ngram_codes(tokens, n, self.V), self.V) for n in range(1, self.order + 1)]
        return self

    def n_stored(self, order: int) -> int:
        """Distinct n-grams kept for a model of this order (its 'parameters')."""
        return sum(len(t) for t in self.tables[:order])

    # -- scoring a stretch of text ------------------------------------------------------------
    def stats(self, tokens: np.ndarray, start: int, end: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """For each target i in [start+1, end) and each order n: (c(h, w), c(h)), where w = tokens[i]
        and h = the n-1 tokens before it. Positions before 0 count as <|endoftext|>."""
        lo = start + 2 - self.order                 # first position any history needs
        pad = max(0, -lo)
        seq = np.concatenate([np.full(pad, self.eot, dtype=np.int64),
                              np.asarray(tokens[max(lo, 0):end], dtype=np.int64)])
        out = []
        for n, table in enumerate(self.tables, start=1):
            codes = ngram_codes(seq, n, self.V)[self.order - n:]     # the n-grams ending at each target
            out.append((_lookup(codes, table.codes, table.counts),
                        _lookup(codes // self.V, table.ctx_codes, table.ctx_counts)))
        return out

    def probs(self, stats, sm: Smoothing) -> np.ndarray:
        """p(target | history) at every position described by `stats`."""
        if sm.kind == "addk":
            c_hw, c_h = stats[sm.order - 1]
            return (c_hw + sm.k) / (c_h + sm.k * self.V)
        p = np.full(len(stats[0][0]), 1.0 / self.V)
        for (c_hw, c_h), lam in zip(stats[:sm.order], sm.lambdas):
            ml = c_hw / np.maximum(c_h, 1)
            p = np.where(c_h > 0, lam * ml + (1 - lam) * p, p)
        return p

    # -- the full next-token distribution (for sampling and for tests) -----------------------
    def next_dist(self, history, sm: Smoothing) -> np.ndarray:
        hist = ([self.eot] * self.order + [int(t) for t in history])[len(history) + 1:]   # last order-1
        rows = []
        for n in range(1, sm.order + 1):
            table = self.tables[n - 1]
            ctx = 0
            for t in hist[len(hist) - (n - 1):]:
                ctx = ctx * self.V + t
            lo, hi = np.searchsorted(table.codes, [ctx * self.V, (ctx + 1) * self.V])
            row = np.zeros(self.V)
            row[table.codes[lo:hi] - ctx * self.V] = table.counts[lo:hi]
            rows.append(row)                          # row.sum() == c(h)
        if sm.kind == "addk":
            return (rows[-1] + sm.k) / (rows[-1].sum() + sm.k * self.V)
        p = np.full(self.V, 1.0 / self.V)
        for row, lam in zip(rows, sm.lambdas):
            if row.sum() > 0:
                p = lam * row / row.sum() + (1 - lam) * p
        return p

    def generate(self, prompt, max_new_tokens: int, sm: Smoothing, rng: np.random.Generator) -> list[int]:
        out = [int(t) for t in prompt]
        for _ in range(max_new_tokens):
            p = self.next_dist(out, sm)
            t = int(rng.choice(self.V, p=p / p.sum()))
            out.append(t)
            if t == self.eot:
                break
        return out
