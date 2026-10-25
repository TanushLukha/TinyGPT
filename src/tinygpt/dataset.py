"""Token files produced by scripts/encode_dataset.py, read without loading them into RAM.

data/tokens/{train,val}.bin is one long uint16 array; documents are separated by
<|endoftext|> and written source by source (wiki_hi, then findnitai, then hinglish_top), so
each source occupies one contiguous range — that is what lets us score Hindi and Hinglish
separately.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

HINDI = ("wiki_hi", "ud_hindi")
HINGLISH = ("findnitai", "hinglish_top")


class TokenData:
    def __init__(self, data_dir: str | Path, split: str):
        data_dir = Path(data_dir)
        self.data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode="r")
        meta = json.loads((data_dir / "meta.json").read_text())
        self.vocab_size: int = meta["vocab_size"]
        self.eot_id: int = meta["eot_id"]
        self.sources: dict[str, dict] = meta["splits"][split]["sources"]
        self.ranges: dict[str, tuple[int, int]] = {}
        start = 0
        for src, s in self.sources.items():           # same order the file was written in
            self.ranges[src] = (start, start + s["tokens"])
            start += s["tokens"]
        if start != len(self.data):
            raise ValueError(f"meta.json says {start} tokens, {split}.bin has {len(self.data)}")

    def __len__(self) -> int:
        return len(self.data)

    def n_bytes(self, sources=None) -> int:
        """UTF-8 bytes of the text behind these tokens (the denominator of bits-per-byte)."""
        return sum(s["bytes"] for name, s in self.sources.items() if sources is None or name in sources)

    def batch(self, batch_size: int, block_size: int, rng: np.random.Generator):
        """Random windows: x = tokens[i : i+T], y = tokens[i+1 : i+T+1] (y is x shifted by one)."""
        ix = rng.integers(0, len(self.data) - block_size - 1, size=batch_size)
        x = np.stack([self.data[i:i + block_size] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1:i + 1 + block_size] for i in ix]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    def windows(self, block_size: int, start: int = 0, end: int | None = None,
                batch_size: int = 32) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Non-overlapping windows covering tokens[start:end] exactly once (for exact evaluation).

        Every token from start+1 to end-1 is predicted exactly once; full windows come in
        batches, the shorter last window on its own.
        """
        end = len(self.data) if end is None else end
        starts = list(range(start, end - 1, block_size))
        full = [s for s in starts if s + block_size + 1 <= end]
        for b in range(0, len(full), batch_size):
            chunk = full[b:b + batch_size]
            x = np.stack([self.data[s:s + block_size] for s in chunk]).astype(np.int64)
            y = np.stack([self.data[s + 1:s + 1 + block_size] for s in chunk]).astype(np.int64)
            yield torch.from_numpy(x), torch.from_numpy(y)
        rest = [s for s in starts if s + block_size + 1 > end]
        for s in rest:                               # at most one, shorter than block_size
            x = np.asarray(self.data[s:end - 1], dtype=np.int64)[None]
            y = np.asarray(self.data[s + 1:end], dtype=np.int64)[None]
            yield torch.from_numpy(x), torch.from_numpy(y)
