"""Turn data/clean/{train,val}.jsonl into token files the model trains on.

    python scripts/encode_dataset.py --tokenizer tokenizer/tinygpt-indic-8k.json

Writes data/tokens/{train,val}.bin: one long array of uint16 token ids, each document
followed by <|endoftext|>. Training (M2/M3) memory-maps these files and samples random
windows from them, so the whole corpus never has to fit in RAM.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tinygpt.bpe import BPETokenizer
from tinygpt.data import read_jsonl

_tok: BPETokenizer | None = None


def _init(path: str) -> None:
    global _tok
    _tok = BPETokenizer.load(path)


def _encode_batch(batch: list[tuple[str, str]]) -> list[tuple[str, list[int], int]]:
    assert _tok is not None
    return [(src, _tok.encode_ordinary(text) + [_tok.eot_id], len(text.encode("utf-8")))
            for text, src in batch]


def batches(path: Path, size: int = 64):
    buf = []
    for item in read_jsonl(path):
        buf.append(item)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default="tokenizer/tinygpt-indic-8k.json")
    ap.add_argument("--data", default="data/clean")
    ap.add_argument("--out", default="data/tokens")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    assert tok.vocab_size <= 2 ** 16, "uint16 storage needs vocab_size <= 65536"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"tokenizer": str(args.tokenizer), "vocab_size": tok.vocab_size, "eot_id": tok.eot_id,
            "dtype": "uint16", "splits": {}}
    for split in ("train", "val"):
        t0 = time.time()
        per_src: dict[str, dict[str, int]] = {}
        with open(out / f"{split}.bin", "wb") as f, Pool(args.workers, _init, (args.tokenizer,)) as pool:
            for results in tqdm(pool.imap(_encode_batch, batches(Path(args.data) / f"{split}.jsonl")),
                                desc=split, unit=" batches"):
                for src, ids, n_bytes in results:
                    np.asarray(ids, dtype=np.uint16).tofile(f)
                    s = per_src.setdefault(src, {"docs": 0, "tokens": 0, "bytes": 0})
                    s["docs"] += 1
                    s["tokens"] += len(ids)
                    s["bytes"] += n_bytes
        total = sum(s["tokens"] for s in per_src.values())
        meta["splits"][split] = {"tokens": total, "sources": per_src, "seconds": round(time.time() - t0, 1)}
        print(f"{split}: {total:,} tokens")
        for src, s in per_src.items():
            print(f"   {src:<14}{s['tokens']:>13,} tokens  {s['bytes'] / s['tokens']:.2f} bytes/token")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"-> {out}/train.bin, val.bin, meta.json")


if __name__ == "__main__":
    main()
