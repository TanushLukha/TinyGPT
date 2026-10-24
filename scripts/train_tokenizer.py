"""Train the byte-level BPE tokenizer on a sample of data/clean/train.jsonl.

    python scripts/train_tokenizer.py                       # 8192 tokens, Devanagari-aware regex
    python scripts/train_tokenizer.py --pattern gpt4        # ablation: GPT-4's regex

Writes tokenizer/<name>.json (the tokenizer) and tokenizer/<name>.vocab.txt (readable vocab).
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

from tinygpt.bpe import BPETokenizer
from tinygpt.data import read_jsonl

EXAMPLES = [
    "भारत की राजधानी नई दिल्ली है।",
    "मुझे हिन्दी साहित्य पढ़ना पसंद है।",
    "kal mai office jaldi jaunga, phir movie dekhenge",
]


def sample_texts(path: Path, sample_mb: float) -> list[str]:
    """The same fraction of every source, chosen by hash (deterministic, keeps the mix)."""
    total = sum(len(t.encode("utf-8")) for t, _ in read_jsonl(path))
    keep = min(1.0, sample_mb * 1e6 / max(total, 1))
    out = []
    for text, _ in read_jsonl(path):
        h = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        if h % 10_000 < keep * 10_000:
            out.append(text)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/clean/train.jsonl")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--pattern", choices=["indic", "gpt4"], default="indic")
    ap.add_argument("--sample-mb", type=float, default=40, help="train on about this much text")
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--out", default="tokenizer")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    texts = sample_texts(Path(args.data), args.sample_mb)
    mb = sum(len(t.encode("utf-8")) for t in texts) / 1e6
    print(f"training on {len(texts):,} documents, {mb:.1f} MB, pattern={args.pattern}")
    t0 = time.time()
    tok = BPETokenizer.train(texts, args.vocab_size, pattern=args.pattern,
                             min_frequency=args.min_frequency, verbose=True)
    print(f"learned {len(tok.merges):,} merges in {time.time() - t0:.0f}s -> vocab {tok.vocab_size:,}")

    name = args.name or f"tinygpt-{args.pattern}-{args.vocab_size // 1024}k"
    out = Path(args.out)
    tok.save(out / f"{name}.json")
    tok.save_vocab(out / f"{name}.vocab.txt")
    print(f"saved {out / (name + '.json')}\n")
    for s in EXAMPLES:
        ids = tok.encode(s)
        print(f"{len(ids):>3} tokens | " + " | ".join(tok.token_str(i) for i in ids))


if __name__ == "__main__":
    main()
