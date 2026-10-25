"""Remove M5's evaluation requests from the pretraining text.

    python scripts/decontaminate.py        # data/clean -> data/clean_dc
    python scripts/encode_dataset.py --data data/clean_dc --out data/tokens_dc

M1 put every Hinglish-TOP request into the pretraining corpus, including the 1,390 validation and
6,513 test requests that M5 scores parses on. The model never saw their parses, but it did read the
requests themselves — a mild form of test contamination. This drops every pretraining line that
matches a validation or test request (ignoring case, punctuation and spacing). Training requests
stay: fine-tuning sees them anyway. Validation text for bits per byte stays the same, so the clean
model's scores remain comparable with M3's.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from tinygpt.top import load, match_key


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/clean")
    ap.add_argument("--out", default="data/clean_dc")
    ap.add_argument("--top-dir", default="data/top")
    args = ap.parse_args()

    banned = {match_key(e.query) for split in ("val", "test") for e in load(split, args.top_dir)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {"banned_requests": len(banned), "lines_kept": 0, "lines_removed": 0, "docs_changed": 0,
             "docs_removed": 0}
    with open(Path(args.data) / "train.jsonl", encoding="utf-8") as fin, \
            open(out / "train.jsonl", "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            if rec["src"] == "hinglish_top":           # documents of 16 requests, one per line
                lines = rec["text"].split("\n")
                kept = [s for s in lines if match_key(s) not in banned]
                stats["lines_kept"] += len(kept)
                stats["lines_removed"] += len(lines) - len(kept)
                if len(kept) < len(lines):
                    stats["docs_changed"] += 1
                if not kept:
                    stats["docs_removed"] += 1
                    continue
                rec["text"] = "\n".join(kept)
                line = json.dumps(rec, ensure_ascii=False) + "\n"
            fout.write(line)
    shutil.copy(Path(args.data) / "val.jsonl", out / "val.jsonl")
    (out / "decontamination.json").write_text(json.dumps(stats, indent=2))
    print(f"removed {stats['lines_removed']:,} of {stats['lines_removed'] + stats['lines_kept']:,} Hinglish-TOP "
          f"lines ({stats['docs_changed']:,} documents changed) -> {out}/")


if __name__ == "__main__":
    main()
