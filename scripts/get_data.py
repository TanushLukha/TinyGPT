"""Download and clean the TinyGPT corpus into data/clean/{train,val}.jsonl.

    python scripts/get_data.py                 # full corpus: Hindi Wikipedia + Hinglish (internet)
    python scripts/get_data.py --source dev    # small dev corpus from GitHub, under a minute

Each line of the output is {"text": <document>, "src": <source name>}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from tqdm import tqdm

from tinygpt import data as D
from tinygpt.text import is_val


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["full", "dev"], default="full")
    ap.add_argument("--out", default="data/clean")
    ap.add_argument("--max-hindi-mb", type=float, default=120, help="cap on Hindi Wikipedia text (MB)")
    ap.add_argument("--val-permille", type=int, default=None,
                    help="documents per 1000 that go to validation (default 5 full, 20 dev)")
    args = ap.parse_args()

    if args.source == "full":
        sources = {
            "wiki_hi": lambda: D.hindi_wikipedia(int(args.max_hindi_mb * 1e6)),
            "findnitai": D.findnitai_hinglish,
            "hinglish_top": D.hinglish_top,
        }
        val_permille = args.val_permille or 5
    else:
        sources = {"ud_hindi": D.ud_hindi, "hinglish_top": lambda: D.hinglish_top(parts=("human",))}
        val_permille = args.val_permille or 20

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen: set[bytes] = set()
    stats: dict[str, dict] = {}
    with open(out / "train.jsonl", "w", encoding="utf-8") as ftrain, \
         open(out / "val.jsonl", "w", encoding="utf-8") as fval:
        for name, make in sources.items():
            st = stats[name] = {"docs": 0, "val_docs": 0, "bytes": 0, "duplicates": 0}
            t0 = time.time()
            try:
                for doc in tqdm(make(), desc=name, unit=" docs"):
                    key = hashlib.md5(doc.encode("utf-8")).digest()
                    if key in seen:
                        st["duplicates"] += 1
                        continue
                    seen.add(key)
                    line = json.dumps({"text": doc, "src": name}, ensure_ascii=False) + "\n"
                    if is_val(doc, val_permille):
                        fval.write(line)
                        st["val_docs"] += 1
                    else:
                        ftrain.write(line)
                    st["docs"] += 1
                    st["bytes"] += len(doc.encode("utf-8"))
            except Exception as e:                      # network / dataset problems: say what to do
                raise SystemExit(f"\n[{name}] failed: {e!r}\nCheck your internet connection, or run "
                                 f"`python scripts/get_data.py --source dev` to try the pipeline offline-ish.")
            st["seconds"] = round(time.time() - t0, 1)

    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    total = sum(s["bytes"] for s in stats.values())
    print(f"\n{'source':<14}{'docs':>10}{'val':>8}{'MB':>9}{'share':>8}{'dups':>8}")
    for name, s in stats.items():
        print(f"{name:<14}{s['docs']:>10,}{s['val_docs']:>8,}{s['bytes'] / 1e6:>9.1f}"
              f"{s['bytes'] / max(total, 1):>8.1%}{s['duplicates']:>8,}")
    print(f"-> {out / 'train.jsonl'}, {out / 'val.jsonl'}  ({total / 1e6:.1f} MB of text)")


if __name__ == "__main__":
    main()
