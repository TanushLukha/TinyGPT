"""Where does each model win? Scores per data source, how much of each source repeats the training
text, and what mixing a neural model with the n-gram model gives.

    python scripts/compare_models.py                       # checkpoints/lstm/best.pt vs the best n-gram
    python scripts/compare_models.py --ckpt checkpoints/lstm/best.pt --ckpt checkpoints/gpt/best.pt

Needs docs/results/m2_ngram.json from scripts/ngram_baselines.py. Writes docs/results/compare.md.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tinygpt.dataset import HINDI, HINGLISH, TokenData
from tinygpt.device import pick_device
from tinygpt.metrics import bits_per_byte, token_logprobs
from tinygpt.models import NGramLM, Smoothing, load_checkpoint

ALPHAS = [round(a / 10, 1) for a in range(11)]           # weight of the neural model in the mixture


def best_ngram(path: Path) -> tuple[str, Smoothing]:
    rows = [r for r in json.loads(path.read_text()) if r.get("params", {}).get("kind") == "jm"]
    if not rows:
        raise SystemExit(f"{path} has no n-gram parameters: run `python scripts/ngram_baselines.py` first")
    row = min(rows, key=lambda r: r["val"]["all"]["loss"])
    p = row["params"]
    return row["model"], Smoothing(order=p["order"], kind="jm", lambdas=tuple(p["lambdas"]))


def mix(logp_a: np.ndarray, logp_b: np.ndarray, alpha: float) -> np.ndarray:
    """ln(alpha * p_a + (1 - alpha) * p_b), computed without leaving log space."""
    if alpha == 0:
        return logp_b
    if alpha == 1:
        return logp_a
    return np.logaddexp(math.log(alpha) + logp_a, math.log(1 - alpha) + logp_b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", help="repeatable (default: checkpoints/lstm/best.pt)")
    ap.add_argument("--data", default="data/tokens")
    ap.add_argument("--ngram-json", default="docs/results/m2_ngram.json")
    ap.add_argument("--out", default="docs/results/compare.md")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    train, val = TokenData(args.data, "train"), TokenData(args.data, "val")
    ng_name, sm = best_ngram(Path(args.ngram_json))
    lm = NGramLM(sm.order, val.vocab_size, val.eot_id).fit(np.asarray(train.data))
    logp: dict[str, dict[str, np.ndarray]] = {ng_name: {}}
    unseen = {}
    for src, (s, e) in val.ranges.items():
        stats = lm.stats(val.data, s, e)
        logp[ng_name][src] = np.log(lm.probs(stats, sm))
        unseen[src] = float((stats[sm.order - 1][0] == 0).mean())    # this n-gram never occurred in training

    device = pick_device(args.device)
    names = []
    for path in args.ckpt or ["checkpoints/lstm/best.pt"]:
        model, ck = load_checkpoint(path, device)
        name = ck["train_config"]["run"]
        names.append(name)
        block = ck["train_config"]["block_size"]
        logp[name] = {src: token_logprobs(model, val, block, device, s, e) for src, (s, e) in val.ranges.items()}

    def bpb(lp: dict[str, np.ndarray], sources) -> float:
        return bits_per_byte(-sum(lp[s].sum() for s in sources), val.n_bytes(sources))

    models = [ng_name] + names
    lines = ["| source | tokens | bytes/token | " + f"{sm.order}-grams unseen in training | "
             + " | ".join(models) + " |", "|---|---:|---:|---:|" + "---:|" * len(models)]
    for src, (s, e) in val.ranges.items():
        lines.append(f"| {src} | {e - s - 1:,} | {val.sources[src]['bytes'] / (e - s):.2f} | {unseen[src]:.1%} | "
                     + " | ".join(f"{bpb(logp[m], [src]):.3f}" for m in models) + " |")
    per_source = "\n".join(lines)

    groups = {"Hindi": [s for s in val.ranges if s in HINDI], "Hinglish": [s for s in val.ranges if s in HINGLISH],
              "all": list(val.ranges)}
    groups = {g: srcs for g, srcs in groups.items() if srcs}
    neural = names[0]
    lines = [f"| weight of {neural} | " + " | ".join(groups) + " |", "|---:|" + "---:|" * len(groups)]
    scores = {}
    for a in ALPHAS:
        mixed = {s: mix(logp[neural][s], logp[ng_name][s], a) for s in val.ranges}
        scores[a] = {g: bpb(mixed, srcs) for g, srcs in groups.items()}
    best = min(ALPHAS, key=lambda a: scores[a]["all"])
    for a in ALPHAS:
        cells = " | ".join(f"{scores[a][g]:.3f}" for g in groups)
        lines.append(f"| {'**' if a == best else ''}{a:.1f}{'**' if a == best else ''} | {cells} |")
    mixture = "\n".join(lines)

    md = ["# Where each model wins", "",
          "Bits per byte on the validation set, by data source (lower is better). The unseen column is "
          f"the share of the source's {sm.order}-grams (a token and the {sm.order - 1} before it) that never "
          "occur in the training data: low means the text repeats training phrases.", "", per_source, "",
          f"## Mixing {neural} with the {ng_name}", "",
          f"p = w · p_{neural} + (1 − w) · p_{ng_name}, for every token; the weight w is chosen on validation.",
          "", mixture, ""]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    print("\n".join(md[4:]))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
