"""Collect every baseline into one table and plot the learning curves.

    python scripts/report.py              # all runs in checkpoints/ + docs/results/m2_ngram.json
    python scripts/report.py --runs lstm

Writes docs/results/m2_baselines.md and docs/results/m2_curves.png.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# chart styling: light surface, recessive grid, 2px lines, categorical hues in a fixed order
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
REFERENCES = ("bigram", "4-gram")                # n-gram rows drawn as reference lines


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def human(n: float) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.1f}k" if n >= 1e3 else f"{n:.0f}"


def table(ngrams: list[dict], runs: list[dict]) -> str:
    rows = ["| model | size | loss/token | bits/byte | Hindi | Hinglish | training |",
            "|---|---:|---:|---:|---:|---:|---|"]
    for r in ngrams:
        if r["model"] != "uniform" and "JM" not in r["smoothing"]:
            continue                              # add-k rows: see m2_ngram.md
        v = r["val"]
        size = f"{human(r['stored'])} n-grams" if r["stored"] else "—"
        how = "counting" if r["stored"] else "—"
        rows.append(f"| {r['model']} | {size} | {v['all']['loss']:.3f} | {v['all']['bpb']:.3f} | "
                    f"{v['hindi']['bpb']:.3f} | {v['hinglish']['bpb']:.3f} | {how} |")
    for r in runs:
        v = r["val"]
        rows.append(f"| **{r['run']}** ({r['model_type']}) | {human(r['params'])} params | "
                    f"{v['all']['loss']:.3f} | **{v['all']['bpb']:.3f}** | {v['hindi']['bpb']:.3f} | "
                    f"{v['hinglish']['bpb']:.3f} | {human(r['tokens_seen'])} tokens, "
                    f"{r['minutes']:.0f} min ({r['device']}) |")
    return "\n".join(rows)


def plot(curves: dict[str, list[dict]], ngrams: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=INK2)
        ax.set_xlabel("training tokens seen (millions)", color=INK2)

    refs = {r["model"]: r["val"]["all"]["bpb"] for r in ngrams if "JM" in r["smoothing"]}
    for name in REFERENCES:
        if name in refs:
            ax1.axhline(refs[name], color=MUTED, linewidth=1)
            ax1.annotate(f"{name} {refs[name]:.3f}", (0, refs[name]), xycoords=("axes fraction", "data"),
                         xytext=(4, 4), textcoords="offset points", ha="left", va="bottom", color=INK2,
                         fontsize=9)
    for color, (run, rows) in zip(SERIES, curves.items()):
        val = [r for r in rows if r["kind"] == "val" and r["iter"] > 0]
        tr = [r for r in rows if r["kind"] == "train"]
        x = [r["tokens"] / 1e6 for r in val]
        ax1.plot(x, [r["bpb"] for r in val], color=color, linewidth=2, label=run, solid_capstyle="round")
        ax1.plot(x[-1:], [val[-1]["bpb"]], "o", color=color, markersize=8, markeredgecolor=SURFACE,
                 markeredgewidth=2)
        ax1.annotate(f"{val[-1]['bpb']:.3f}", (x[-1], val[-1]["bpb"]), xytext=(0, -14),
                     textcoords="offset points", ha="center", color=INK, fontsize=9)
        ax2.plot([r["tokens"] / 1e6 for r in tr], [r["loss"] for r in tr], color=color, linewidth=1,
                 alpha=0.45, label=f"{run} train")
        ax2.plot(x, [r["loss"] for r in val], color=color, linewidth=2, label=f"{run} validation")
    ax1.set_title("Validation bits per byte (lower is better)", color=INK, loc="left", fontsize=11)
    ax1.set_ylabel("bits / byte", color=INK2)
    finals = [[r for r in rows if r["kind"] == "val"][-1]["bpb"] for rows in curves.values()]
    lows = [r["bpb"] for rows in curves.values() for r in rows if r["kind"] == "val" and r["iter"] > 0]
    if refs:                                      # zoom in on the n-gram lines and where the runs end
        lo = min([refs.get(n, math.inf) for n in REFERENCES] + lows)
        hi = max([refs.get("bigram", 0) * 1.12] + [f * 1.06 for f in finals])
        ax1.set_ylim(lo * 0.95, hi)
    ax2.set_title("Train vs validation loss (nats / token)", color=INK, loc="left", fontsize=11)
    ax2.set_ylabel("cross-entropy (nats)", color=INK2)
    for ax in ([ax1] if len(curves) > 1 else []) + [ax2]:   # one series needs no legend box
        ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--runs", nargs="*", help="run names (default: every run with a results.json)")
    ap.add_argument("--out", default="docs/results")
    args = ap.parse_args()

    out = Path(args.out)
    ngram_path = out / "m2_ngram.json"
    ngrams = json.loads(ngram_path.read_text()) if ngram_path.exists() else []
    ck = Path(args.checkpoints)
    names = args.runs or sorted(p.parent.name for p in ck.glob("*/results.json"))
    runs = [json.loads((ck / n / "results.json").read_text()) for n in names]
    curves = {n: read_jsonl(ck / n / "metrics.jsonl") for n in names}

    md = ["# M2 — all baselines", "",
          "Validation set: every token of `data/tokens/val.bin` (99.9k tokens). Bits per byte = total "
          "cross-entropy in bits ÷ UTF-8 bytes of the text, so it doesn't depend on the tokenizer. "
          "Lower is better.", "", table(ngrams, runs), "",
          "![learning curves](m2_curves.png)", ""]
    (out / "m2_baselines.md").write_text("\n".join(md))
    if curves:
        plot(curves, ngrams, out / "m2_curves.png")
    print("\n".join(md[4:5]))
    print(f"-> {out / 'm2_baselines.md'}" + (f", {out / 'm2_curves.png'}" if curves else ""))


if __name__ == "__main__":
    main()
