"""Where does each model win, and why? Four views of the same validation predictions:

  1. scores per data source, next to how much of each source repeats the training text
  2. a mixture of the first neural model with the best n-gram model
  3. loss vs how much context the model has seen (position inside the 256-token window)
  4. a copying test: a random token sequence shown twice — can the model use the first copy?

    python scripts/compare_models.py                       # checkpoints/lstm/best.pt vs the best n-gram
    python scripts/compare_models.py --ckpt checkpoints/gpt/best.pt --ckpt checkpoints/lstm/best.pt \\
        --out docs/results/m3_compare.md --plot docs/results/m3_context.png

Needs docs/results/m2_ngram.json from scripts/ngram_baselines.py.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from tinygpt.dataset import HINDI, HINGLISH, TokenData
from tinygpt.device import pick_device
from tinygpt.metrics import bits_per_byte, token_logprobs
from tinygpt.models import NGramLM, Smoothing, load_checkpoint
from tinygpt.plotting import INK, INK2, MUTED, SURFACE, run_colors, style_axes

ALPHAS = [round(a / 10, 1) for a in range(11)]           # weight of the neural model in the mixture
BIN_EDGES = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]         # window positions, grouped on a log scale


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


def bin_labels(block: int) -> list[str]:
    edges = [e for e in BIN_EDGES if e < block] + [block]
    return [f"{a + 1}" if b - a == 1 else f"{a + 1}–{b}" for a, b in zip(edges, edges[1:])]


def by_context(lp: np.ndarray, block: int) -> list[float]:
    """Mean loss (nats/token) grouped by how many tokens of the window the model had seen."""
    pos = np.arange(len(lp)) % block                  # windows are contiguous: target j sits at j % block
    edges = [e for e in BIN_EDGES if e < block] + [block]
    return [float(-lp[(pos >= a) & (pos < b)].mean()) for a, b in zip(edges, edges[1:])]


def copy_len(block: int) -> int:
    return min(100, (block - 1) // 2)


@torch.no_grad()
def copy_test(model, block: int, V: int, eot: int, device, n_seq: int = 32, seed: int = 0) -> tuple[float, float]:
    """Random tokens r, then r again. Loss on the 1st copy is pure guessing; a model that can look
    back and copy (as attention can) gets the 2nd copy almost free. Returns (loss 1st, loss 2nd)."""
    L = copy_len(block)
    r = np.random.default_rng(seed).integers(256, eot, size=(n_seq, L))     # merged tokens, no specials
    seq = torch.tensor(np.concatenate([np.full((n_seq, 1), eot), r, r], axis=1), device=device)
    logits, _ = model(seq[:, :-1])
    lp = torch.log_softmax(logits.float(), -1).gather(-1, seq[:, 1:, None]).squeeze(-1).cpu().numpy()
    return float(-lp[:, :L].mean()), float(-lp[:, L + 1:].mean())   # 2nd copy minus its first token


def plot_context(curves: dict[str, list[float]], labels: list[str], ngram: str, path: Path, source: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    style_axes(ax)
    x = np.arange(len(labels))
    colors = run_colors([n for n in curves if n != ngram])
    first = True
    for name, ys in curves.items():
        if name == ngram:
            ax.plot(x, ys, color=MUTED, linewidth=1.5, label=f"{name} (sees 3 tokens back)")
        else:
            ax.plot(x, ys, color=colors[name], linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=name)
            if first:                              # label only the main model's end; the table has the rest
                ax.annotate(f"{ys[-1]:.2f}", (x[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                            va="center", color=INK, fontsize=9)
                first = False
    ax.set_xticks(x, labels)
    ax.set_xlabel("tokens of the window seen so far", color=INK2)
    ax.set_ylabel("loss (nats / token)", color=INK2)
    ax.set_title(f"Loss vs context, {source} (lower is better)", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", help="repeatable; the first one is mixed with the n-gram "
                                                    "(default: checkpoints/lstm/best.pt)")
    ap.add_argument("--data", default="data/tokens")
    ap.add_argument("--ngram-json", default="docs/results/m2_ngram.json")
    ap.add_argument("--out", default="docs/results/compare.md")
    ap.add_argument("--plot", default="docs/results/context.png")
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
    names, copies, block = [], {}, None
    for path in args.ckpt or ["checkpoints/lstm/best.pt"]:
        model, ck = load_checkpoint(path, device)
        name = ck["train_config"]["run"]
        names.append(name)
        if block is not None and ck["train_config"]["block_size"] != block:
            raise SystemExit("all checkpoints must use the same window length (block_size)")
        block = ck["train_config"]["block_size"]
        logp[name] = {src: token_logprobs(model, val, block, device, s, e) for src, (s, e) in val.ranges.items()}
        copies[name] = copy_test(model, block, val.vocab_size, val.eot_id, device)

    def bpb(lp: dict[str, np.ndarray], sources) -> float:
        return bits_per_byte(-sum(lp[s].sum() for s in sources), val.n_bytes(sources))

    # 1. per source
    models = [ng_name] + names
    lines = ["| source | tokens | bytes/token | " + f"{sm.order}-grams unseen in training | "
             + " | ".join(models) + " |", "|---|---:|---:|---:|" + "---:|" * len(models)]
    for src, (s, e) in val.ranges.items():
        lines.append(f"| {src} | {e - s - 1:,} | {val.sources[src]['bytes'] / (e - s):.2f} | {unseen[src]:.1%} | "
                     + " | ".join(f"{bpb(logp[m], [src]):.3f}" for m in models) + " |")
    per_source = "\n".join(lines)

    # 2. mixture of the first neural model with the n-gram
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

    # 3. loss vs context, on the long documents (Hindi Wikipedia) if present
    src = "wiki_hi" if "wiki_hi" in val.ranges else next(iter(val.ranges))
    labels = bin_labels(block)
    curves = {m: by_context(logp[m][src], block) for m in models}
    lines = ["| tokens seen | " + " | ".join(models) + " |", "|---:|" + "---:|" * len(models)]
    for i, lab in enumerate(labels):
        lines.append(f"| {lab} | " + " | ".join(f"{curves[m][i]:.3f}" for m in models) + " |")
    context = "\n".join(lines)
    plot_path = Path(args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_context(curves, labels, ng_name, plot_path, src)

    # 4. copying
    lines = ["| model | 1st copy | 2nd copy |", "|---|---:|---:|"]
    lines += [f"| {n} | {c1:.2f} | {c2:.2f} |" for n, (c1, c2) in copies.items()]
    copying = "\n".join(lines)

    md = ["# Where each model wins", "",
          "Bits per byte on the validation set, by data source (lower is better). The unseen column is "
          f"the share of the source's {sm.order}-grams (a token and the {sm.order - 1} before it) that never "
          "occur in the training data: low means the text repeats training phrases.", "", per_source, "",
          f"## Mixing {neural} with the {ng_name}", "",
          f"p = w · p_{neural} + (1 − w) · p_{ng_name}, for every token; the weight w is chosen on validation.",
          "", mixture, "",
          f"## Loss vs context ({src})", "",
          f"Mean loss (nats/token) by how many tokens of its {block}-token window a model had read when "
          f"predicting. The {ng_name} always looks {sm.order - 1} tokens back, so it barely changes.", "",
          context, "", f"![loss vs context]({plot_path.name})", "",
          "## Copying a random sequence", "",
          f"Loss (nats/token) on 32 sequences of {copy_len(block)} random tokens, each shown twice. The first "
          "copy can only be guessed; the second can be copied from the first by a model that can look back "
          "that far.", "",
          copying, ""]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    print("\n".join(md[4:]))
    print(f"-> {out}, {plot_path}")


if __name__ == "__main__":
    main()
