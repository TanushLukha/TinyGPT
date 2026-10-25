"""Head census: what does each attention head mostly do?

    python scripts/heads.py --ckpt checkpoints/gpt/best.pt --ckpt checkpoints/gpt-llama/best.pt

For every head, on 16 windows of Hindi Wikipedia: the share of its attention on the previous token,
on itself, on the window's first token, and how far back it looks on average. And in the copying test
(100 random tokens shown twice): the share on the token right *after* the earlier occurrence of the
current token — the signature of an induction head. Writes docs/results/heads.md and heads.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tinygpt.dataset import TokenData
from tinygpt.models import load_checkpoint
from tinygpt.plotting import INK, INK2, SURFACE

RAMP = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


@torch.no_grad()
def census(model, val: TokenData, n_windows: int = 16, copy_len: int = 100) -> dict[str, np.ndarray]:
    T = model.cfg.block_size
    s, _ = val.ranges["wiki_hi"] if "wiki_hi" in val.ranges else next(iter(val.ranges.values()))
    for block in model.blocks:
        block.attn.record = True
    x = torch.from_numpy(np.stack([val.data[s + i * T: s + (i + 1) * T] for i in range(n_windows)]).astype(np.int64))
    model(x)
    q = torch.arange(16, T)                                   # queries with at least 16 tokens of history
    keys = torch.arange(T)
    out = {k: [] for k in ("prev", "self", "first", "distance", "induction")}
    for block in model.blocks:
        a = block.attn.last_attn                              # (B, H, T, T)
        out["prev"].append(a[:, :, q, q - 1].mean((0, 2)))
        out["self"].append(a[:, :, q, q].mean((0, 2)))
        out["first"].append(a[:, :, q, 0].mean((0, 2)))
        back = (q[:, None] - keys[None, :]).clamp(min=0).float()
        out["distance"].append((a[:, :, q, :] * back).sum(-1).mean((0, 2)))
    L = min(copy_len, (T - 1) // 2)
    r = np.random.default_rng(0).integers(256, val.eot_id, size=(n_windows, L))
    seq = torch.tensor(np.concatenate([np.full((n_windows, 1), val.eot_id), r, r], axis=1))[:, :-1]
    model(seq)
    qs = torch.arange(L + 1, 2 * L)                           # second copy; its token at position p
    for block in model.blocks:                                # appeared at p - L; what followed it is at p - L + 1
        out["induction"].append(block.attn.last_attn[:, :, qs, qs - (L - 1)].mean((0, 2)))
    for block in model.blocks:
        block.attn.record = False
    return {k: torch.stack(v).numpy() for k, v in out.items()}       # each (layers, heads)


def plot(results: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("heads", RAMP)
    cols = [("prev", "attention on the previous token", 1.0), ("induction", "induction score (copy test)", 0.5)]
    fig, axes = plt.subplots(len(results), 2, figsize=(8.2, 2.9 * len(results)), facecolor=SURFACE, squeeze=False)
    for r, (name, res) in enumerate(results.items()):
        for c, (key, title, vmax) in enumerate(cols):
            ax, m = axes[r][c], res[key]
            ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax)
            for (i, j), v in np.ndenumerate(m):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if v > 0.6 * vmax else INK)
            ax.set_xticks(range(m.shape[1]), [f"h{h}" for h in range(m.shape[1])], fontsize=8, color=INK2)
            ax.set_yticks(range(m.shape[0]), [f"layer {i}" for i in range(m.shape[0])], fontsize=8, color=INK2)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(f"{name}: {title} (0–{vmax:g})", color=INK, fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", required=True, help="GPT checkpoints (repeatable)")
    ap.add_argument("--data", default="data/tokens")
    ap.add_argument("--out", default="docs/results/heads.md")
    ap.add_argument("--plot", default="docs/results/heads.png")
    args = ap.parse_args()

    val = TokenData(args.data, "val")
    results = {}
    for path in args.ckpt:
        model, ck = load_checkpoint(path, "cpu")
        results[ck["train_config"]["run"]] = census(model, val)

    lines = ["| model | strongest previous-token head | heads above 50% | strongest induction head | "
             "mean look-back, first layer → last |", "|---|---|---:|---|---|"]
    for name, res in results.items():
        p, ind, d = res["prev"], res["induction"], res["distance"].mean(axis=1)
        pi, ii = np.unravel_index(p.argmax(), p.shape), np.unravel_index(ind.argmax(), ind.shape)
        lines.append(f"| {name} | {p.max():.0%} (layer {pi[0]}, head {pi[1]}) | {(p > 0.5).sum()} | "
                     f"{ind.max():.0%} (layer {ii[0]}, head {ii[1]}) | "
                     + " → ".join(f"{x:.0f}" for x in d) + " tokens |")
    detail = []
    for name, res in results.items():
        detail += [f"### {name}", "", "| layer.head | previous | self | first | look-back | induction |",
                   "|---|---:|---:|---:|---:|---:|"]
        for (i, j), _ in np.ndenumerate(res["prev"]):
            detail.append(f"| {i}.{j} | {res['prev'][i, j]:.2f} | {res['self'][i, j]:.2f} | {res['first'][i, j]:.2f} "
                          f"| {res['distance'][i, j]:.1f} | {res['induction'][i, j]:.2f} |")
        detail.append("")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_path = Path(args.plot)
    plot(results, plot_path)
    md = ["# Attention-head census", "",
          "Measured on 16 windows of Hindi Wikipedia (queries with ≥ 16 tokens of history) and, for the "
          "induction score, on 100 random tokens shown twice. Shares are averages of attention weights.", "",
          "\n".join(lines), "", f"![head census]({plot_path.name})", "", "## Every head", ""] + detail
    out.write_text("\n".join(md))
    print("\n".join(lines))
    print(f"-> {out}, {plot_path}")


if __name__ == "__main__":
    main()
