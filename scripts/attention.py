"""Plot what each attention head looks at.

    python scripts/attention.py                                   # checkpoints/gpt/best.pt, all heads
    python scripts/attention.py --layer 0 --text "mujhe kal subah 7 baje ka alarm set karna hai"

Each small square is one head: row = the token being predicted from, column = the token it
attends to; darker = more attention. Everything above the diagonal is empty: no peeking ahead.
Writes docs/results/attention.png (or --out).

Matplotlib can't draw Devanagari properly (no text shaping), so for Hindi text the axes show token
numbers and the tokens are printed in the terminal.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tinygpt.bpe import BPETokenizer
from tinygpt.models import load_checkpoint
from tinygpt.plotting import INK, INK2, MUTED, SURFACE

RAMP = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]    # one hue, light -> dark


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/gpt/best.pt")
    ap.add_argument("--tokenizer", default="tokenizer/tinygpt-indic-8k.json")
    ap.add_argument("--text", default="mujhe kal subah 7 baje ka alarm set karna hai")
    ap.add_argument("--layer", type=int, help="one layer, with token labels (default: every layer)")
    ap.add_argument("--out", default="docs/results/attention.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    model, ck = load_checkpoint(args.ckpt, "cpu")
    if ck["model_type"] != "gpt":
        raise SystemExit("attention maps need a GPT checkpoint")
    tok = BPETokenizer.load(args.tokenizer)
    ids = [tok.eot_id] + tok.encode_ordinary(args.text)
    labels = ["<eot>"] + [tok.decode([i]).strip() or "_" for i in ids[1:]]    # "_" = a whitespace token
    if any(not lab.isascii() for lab in labels[1:]):
        print("tokens:", "  ".join(f"{i}:{lab!r}" for i, lab in enumerate(labels)))
        labels = [str(i) for i in range(len(ids))]
    for block in model.blocks:
        block.attn.record = True
    model(torch.tensor([ids]))
    maps = torch.stack([b.attn.last_attn[0] for b in model.blocks])          # (layers, heads, T, T)

    layers = [args.layer] if args.layer is not None else list(range(maps.shape[0]))
    H, T = maps.shape[1], len(ids)
    size = 2.6 if args.layer is not None else 1.35
    fig, axes = plt.subplots(len(layers), H, figsize=(H * size + 0.6, len(layers) * size + 0.8),
                             facecolor=SURFACE, squeeze=False)
    cmap = LinearSegmentedColormap.from_list("attention", RAMP)
    for r, layer in enumerate(layers):
        for h in range(H):
            ax = axes[r][h]
            ax.imshow(maps[layer, h].numpy(), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(MUTED)
                spine.set_linewidth(0.5)
            if r == 0:
                ax.set_title(f"head {h}", color=INK2, fontsize=9)
            if h == 0:
                ax.set_ylabel(f"layer {layer}", color=INK2, fontsize=9)
            if args.layer is not None:
                ax.set_xticks(range(T), labels, rotation=90, fontsize=7, color=INK2)
                if h == 0:
                    ax.set_yticks(range(T), labels, fontsize=7, color=INK2)
    shown = f"“{args.text}”" if args.text.isascii() else "(tokens numbered as printed in the terminal)"
    fig.suptitle(f"Attention of {Path(args.ckpt).parent.name}: {shown}", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
