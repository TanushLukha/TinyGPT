"""M5's tables and charts, from the fine-tuning runs.

    python scripts/report_m5.py --runs runs/m5-annotated --other-runs runs/m5

Writes docs/results/m5_finetune.md, m5_rank.png (exact match vs trainable weights) and m5_data.png
(exact match vs number of training examples). Learning rates are chosen on validation exact match
(scripts/m5_best.py); every other number is on the test set.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from m5_best import best_run  # noqa: E402

from tinygpt.bpe import BPETokenizer  # noqa: E402
from tinygpt.device import pick_device  # noqa: E402
from tinygpt.models import load_checkpoint  # noqa: E402
from tinygpt.plotting import AXIS, INK, INK2, SURFACE, run_colors, spread_labels, style_axes  # noqa: E402
from tinygpt.top import load, match_key, parse_tree, score, slot_items, target_text, to_model, top_intent  # noqa: E402

LRS = ["3e-5", "1e-4", "3e-4", "1e-3", "3e-3", "1e-2"]
NAMES = {"lora": "LoRA", "full": "full fine-tuning", "scratch": "from scratch"}


def pct(x) -> str:
    return "—" if x is None else f"{x:.1%}"


def human(n: float) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K" if n >= 1e3 else f"{n:.0f}"


def size(n_bytes) -> str:
    return "—" if not n_bytes else f"{n_bytes / 1e6:.1f} MB" if n_bytes >= 1e6 else f"{n_bytes / 1e3:.0f} KB"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(xs) -> float | None:
    return float(np.mean(xs)) if len(xs) else None


def same(a: str, b: str) -> bool:
    return " ".join(a.split()) == " ".join(b.split())


def error_kind(pred: str, gold: str) -> str:
    if same(pred, gold):
        return "correct"
    try:
        p = parse_tree(pred)
    except ValueError:
        return "not a valid tree"
    g = parse_tree(gold)
    if p.label != g.label:
        return "wrong intent"
    ps, gs = slot_items(p), slot_items(g)
    if sorted(label for label, _ in ps) != sorted(label for label, _ in gs):
        return "right intent, a slot missing, extra or mislabelled"
    if sorted(ps) != sorted(gs):
        return "right intent and slots, wrong words in a slot"
    return "right slots, wrong nesting or order"


@torch.no_grad()
def request_loss(ckpt: str, tok, queries: list[str], device) -> float:
    """Average -ln p per token of each request, read as a fresh document (<|endoftext|> before it)."""
    model, _ = load_checkpoint(ckpt, device)
    nats, count = 0.0, 0
    by_len: dict[int, list[list[int]]] = {}
    for q in queries:
        ids = [tok.eot_id] + tok.encode_ordinary(q)
        by_len.setdefault(len(ids), []).append(ids)
    for seqs in by_len.values():
        for s in range(0, len(seqs), 256):
            x = torch.tensor(seqs[s:s + 256], device=device)
            logp = torch.log_softmax(model(x[:, :-1])[0].float(), -1).gather(-1, x[:, 1:, None])
            nats -= logp.sum().item()
            count += logp.numel()
    return nats / count


def plot_rank(rows: list[tuple[int, int, float]], refs: dict[str, tuple[int, float]], path: Path,
              longer: dict[str, tuple[int, float]] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    style_axes(ax)
    colors = run_colors(["LoRA", "full fine-tuning", "from scratch"])
    x = [p for _, p, _ in rows]
    y = [em * 100 for _, _, em in rows]
    ax.plot(x, y, color=colors["LoRA"], linewidth=2, solid_capstyle="round", label="LoRA (rank r)", zorder=2)
    ax.plot(x, y, "o", color=colors["LoRA"], markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    for r, p, em in rows:
        ax.annotate(f"r={r}", (p, em * 100), xytext=(0, -16), textcoords="offset points", ha="center",
                    color=INK2, fontsize=9)
    for name, (p, em) in refs.items():
        ax.plot([p], [em * 100], "o", color=colors[name], markersize=9, markeredgecolor=SURFACE,
                markeredgewidth=2, label=name, zorder=3)
        ax.annotate(f"{name} {em:.1%}", (p, em * 100), xytext=(-8, 0), textcoords="offset points", ha="right",
                    va="center", color=INK2, fontsize=9)
    if longer:                                    # same colour as the method, hollow: 3x the steps
        ax.plot([], [], "o", markerfacecolor=SURFACE, markeredgecolor=INK2, markeredgewidth=2, markersize=8,
                linestyle="none", label="3× longer training")
    for name, (p, em) in (longer or {}).items():
        ax.plot([p], [em * 100], "o", markerfacecolor=SURFACE, markeredgecolor=colors[name], markeredgewidth=2,
                markersize=8, zorder=3)
        right = name == "LoRA"                    # room on the right of the LoRA point, not of full fine-tuning
        ax.annotate(f"{name}, 3× longer {em:.1%}", (p, em * 100), xytext=(8 if right else -8, 0),
                    textcoords="offset points", ha="left" if right else "right", va="center", color=INK2, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([1e4, 1e5, 1e6, 1e7])
    ax.set_xticklabels(["10K", "100K", "1M", "10M"])
    ax.set_xlabel("trainable weights (log scale)", color=INK2)
    ax.set_ylabel("test exact match (%)", color=INK2)
    every = y + [em * 100 for _, em in refs.values()] + [em * 100 for _, em in (longer or {}).values()]
    ax.set_ylim(max(0, min(every) - 8), min(100, max(every) + 4))
    ax.set_title("Parsing accuracy vs how many weights are trained", color=INK, loc="left", fontsize=10)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_data(curves: dict[str, list[tuple[int, float]]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    style_axes(ax)
    colors = run_colors(list(curves))
    ends = {}
    for name, pts in curves.items():
        xs, ys = [n for n, _ in pts], [em * 100 for _, em in pts]
        ax.plot(xs, ys, color=colors[name], linewidth=2, solid_capstyle="round", label=name)
        ax.plot(xs, ys, "o", color=colors[name], markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ends[name] = (xs[-1], ys[-1])
    names = list(ends)
    for name, ly in zip(names, spread_labels([ends[n][1] for n in names], gap=6)):   # in % points
        x, y = ends[name]                         # a label moved off its point gets a thin leader line
        ax.annotate(f"{name} {y:.1f}%", (x, y), xytext=(x * 1.15, ly), textcoords="data", va="center",
                    color=INK2, fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=1, shrinkA=2, shrinkB=5) if abs(ly - y) > 1 else None)
    ax.set_xscale("log")
    xs = sorted({n for pts in curves.values() for n, _ in pts})
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{n:,}" for n in xs])
    ax.minorticks_off()
    ax.set_xlim(xs[0] / 1.4, xs[-1] * 3.2)
    ax.set_ylim(0, 10 * math.ceil((max(em for pts in curves.values() for _, em in pts) * 100 + 8) / 10))
    ax.set_xlabel("training examples (log scale)", color=INK2)
    ax.set_ylabel("test exact match (%)", color=INK2)
    ax.set_title("How much labelled data each method needs", color=INK, loc="left", fontsize=10)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/m5", help="runs in the output format chosen on validation")
    ap.add_argument("--other-runs", default=None, help="runs in the other output format")
    ap.add_argument("--out", default="docs/results/m5_finetune.md")
    ap.add_argument("--clean-base", default="checkpoints/gpt-llama-clean")
    ap.add_argument("--orig-base", default="checkpoints/gpt-llama")
    ap.add_argument("--data-dir", default="data/top")
    ap.add_argument("--tokenizer", default="tokenizer/tinygpt-indic-8k.json")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    runs = {p.parent.name: json.loads(p.read_text()) for p in Path(args.runs).glob("*/results.json")}
    main_runs = {m: best_run(f"{m}-lr", args.runs)[0] for m in ("lora", "full", "scratch")}
    target = runs[main_runs["lora"]]["config"].get("target", "decoupled")
    R = lambda name: runs.get(name)  # noqa: E731
    test = lambda name, key="exact_match": R(name)["test"][key] if R(name) and "test" in R(name) else None  # noqa: E731
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base_bpb = json.loads((Path(args.clean_base) / "results.json").read_text())["val"]["all"]["bpb"]
    written = {"decoupled": "the decoupled tree", "annotated": "the request annotated with brackets (scored decoupled)"}
    md = ["# M5 — fine-tuning results", "",
          "Task: Hinglish-TOP semantic parsing (2,993 training, 1,390 validation, 6,513 test requests). Base model: "
          "`gpt-llama-clean` (M3's Llama recipe, pretrained without the validation and test requests). What the model "
          f"writes: {written[target]}. The output format and the learning rates were chosen on validation exact "
          "match; everything else is on the test set.", ""]

    # ---- main comparison
    md += ["## Methods", "",
           "| method | trainable weights | to store per task | val exact match | test exact match | intent | slot F1 | "
           "valid trees | general text after (bits/byte) | training |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    train_ex, test_ex = load("train", args.data_dir), load("test", args.data_dir)
    lookup: dict[str, str] = {}
    for e in train_ex:
        lookup.setdefault(match_key(e.query), e.parse)
    majority = Counter(top_intent(e.parse) for e in train_ex).most_common(1)[0][0]
    gold = [e.parse for e in test_ex]
    for label, guess in [("lookup table: the parse of the same training request, if there is one",
                          [lookup.get(match_key(e.query), "") for e in test_ex]),
                         (f"always the most common intent ([{majority} ])", [f"[{majority} ]"] * len(test_ex))]:
        t = score(guess, gold)
        md.append(f"| {label} | 0 | — | — | {pct(t['exact_match'])} | {pct(t['intent_acc'])} | {pct(t['slot_f1'])} | "
                  f"{pct(t['valid'])} | — | — |")
    rows = [("few-shot: 3 solved examples in the prompt", "fewshot-k3"), ("from scratch (random weights)", main_runs["scratch"]),
            ("full fine-tuning", main_runs["full"]), ("LoRA r=8, attention + MLP", main_runs["lora"]),
            ("LoRA r=8, attention only", "lora-attn")]
    for label, name in rows:
        r = R(name)
        if not r:
            continue
        t = r.get("test", {})
        bpb = r.get("lm_bpb", {}).get("all", base_bpb if r["method"] == "fewshot" else None)
        took = "—" if r["method"] == "fewshot" else f"{r['train_seconds'] / 60:.1f} min"
        md.append(f"| {label} | {human(r['trainable_params'])} | {size(4 * r['trainable_params'])} | "
                  f"{pct(r['val']['exact_match'])} | **{pct(t.get('exact_match'))}** | {pct(t.get('intent_acc'))} | "
                  f"{pct(t.get('slot_f1'))} | {pct(t.get('valid'))} | {bpb:.3f} | {took} |")
    md += ["", "To store per task: the trained weights at 4 bytes each (LoRA: the adapter; the others: a whole model). "
           f"General text: bits per byte on M1's validation text (base model: {base_bpb:.3f}); with LoRA, removing "
           "the adapter gives the base model back exactly.", ""]

    # ---- learning rates
    md += ["## Learning rates (validation exact match)", "",
           "| method | " + " | ".join(LRS) + " |", "|---|" + "---:|" * len(LRS)]
    for m in ("lora", "full", "scratch"):
        cells = []
        for lr in LRS:
            r = R(f"{m}-lr{lr}")
            cells.append("—" if not r else (f"**{pct(r['val']['exact_match'])}**" if f"{m}-lr{lr}" == main_runs[m]
                                              else pct(r["val"]["exact_match"])))
        md.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
    md.append("")

    # ---- LoRA rank and layers
    rank_rows = []
    md += ["## LoRA: rank and layers", "",
           "| rank | layers | trainable weights | share of the model | adapter (fp32) | test exact match | slot F1 |",
           "|---:|---|---:|---:|---:|---:|---:|"]
    for r_, name, layers in [(1, "lora-r1", "attention + MLP"), (4, "lora-r4", "attention + MLP"),
                             (8, main_runs["lora"], "attention + MLP"), (16, "lora-r16", "attention + MLP"),
                             (64, "lora-r64", "attention + MLP"), (8, "lora-attn", "attention only")]:
        r = R(name)
        if not r or "test" not in r:
            continue
        base_params = r["total_params"] - r["trainable_params"]
        md.append(f"| {r_} | {layers} | {r['trainable_params']:,} | {r['trainable_params'] / base_params:.2%} | "
                  f"{size(4 * r['trainable_params'])} | {pct(r['test']['exact_match'])} | {pct(r['test']['slot_f1'])} |")
        if layers == "attention + MLP":
            rank_rows.append((r_, r["trainable_params"], r["test"]["exact_match"]))
    md.append("")

    # ---- output format
    if args.other_runs and Path(args.other_runs).exists():
        md += ["## What the model writes (600 steps; learning rate chosen per format on validation)", "",
               "| output | example | LoRA | full fine-tuning | from scratch |", "|---|---|---:|---:|---:|"]
        example = {"decoupled": "`[in:create_alarm [sl:date_time do ghante ke liye ] ]`",
                   "annotated": "`[in:create_alarm [sl:date_time do ghante ke liye ] alarm set kardo ]`"}
        for d in (args.runs, args.other_runs):
            res = {p.parent.name: json.loads(p.read_text()) for p in Path(d).glob("*/results.json")}
            fmt = next(iter(res.values()))["config"].get("target", "decoupled") if res else "?"
            cells = []
            for m in ("lora", "full", "scratch"):
                try:
                    r = res[best_run(f"{m}-lr", d)[0]]
                    cells.append(f"{pct(r['test']['exact_match'])} test ({pct(r['val']['exact_match'])} val)"
                                 if "test" in r else f"{pct(r['val']['exact_match'])} val")
                except SystemExit:
                    cells.append("—")
            md.append(f"| {fmt} | {example.get(fmt, '')} | " + " | ".join(cells) + " |")
        md.append("")

    # ---- longer training
    long_rows = [(NAMES[m], main_runs[m], f"{m}-long") for m in ("lora", "full")]
    if all(test(n) is not None for _, _, n in long_rows):
        md += ["## Longer training (3× the steps)", "",
               "| method | 600 steps | 1,800 steps | best step |", "|---|---:|---:|---:|"]
        for label, short, long_ in long_rows:
            md.append(f"| {label} | {pct(test(short))} | {pct(test(long_))} | {R(long_)['best_step']:,} |")
        md.append("")

    # ---- seeds
    seeds = [(s, test(n)) for s, n in [(0, main_runs["lora"]), (1, "lora-seed1"), (2, "lora-seed2")] if test(n) is not None]
    if len(seeds) > 1:
        ems = np.array([em for _, em in seeds])
        md += ["## Run-to-run noise (LoRA r=8, three seeds)", "",
               "| seed | " + " | ".join(str(s) for s, _ in seeds) + " | mean | max − min |",
               "|---|" + "---:|" * (len(seeds) + 2),
               "| test exact match | " + " | ".join(pct(em) for _, em in seeds) +
               f" | {pct(ems.mean())} | {(ems.max() - ems.min()) * 100:.1f} points |", ""]

    # ---- label format
    if R("lora-upper") and "test" in R("lora-upper"):
        tok = BPETokenizer.load(args.tokenizer)
        train = load("train", args.data_dir)
        n_tok = {lab: np.mean([len(tok.encode_ordinary(" " + to_model(target_text(e, target), lab))) for e in train])
                 for lab in ("lower", "upper")}
        md += ["## How the model writes labels (LoRA r=8)", "",
               "| labels | example | tokens per parse | test exact match | training |", "|---|---|---:|---:|---:|"]
        for lab, name, ex in [("upper (as in the dataset)", "lora-upper", "[IN:CREATE_ALARM"),
                              ("lower case", main_runs["lora"], "[in:create_alarm")]:
            r = R(name)
            md.append(f"| {lab} | `{ex}` | {n_tok['upper' if 'upper' in lab else 'lower']:.1f} | "
                      f"{pct(test(name))} | {r['train_seconds'] / 60:.1f} min |")
        md.append("")

    # ---- data efficiency
    sizes = [100, 1000, 2993]
    curves = {}
    md += ["## How much labelled data (test exact match)", "",
           "| method | " + " | ".join(f"{n:,} examples" for n in sizes) + " |", "|---|" + "---:|" * len(sizes)]
    for m in ("lora", "full", "scratch"):
        names = [f"{m}-n100", f"{m}-n1000", main_runs[m]]
        pts = [(n, test(name)) for n, name in zip(sizes, names) if test(name) is not None]
        if pts:
            curves[NAMES[m]] = pts
        md.append(f"| {NAMES[m]} | " + " | ".join(pct(test(n)) for n in names) + " |")
    md.append("")
    if len(curves) > 1:
        plot_data(curves, out_dir / "m5_data.png")
        md += ["![data efficiency](m5_data.png)", ""]
    if rank_rows:
        refs = {NAMES[m]: (R(main_runs[m])["trainable_params"], test(main_runs[m])) for m in ("full", "scratch")
                if test(main_runs[m]) is not None}
        longer = {NAMES[m]: (R(f"{m}-long")["trainable_params"], test(f"{m}-long")) for m in ("lora", "full")
                  if test(f"{m}-long") is not None}
        plot_rank(sorted(rank_rows), refs, out_dir / "m5_rank.png", longer)
        md += ["![rank](m5_rank.png)", ""]

    # ---- contamination
    if R("orig-lora") and R("orig-full"):
        device = pick_device(args.device)
        tok = BPETokenizer.load(args.tokenizer)
        tests_q = [e.query for e in load("test", args.data_dir)]
        train_q = [e.query for e in load("train", args.data_dir)]
        loss = {b: (request_loss(f"{d}/best.pt", tok, tests_q, device), request_loss(f"{d}/best.pt", tok, train_q, device))
                for b, d in (("orig", args.orig_base), ("clean", args.clean_base))}
        md += ["## Did reading the test requests during pretraining help?", "",
               "| base model | pretraining saw the test requests | LoRA test exact match | full fine-tuning test "
               "exact match | base loss on test requests | base loss on training requests |",
               "|---|---|---:|---:|---:|---:|",
               f"| gpt-llama (M3) | yes (not their parses) | {pct(test('orig-lora'))} | {pct(test('orig-full'))} | "
               f"{loss['orig'][0]:.3f} | {loss['orig'][1]:.3f} |",
               f"| gpt-llama-clean | no | {pct(test(main_runs['lora']))} | {pct(test(main_runs['full']))} | "
               f"{loss['clean'][0]:.3f} | {loss['clean'][1]:.3f} |", "",
               "Loss: nats per token of each request read as a new document. Both models read the training requests, "
               "so the last column is the control.", ""]

    # ---- seen vs unseen, domains, errors (best runs)
    train_keys = {match_key(e.query) for e in load("train", args.data_dir)}
    preds = {m: read_jsonl(Path(args.runs) / main_runs[m] / "predictions_test.jsonl") for m in ("lora", "full", "scratch")
             if (Path(args.runs) / main_runs[m] / "predictions_test.jsonl").exists()}
    if preds:
        val_ex = load("val", args.data_dir)
        n_val = sum(match_key(e.query) in train_keys for e in val_ex)
        md += ["## Requests already in the training set", "",
               f"{n_val / len(val_ex):.0%} of validation requests also appear in training, against "
               f"{sum(match_key(e.query) in train_keys for e in test_ex) / len(test_ex):.1%} of test requests — one reason "
               "validation scores run higher than test scores.", "",
               "| method | test requests also in training (exact match) | new requests (exact match) |", "|---|---:|---:|"]
        for m, rows_ in preds.items():
            seen = [same(r["pred"], r["gold"]) for r in rows_ if match_key(r["query"]) in train_keys]
            new = [same(r["pred"], r["gold"]) for r in rows_ if match_key(r["query"]) not in train_keys]
            md.append(f"| {NAMES[m]} | {pct(mean(seen))} ({len(seen):,}) | {pct(mean(new))} ({len(new):,}) |")
        md.append("")
        domains = sorted({r["domain"] for r in next(iter(preds.values()))})
        md += ["## By domain (test exact match)", "", "| method | " + " | ".join(domains) + " |",
               "|---|" + "---:|" * len(domains)]
        for m, rows_ in preds.items():
            md.append(f"| {NAMES[m]} | " + " | ".join(
                pct(mean([same(r["pred"], r["gold"]) for r in rows_ if r["domain"] == d])) for d in domains) + " |")
        md.append("")
    if "lora" in preds:
        kinds = Counter(error_kind(r["pred"], r["gold"]) for r in preds["lora"])
        n_err = sum(v for k, v in kinds.items() if k != "correct")
        md += ["## What LoRA gets wrong (test)", "", "| error | share of errors |", "|---|---:|"]
        for k, v in kinds.most_common():
            if k != "correct":
                md.append(f"| {k} | {v / n_err:.0%} ({v:,}) |")
        md += ["", "Examples:", ""]
        shown = Counter()
        for r in preds["lora"]:
            k = error_kind(r["pred"], r["gold"])
            if k != "correct" and shown[k] < 2:
                shown[k] += 1
                md += [f"- *{k}* — `{r['query']}`", f"  - gold: `{r['gold']}`", f"  - LoRA: `{r['pred']}`"]
        md.append("")

    Path(args.out).write_text("\n".join(md), encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
