"""How fast is generation, with and without the KV-cache?

    python scripts/bench_generate.py          # gpt-llama (cache on/off) and the LSTM, on CPU and GPU
    python scripts/bench_generate.py --replot # redraw the table and chart from the saved timings

For every model and device: generate 240 tokens after an 8-token prompt (greedy, batch 1), timing
every token; then the same with 16 sequences at once. Saves the raw timings to
docs/results/m4_generation.json, then writes m4_generation.md and m4_latency.png (milliseconds per
token vs how many tokens were already generated).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from tinygpt.dataset import TokenData
from tinygpt.models import load_checkpoint
from tinygpt.plotting import INK, INK2, SURFACE, run_colors, style_axes


def timed(model, prompt: torch.Tensor, n: int, use_cache: bool) -> np.ndarray:
    """Seconds spent on each generated token (the first one includes reading the prompt)."""
    times, t = [], time.perf_counter()
    for _ in model.iter_tokens(prompt, n, temperature=0, use_cache=use_cache):
        now = time.perf_counter()                 # choosing a token moves it to the CPU: the GPU is done
        times.append(now - t)
        t = now
    return np.array(times)


def measure(ckpts: list[str], data_dir: str, new_tokens: int, batch: int) -> dict:
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + \
              (["cuda"] if torch.cuda.is_available() else [])
    val = TokenData(data_dir, "val")
    s, _ = next(iter(val.ranges.values()))
    prompt_ids = [val.eot_id] + val.data[s:s + 7].astype(int).tolist()
    runs = []
    for path in ckpts:
        for dev in devices:
            model, ck = load_checkpoint(path, dev)
            prompt = torch.tensor([prompt_ids], device=dev)
            modes = [("no cache", False), ("KV-cache", True)] if ck["model_type"] == "gpt" else [("state", True)]
            for mode, use_cache in modes:
                timed(model, prompt, 8, use_cache)                      # warm-up: first calls set up kernels
                t = timed(model, prompt, new_tokens, use_cache)
                t0 = time.perf_counter()
                for _ in model.iter_tokens(prompt.repeat(batch, 1), new_tokens, temperature=0, use_cache=use_cache):
                    pass
                run = dict(model=ck["train_config"]["run"], mode=mode, device=dev, prompt_ms=t[0] * 1000,
                           ms=[round(x * 1000, 4) for x in t[1:]],
                           batch_tokens_per_s=batch * new_tokens / (time.perf_counter() - t0))
                runs.append(run)
                ms = np.array(run["ms"])
                print(f"{run['model'] + ', ' + mode:<24} {dev:<4} {1000 / ms.mean():7.1f} tok/s | ms/token (median): "
                      f"first 20 {np.median(ms[:20]):5.1f}, last 20 {np.median(ms[-20:]):5.1f}, slowest "
                      f"{ms.max():6.1f} | batch {batch}: {run['batch_tokens_per_s']:7.0f} tok/s")
    return dict(prompt_tokens=len(prompt_ids), new_tokens=new_tokens, batch=batch, runs=runs)


def report(res: dict, out: Path, plot_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs, window = res["runs"], 9
    devices = list(dict.fromkeys(r["device"] for r in runs))
    fig, axes = plt.subplots(1, len(devices), figsize=(5.6 * len(devices), 4), facecolor=SURFACE, squeeze=False)
    colors = run_colors(list(dict.fromkeys(r["model"] for r in runs)))
    for ax, dev in zip(axes[0], devices):
        style_axes(ax)
        for r in (r for r in runs if r["device"] == dev):            # colour = model, dashed = without the cache
            # median of 9 neighbouring tokens: shows the trend, ignores one-off hiccups (they are in the table)
            smooth = np.median(np.lib.stride_tricks.sliding_window_view(np.array(r["ms"]), window), axis=1)
            ax.plot(np.arange(len(smooth)) + window // 2, smooth, color=colors[r["model"]], linewidth=2,
                    linestyle="--" if r["mode"] == "no cache" else "-", label=f"{r['model']}, {r['mode']}")
        ax.set_title(f"Time per generated token, {dev.upper()} (lower is better)", color=INK, loc="left",
                     fontsize=10)
        ax.set_xlabel("tokens already generated", color=INK2)
        ax.set_ylabel("milliseconds per token", color=INK2)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160, facecolor=SURFACE)

    table = ["| model | device | prompt (ms) | tokens/s | ms/token, first 20 | ms/token, last 20 | slowest token (ms) | "
             f"tokens/s, batch of {res['batch']} |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        ms = np.array(r["ms"])
        table.append(f"| {r['model']}, {r['mode']} | {r['device']} | {r['prompt_ms']:.1f} | {1000 / ms.mean():.1f} | "
                     f"{np.median(ms[:20]):.1f} | {np.median(ms[-20:]):.1f} | {ms.max():.1f} | "
                     f"{r['batch_tokens_per_s']:.0f} |")
    md = ["# M4 — generation speed", "",
          f"Greedy, {res['new_tokens']} new tokens after a prompt of {res['prompt_tokens']} tokens, batch 1. Each token is "
          "timed from the previous one (choosing a token waits for the GPU, so the times are real). tokens/s = "
          "tokens ÷ total time; first/last 20 = median time of those tokens; the chart shows the median of every 9 "
          "neighbouring tokens. Raw timings: `m4_generation.json` (redraw with `--replot`).", "",
          "\n".join(table), "", f"![latency]({plot_path.name})", ""]
    out.write_text("\n".join(md))
    print(f"-> {out}, {plot_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", help="default: gpt-llama and lstm")
    ap.add_argument("--data", default="data/tokens")
    ap.add_argument("--new-tokens", type=int, default=240)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="docs/results/m4_generation.md")
    ap.add_argument("--plot", default="docs/results/m4_latency.png")
    ap.add_argument("--replot", action="store_true", help="only redraw from the saved .json")
    args = ap.parse_args()

    out = Path(args.out)
    raw = out.with_suffix(".json")
    if args.replot:
        res = json.loads(raw.read_text())
    else:
        res = measure(args.ckpt or ["checkpoints/gpt-llama/best.pt", "checkpoints/lstm/best.pt"], args.data,
                      args.new_tokens, args.batch)
        out.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(json.dumps(res))
    report(res, out, Path(args.plot))


if __name__ == "__main__":
    main()
