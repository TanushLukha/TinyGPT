"""Time a few training steps on every available device and project how long a full run takes.

    python scripts/bench.py                     # M2's LSTM, 32 x 256 tokens per step
    python scripts/bench.py --model gpt         # M3's GPT
    python scripts/bench.py --model gpt --style llama --attn manual

Measure instead of guessing: on Apple Silicon the CPU (with Apple's matrix units) can be
competitive with the GPU for small recurrent models.
"""
from __future__ import annotations

import argparse
import time

import torch

from tinygpt.models import build_model
from tinygpt.trainer import TrainConfig, make_optimizer


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="lstm")
    ap.add_argument("--d-model", type=int)
    ap.add_argument("--n-layers", type=int)
    ap.add_argument("--n-heads", type=int, help="gpt only")
    ap.add_argument("--style", help="gpt only: gpt2 or llama")
    ap.add_argument("--attn", help="gpt only: sdpa or manual")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    ap.add_argument("--block-size", type=int, default=TrainConfig.block_size)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--max-iters", type=int, default=TrainConfig.max_iters, help="for the projection")
    args = ap.parse_args()

    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + \
              (["cuda"] if torch.cuda.is_available() else [])
    tokens = args.batch_size * args.block_size
    config = {k: v for k, v in dict(vocab_size=args.vocab_size, d_model=args.d_model, n_layers=args.n_layers,
                                     n_heads=args.n_heads, style=args.style, attn=args.attn).items()
              if v is not None}
    if args.model == "gpt":
        config["block_size"] = args.block_size
    probe = build_model(args.model, **config)
    print(f"{args.model} {probe.num_params() / 1e6:.1f}M parameters {config} | {tokens:,} tokens/step | "
          f"torch {torch.__version__}")
    for name in devices:
        device = torch.device(name)
        torch.manual_seed(0)
        model = build_model(args.model, **config).to(device).train()
        opt = make_optimizer(model, TrainConfig())
        x = torch.randint(0, args.vocab_size, (args.batch_size, args.block_size), device=device)
        y = torch.randint(0, args.vocab_size, (args.batch_size, args.block_size), device=device)
        for i in range(3 + args.steps):          # 3 warm-up steps: first calls compile kernels
            if i == 3:
                sync(device)
                t0 = time.time()
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sync(device)
        dt = (time.time() - t0) / args.steps
        print(f"  {name:<5} {dt * 1000:7.0f} ms/step | {tokens / dt / 1e3:6.1f}k tokens/s | "
              f"{args.max_iters} steps = {args.max_iters * dt / 60:5.1f} min")


if __name__ == "__main__":
    main()
