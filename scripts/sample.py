"""Generate text from a trained checkpoint.

    python scripts/sample.py                                  # checkpoints/lstm/best.pt, built-in prompts
    python scripts/sample.py --prompt "दिल्ली में" --n 3 --temperature 0.7
    python scripts/sample.py --temperature 0                  # greedy: always the most likely token
"""
from __future__ import annotations

import argparse

import torch

from tinygpt.bpe import BPETokenizer
from tinygpt.device import pick_device
from tinygpt.models import load_checkpoint
from tinygpt.sampling import complete

PROMPTS = ["भारत की राजधानी", "महात्मा गांधी का जन्म", "yaar kal movie dekhne", "mujhe kal subah 7 baje"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/lstm/best.pt")
    ap.add_argument("--tokenizer", default="tokenizer/tinygpt-indic-8k.json")
    ap.add_argument("--prompt", action="append", help="repeat for several prompts (default: 4 built-in)")
    ap.add_argument("--n", type=int, default=2, help="samples per prompt")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50, help="0 = no top-k cut-off")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, ck = load_checkpoint(args.ckpt, device)
    tok = BPETokenizer.load(args.tokenizer)
    gen = torch.Generator().manual_seed(args.seed)
    print(f"{ck['model_type']} from {args.ckpt} (iter {ck['iter']}) | temperature {args.temperature}, "
          f"top-k {args.top_k or 'off'}\n")
    for prompt in args.prompt or PROMPTS:
        for i in range(args.n):
            text = complete(model, tok, prompt, args.max_new_tokens, args.temperature, args.top_k or None,
                            device=device, generator=gen)
            print(f"--- {prompt}  [{i + 1}]\n{text}\n")


if __name__ == "__main__":
    main()
