"""Generate text from a trained checkpoint.

    python scripts/sample.py                                   # checkpoints/gpt-llama/best.pt, built-in prompts
    python scripts/sample.py --prompt "दिल्ली में" --n 3 --temperature 0.7
    python scripts/sample.py --temperature 0                   # greedy: always the most likely token
    python scripts/sample.py --top-p 0.9 --top-k 0             # nucleus sampling
    python scripts/sample.py --stream                          # print text as it is generated
"""
from __future__ import annotations

import argparse
import sys

import torch

from tinygpt.bpe import BPETokenizer
from tinygpt.device import pick_device
from tinygpt.models import load_checkpoint
from tinygpt.sampling import stream

PROMPTS = ["भारत की राजधानी", "महात्मा गांधी का जन्म", "yaar kal movie dekhne", "mujhe kal subah 7 baje"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/gpt-llama/best.pt")
    ap.add_argument("--tokenizer", default="tokenizer/tinygpt-indic-8k.json")
    ap.add_argument("--prompt", action="append", help="repeat for several prompts (default: 4 built-in)")
    ap.add_argument("--n", type=int, default=2, help="samples per prompt")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50, help="0 = off")
    ap.add_argument("--top-p", type=float, default=None, help="nucleus sampling, e.g. 0.9")
    ap.add_argument("--min-p", type=float, default=None, help="e.g. 0.1")
    ap.add_argument("--repetition-penalty", type=float, default=1.0, help="e.g. 1.3")
    ap.add_argument("--no-cache", action="store_true", help="re-read the whole context for every token")
    ap.add_argument("--stream", action="store_true", help="print each piece of text as soon as it exists")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, ck = load_checkpoint(args.ckpt, device)
    tok = BPETokenizer.load(args.tokenizer)
    gen = torch.Generator().manual_seed(args.seed)
    sampling = dict(temperature=args.temperature, top_k=args.top_k or None, top_p=args.top_p, min_p=args.min_p,
                    repetition_penalty=args.repetition_penalty, use_cache=not args.no_cache)
    print(f"{ck['model_type']} from {args.ckpt} (iter {ck['iter']}) | "
          + ", ".join(f"{k} {v}" for k, v in sampling.items() if v not in (None, 1.0) or k == "temperature") + "\n")
    for prompt in args.prompt or PROMPTS:
        for i in range(args.n):
            print(f"--- {prompt}  [{i + 1}]\n{prompt}", end="", flush=True)
            pieces = stream(model, tok, prompt, args.max_new_tokens, device=device, generator=gen, **sampling)
            if args.stream:
                for piece in pieces:
                    sys.stdout.write(piece)
                    sys.stdout.flush()
            else:
                sys.stdout.write("".join(pieces))
            print("\n")


if __name__ == "__main__":
    main()
