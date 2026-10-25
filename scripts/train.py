"""Train a neural language model on data/tokens.

    python scripts/train.py                              # M2's LSTM baseline (checkpoints/lstm/)
    python scripts/train.py --resume                     # continue an interrupted run where it stopped
    python scripts/train.py --run lstm-small --d-model 256 --max-iters 2000

Every field of TrainConfig (src/tinygpt/trainer.py) and of the model's config
(src/tinygpt/models/lstm.py) is a flag: --batch-size, --lr, --d-model, --n-layers, --dropout ...
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch

from tinygpt.bpe import BPETokenizer
from tinygpt.dataset import TokenData
from tinygpt.device import seed_everything
from tinygpt.models import REGISTRY, build_model
from tinygpt.sampling import complete
from tinygpt.trainer import TrainConfig, train

PROMPTS = ["भारत की राजधानी", "yaar kal movie dekhne"]
NOT_FLAGS = {"vocab_size"}                       # comes from the data, not the command line


def _bool(s: str) -> bool:
    if s.lower() in ("1", "true", "yes", "y"):
        return True
    if s.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {s!r}")


def add_flags(ap: argparse.ArgumentParser, cls, title: str) -> None:
    group = ap.add_argument_group(title)
    for f in dataclasses.fields(cls):
        if f.name in NOT_FLAGS:
            continue
        kind = type(f.default)
        group.add_argument("--" + f.name.replace("_", "-"), dest=f.name, default=None,
                           type=_bool if kind is bool else kind, help=f"default: {f.default}")


def pick(cls, args, base: dict | None = None) -> dict:
    """Config values: defaults <- `base` (saved config, when resuming) <- flags given now."""
    values = {f.name: f.default for f in dataclasses.fields(cls)}
    values.update(base or {})
    values.update({k: v for k, v in vars(args).items() if k in values and v is not None})
    return values


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", choices=sorted(REGISTRY), default="lstm")
    model_type = pre.parse_known_args()[0].model
    cfg_cls = REGISTRY[model_type][0]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(REGISTRY), default="lstm")
    ap.add_argument("--resume", action="store_true", help="continue checkpoints/<run>/last.pt")
    add_flags(ap, TrainConfig, "training")
    add_flags(ap, cfg_cls, f"{model_type} model")
    args = ap.parse_args()

    saved = None
    if args.resume:
        run = args.run or TrainConfig.run
        out_dir = args.out_dir or TrainConfig.out_dir
        saved = json.loads((Path(out_dir) / run / "config.json").read_text())
        model_type = saved["model_type"]
    tcfg = TrainConfig(**pick(TrainConfig, args, saved and saved["train"]))
    mcfg = pick(REGISTRY[model_type][0], args, saved and saved["model"])
    meta = TokenData(tcfg.data_dir, "val")
    mcfg["vocab_size"] = meta.vocab_size

    seed_everything(tcfg.seed)                   # same seed -> same initial weights
    model = build_model(model_type, **mcfg)
    results = train(model, model_type, tcfg, resume=args.resume)

    tok_path = json.loads((Path(tcfg.data_dir) / "meta.json").read_text())["tokenizer"]
    if results and Path(tok_path).exists():
        tok = BPETokenizer.load(tok_path)
        device = next(model.parameters()).device
        gen = torch.Generator().manual_seed(0)
        print("\nsamples (temperature 0.8, top-k 50):")
        for prompt in PROMPTS:
            text = complete(model, tok, prompt, max_new_tokens=80, temperature=0.8, top_k=50,
                            device=device, generator=gen)
            print(f"--- {prompt}\n{text}\n")


if __name__ == "__main__":
    main()
