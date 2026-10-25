"""Train a neural language model on data/tokens.

    python scripts/train.py                              # M2's LSTM baseline   -> checkpoints/lstm/
    python scripts/train.py --model gpt                  # M3's GPT             -> checkpoints/gpt/
    python scripts/train.py --model gpt --run gpt-llama --style llama
    python scripts/train.py --resume --run gpt           # continue an interrupted run where it stopped

Every field of TrainConfig (src/tinygpt/trainer.py) and of the model's config (models/lstm.py,
models/gpt.py) is a flag: --batch-size, --lr, --d-model, --n-layers, --dropout ... Defaults that
differ between architectures (learning rate, weight decay, run name) are in TRAIN_DEFAULTS
(models/__init__.py). `python scripts/train.py --model gpt --help` lists them.
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
from tinygpt.models import REGISTRY, TRAIN_DEFAULTS, build_model
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


def add_flags(ap: argparse.ArgumentParser, cls, title: str, seen: set, overrides: dict) -> None:
    """One --flag per dataclass field. A field in both configs (block_size) gets one flag for both."""
    group = ap.add_argument_group(title)
    for f in dataclasses.fields(cls):
        if f.name in NOT_FLAGS or f.name in seen:
            continue
        seen.add(f.name)
        kind = type(f.default)
        group.add_argument("--" + f.name.replace("_", "-"), dest=f.name, default=None,
                           type=_bool if kind is bool else kind,
                           help=f"default: {overrides.get(f.name, f.default)}")


def pick(cls, args, *bases: dict | None) -> dict:
    """Config values: dataclass defaults <- each base in turn <- flags given now."""
    values = {f.name: f.default for f in dataclasses.fields(cls)}
    for base in bases:
        values.update({k: v for k, v in (base or {}).items() if k in values})
    values.update({k: v for k, v in vars(args).items() if k in values and v is not None})
    return values


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", choices=sorted(REGISTRY), default="lstm")
    pre.add_argument("--resume", action="store_true")
    pre.add_argument("--run")
    pre.add_argument("--out-dir", dest="out_dir")
    known = pre.parse_known_args()[0]
    model_type, saved = known.model, None
    if known.resume:                             # the saved run decides the architecture
        run = known.run or TRAIN_DEFAULTS[model_type]["run"]
        saved = json.loads((Path(known.out_dir or TrainConfig.out_dir) / run / "config.json").read_text())
        model_type = saved["model_type"]
    defaults = TRAIN_DEFAULTS[model_type]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(REGISTRY), default="lstm")
    ap.add_argument("--resume", action="store_true", help="continue checkpoints/<run>/last.pt")
    seen: set = set()
    add_flags(ap, TrainConfig, "training", seen, defaults)
    add_flags(ap, REGISTRY[model_type][0], f"{model_type} model", seen, {})
    args = ap.parse_args()

    tcfg = TrainConfig(**pick(TrainConfig, args, defaults, saved and saved["train"]))
    mcfg = pick(REGISTRY[model_type][0], args, saved and saved["model"])
    mcfg["vocab_size"] = TokenData(tcfg.data_dir, "val").vocab_size

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
