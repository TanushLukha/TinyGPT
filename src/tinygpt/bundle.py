"""A model bundle: one folder with everything needed to run one trained model (M6).

    models/lm/
        model.safetensors   the weights (the tied embedding / output matrix stored once)
        config.json         architecture, task, tokenizer, where the weights came from, their scores,
                            and a SHA-256 of every file
        tokenizer.json      the BPE tokenizer the model was trained with

Why safetensors and not torch.save: a .pt file is a pickle, and unpickling can run code — so a
model downloaded from someone else is a program you execute. (torch.load(weights_only=True) blocks
that, but only if every loader remembers it.) safetensors is a JSON header plus raw tensor bytes:
nothing to execute, fast to read (memory-mapped), readable from other languages, and the format the
Hugging Face Hub expects. Training checkpoints stay .pt: they also hold the optimizer and RNG state.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_model, save_model

from .bpe import BPETokenizer
from .models import build_model

FORMAT, VERSION = "tinygpt-bundle", 1
WEIGHTS, CONFIG, TOKENIZER = "model.safetensors", "config.json", "tokenizer.json"


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_bundle(out_dir: str | Path, model: torch.nn.Module, tokenizer_path: str | Path, task: dict,
                source: dict | None = None, scores: dict | None = None) -> Path:
    """Write `model` (a GPT) as a bundle. `task`: {"name": "lm"}, or {"name": "hinglish-top",
    "labels": ..., "target": ...} for the parser; `source` and `scores` are recorded as given."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if next(model.parameters()).device.type != "cpu":
        model = copy.deepcopy(model).cpu()          # a CPU copy; the caller's model stays where it is
    # save_model (not save_file) knows that head.weight and wte.weight are one tensor: it stores it
    # once, and load_model ties it again. save_file refuses shared tensors.
    save_model(model, str(out / WEIGHTS), metadata={"format": "pt"})
    (out / WEIGHTS).chmod(0o644)       # safetensors creates it rw------- (owner only); a bundle is shared
    shutil.copyfile(tokenizer_path, out / TOKENIZER)
    config = {
        "format": FORMAT, "version": VERSION,
        "model_type": "gpt", "model_config": dataclasses.asdict(model.cfg),
        "params": sum(p.numel() for p in model.parameters()),
        "task": task, "tokenizer": TOKENIZER,
        "source": source or {}, "scores": scores or {},
        "sha256": {name: sha256(out / name) for name in (WEIGHTS, TOKENIZER)},
    }
    (out / CONFIG).write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    return out


def read_config(path: str | Path) -> dict:
    config = json.loads((Path(path) / CONFIG).read_text())
    if config.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT}")
    if config.get("version", 0) > VERSION:
        raise ValueError(f"{path} is bundle version {config['version']}; this code reads up to {VERSION}")
    return config


def load_bundle(path: str | Path, device="cpu", verify: bool = True):
    """-> (model on `device` in eval mode, tokenizer, config). verify=True checks every file against
    the SHA-256 in config.json first, so a truncated download or a swapped file fails loudly."""
    path = Path(path)
    config = read_config(path)
    if verify:
        for name, digest in config["sha256"].items():
            if sha256(path / name) != digest:
                raise ValueError(f"{path / name} does not match the checksum in {CONFIG}")
    model = build_model(config["model_type"], **config["model_config"])
    load_model(model, str(path / WEIGHTS), strict=True)      # every weight present, nothing extra
    tok = BPETokenizer.load(path / config["tokenizer"])
    if tok.vocab_size != model.cfg.vocab_size:
        raise ValueError(f"tokenizer has {tok.vocab_size} tokens, the model {model.cfg.vocab_size}")
    return model.to(device).eval(), tok, config
