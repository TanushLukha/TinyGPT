"""Models, and how to rebuild one from a checkpoint."""
from __future__ import annotations

import torch

from .gpt import GPT, GPTConfig
from .lstm import LSTMConfig, LSTMLM, ScratchLSTM
from .ngram import NGramLM, Smoothing

# model_type -> (config class, model class)
REGISTRY = {"lstm": (LSTMConfig, LSTMLM), "gpt": (GPTConfig, GPT)}

# Training settings that differ by architecture (everything else: TrainConfig's defaults).
# Each was chosen by experiment — see docs/M2_baselines.md and docs/M3_gpt.md.
TRAIN_DEFAULTS = {
    "lstm": {"run": "lstm", "lr": 1e-2, "min_lr": 1e-3, "weight_decay": 0.01},
    "gpt": {"run": "gpt", "lr": 1e-3, "min_lr": 1e-4, "weight_decay": 0.1},
}


def build_model(model_type: str, **config):
    cfg_cls, model_cls = REGISTRY[model_type]
    return model_cls(cfg_cls(**config))


def load_checkpoint(path, device="cpu"):
    """-> (model on `device` in eval mode, the checkpoint dict).

    weights_only=True: the file may only contain tensors and plain Python values, so loading a
    checkpoint can never run code hidden inside a pickle.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["model_type"], **ckpt["model_config"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


__all__ = ["GPT", "GPTConfig", "LSTMConfig", "LSTMLM", "ScratchLSTM", "NGramLM", "Smoothing", "REGISTRY",
           "TRAIN_DEFAULTS", "build_model", "load_checkpoint"]
