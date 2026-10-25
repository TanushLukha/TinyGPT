"""Models, and how to rebuild one from a checkpoint."""
from __future__ import annotations

import torch

from .lstm import LSTMConfig, LSTMLM, ScratchLSTM
from .ngram import NGramLM, Smoothing

# model_type -> (config class, model class). M3 adds "gpt".
REGISTRY = {"lstm": (LSTMConfig, LSTMLM)}


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


__all__ = ["LSTMConfig", "LSTMLM", "ScratchLSTM", "NGramLM", "Smoothing", "REGISTRY", "build_model",
           "load_checkpoint"]
