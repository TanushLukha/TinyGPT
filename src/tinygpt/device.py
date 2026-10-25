"""Device choice and seeding."""
from __future__ import annotations

import random

import numpy as np
import torch


def pick_device(prefer: str = "auto") -> torch.device:
    """Apple GPU (MPS) on a Mac, CUDA on an NVIDIA machine (e.g. Colab), otherwise CPU."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)          # also seeds CUDA and MPS
