"""LoRA (Hu et al., 2021): fine-tune a big frozen matrix W through a small low-rank update.

    y = x Wᵀ + b  +  (α / r) · x Aᵀ Bᵀ          W (out × in), b: pretrained, frozen
                                                  A (r × in), B (out × r): the only trainable weights

ΔW = (α/r)·BA has rank ≤ r, so a 384 × 1,152 projection learns through 8 × (384 + 1,152) = 12,288
numbers instead of 442,368. B starts at zero, so the adapted model starts *exactly* as the
pretrained one; A starts random (as nn.Linear's default init, like Microsoft's loralib), so B gets
useful gradients from the first step. After training, ΔW can be added into W ("merging"): the
model is then an ordinary GPT again, with no extra cost at inference.

    names = apply_lora(model, r=8, alpha=16, targets=("attn", "mlp"))   # freezes everything else
    ...train...
    torch.save(lora_state_dict(model), "adapter.pt")                    # ~0.4 MB instead of 55 MB
    merge_lora(model)                                                   # plain GPT, same outputs
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

# which nn.Linear layers of a GPT block each target name covers (both recipes)
TARGETS = {
    "attn": ("attn.qkv", "attn.proj"),
    "mlp": ("mlp.fc", "mlp.gate", "mlp.up", "mlp.proj"),
}


class LoRALinear(nn.Module):
    """An nn.Linear plus a trainable low-rank update; the wrapped layer itself stays frozen."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if r < 1:
            raise ValueError("LoRA rank must be at least 1")
        self.base, self.r, self.alpha, self.scaling = base, r, alpha, alpha / r
        w = base.weight
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=w.device, dtype=w.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=w.device, dtype=w.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for p in base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # two thin matrix products (in → r → out) instead of forming the full out × in update
        return self.base(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def delta(self) -> torch.Tensor:
        """The full update ΔW = (α/r)·BA, shape (out, in)."""
        return (self.lora_B @ self.lora_A) * self.scaling

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}"


def _target_names(model: nn.Module, targets) -> list[str]:
    suffixes = tuple(s for t in targets for s in TARGETS[t])
    return [name for name, m in model.named_modules()
            if isinstance(m, nn.Linear) and name.startswith("blocks.") and name.endswith(suffixes)]


def _set_module(model: nn.Module, name: str, module: nn.Module) -> None:
    parent, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, child, module)


def apply_lora(model: nn.Module, r: int = 8, alpha: float = 16, dropout: float = 0.0,
               targets=("attn", "mlp")) -> list[str]:
    """Freeze every weight of `model`, then wrap the target layers of every block in LoRALinear.
    Returns the names of the wrapped layers."""
    unknown = set(targets) - set(TARGETS)
    if unknown:
        raise ValueError(f"unknown LoRA targets {sorted(unknown)}; choose from {sorted(TARGETS)}")
    for p in model.parameters():
        p.requires_grad_(False)
    names = _target_names(model, targets)
    for name in names:
        _set_module(model, name, LoRALinear(model.get_submodule(name), r, alpha, dropout))
    return names


def lora_layers(model: nn.Module) -> dict[str, LoRALinear]:
    return {name: m for name, m in model.named_modules() if isinstance(m, LoRALinear)}


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the adapter: the lora_A / lora_B matrices (the frozen base is saved elsewhere, once)."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if ".lora_" in k}


def load_lora(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load an adapter saved by lora_state_dict into a model that went through apply_lora."""
    missing = [k for k in model.state_dict() if ".lora_" in k and k not in state]
    unexpected = [k for k in state if k not in model.state_dict()]
    if missing or unexpected:
        raise KeyError(f"adapter does not fit this model: missing {missing[:3]}, unexpected {unexpected[:3]}")
    model.load_state_dict(state, strict=False)


@torch.no_grad()
def merge_lora(model: nn.Module) -> nn.Module:
    """Fold every adapter into its layer (W ← W + ΔW) and unwrap it: the result is an ordinary
    model with the original parameter names, so a normal checkpoint can be saved and loaded."""
    for name, m in lora_layers(model).items():
        m.base.weight += m.delta()
        _set_module(model, name, m.base)
    return model


def count_params(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts; a tied matrix counts once."""
    seen, trainable, total = set(), 0, 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        trainable += p.numel() if p.requires_grad else 0
    return trainable, total
