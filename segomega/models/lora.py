"""Minimal LoRA for nn.Linear, used to fine-tune the VGGT-Omega backbone cheaply.

SegVGGT fine-tunes the backbone's frame/global attention with LoRA rank=32, alpha=32 (and trains the
new modules fully).  We replicate that option here: inject LoRA into the aggregator's attention/MLP
Linear layers, freeze the original weights, and only train the LoRA adapters + new heads.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 32, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True  # toggle at inference: off -> forward == original frozen Linear (no drift)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)  # start as identity (no change)

    # transparent drop-in: expose the wrapped Linear's attributes so introspecting code
    # (e.g. backbone modules reading .in_features / .weight) keeps working after injection.
    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.base(x)
        return self.base(x) + self.lora_b(self.lora_a(self.drop(x))) * self.scaling


def set_lora_enabled(module: nn.Module, flag: bool) -> int:
    """Toggle ALL LoRALinear adapters under `module`. off => backbone == original frozen weights.
    Returns the number of adapters toggled."""
    n = 0
    for m in module.modules():
        if isinstance(m, LoRALinear):
            m.enabled = flag
            n += 1
    return n


def inject_lora(
    module: nn.Module,
    rank: int = 32,
    alpha: float = 32.0,
    target_substrings: Iterable[str] = ("qkv", "q_proj", "k_proj", "v_proj", "proj", "fc1", "fc2"),
    verbose: bool = True,
) -> int:
    """Recursively replace matching nn.Linear children with LoRALinear. Returns #replaced."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and any(s in name for s in target_substrings):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            n += 1
        else:
            n += inject_lora(child, rank, alpha, target_substrings, verbose=False)
    if verbose:
        print(f"[LoRA] injected into {n} Linear layers (rank={rank}, alpha={alpha})")
    return n
