"""Optimizer / scheduler / param-group construction.

Mirrors the SegVGGT & IGGT recipe: a *small* LR for the pretrained VGGT-Omega backbone and a *large*
LR for the freshly-initialized heads (mask-feature head + query decoder).  Optionally inject LoRA into
the backbone and train only the adapters + heads.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from ..models.lora import inject_lora, LoRALinear


def setup_trainable(model, mode: str = "lora", lora_rank: int = 32, lora_alpha: float = 32.0):
    """mode: 'full' (low-LR full backbone) | 'lora' (LoRA backbone) | 'frozen' (backbone frozen)
             | 'lora_frozen' (inject LoRA so a LoRA-trained ckpt loads, but FREEZE both the base weights
                              AND the adapters -> only decoder + head train; for high-quality head-only polish)."""
    if mode == "frozen":
        model.backbone.freeze()
    elif mode == "lora":
        model.backbone.freeze()
        inject_lora(model.backbone.net.aggregator, rank=lora_rank, alpha=lora_alpha)
    elif mode == "lora_frozen":
        model.backbone.freeze()
        inject_lora(model.backbone.net.aggregator, rank=lora_rank, alpha=lora_alpha)
        # freeze the freshly-injected adapters too (inject_lora leaves lora_a/lora_b trainable);
        # net effect: the ENTIRE backbone (base + LoRA) is frozen, only query_decoder + mask head train.
        n_frozen = 0
        for m in model.backbone.modules():
            if isinstance(m, LoRALinear):
                for p in m.parameters():
                    if p.requires_grad:
                        p.requires_grad_(False); n_frozen += 1
        print(f"[lora_frozen] LoRA adapters frozen ({n_frozen} tensors); backbone fully frozen, decoder+head trainable", flush=True)
    elif mode == "full":
        pass  # everything trainable; backbone gets a small LR via param groups
    else:
        raise ValueError(mode)
    return model


def build_param_groups(model, lr_backbone=1e-5, lr_head=2e-4, weight_decay=0.05, lr_uplift=None):
    # lr_uplift not None -> carve the learnable upsampler (.uplifts.) into its own (small-LR) group,
    #        tagged name="uplift" so the trainer can zero its LR (freeze) at a fixed step.
    backbone_params, head_params, uplift_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone."):
            backbone_params.append(p)
        elif lr_uplift is not None and ".uplifts." in n:
            uplift_params.append(p)
        else:
            head_params.append(p)
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head})
    if uplift_params:
        groups.append({"params": uplift_params, "lr": lr_uplift, "name": "uplift"})
    return groups, weight_decay


def build_optimizer(model, lr_backbone=1e-5, lr_head=2e-4, weight_decay=0.05, lr_uplift=None):
    groups, wd = build_param_groups(model, lr_backbone, lr_head, weight_decay, lr_uplift=lr_uplift)
    return torch.optim.AdamW(groups, weight_decay=wd, betas=(0.9, 0.95))


class WarmupCosine:
    """Per-group linear warmup -> cosine decay (keeps each group's base lr ratio)."""

    def __init__(self, optimizer, total_steps, warmup_frac=0.05, min_lr_ratio=0.01,
                 uplift_gi=None, uplift_anneal_step=0):
        self.opt = optimizer
        self.total = total_steps
        self.warmup = max(1, int(total_steps * warmup_frac))
        self.min_ratio = min_lr_ratio
        self.base = [g["lr"] for g in optimizer.param_groups]
        self.t = 0
        # the upsampler group warms up like the rest but then anneals on its OWN fast cosine
        # to EXACTLY 0 by uplift_anneal_step (smooth freeze right at the ~12k peak), others stay on the 100k cosine.
        self.uplift_gi = uplift_gi
        self.uplift_anneal = uplift_anneal_step

    def step(self):
        self.t += 1
        if self.t <= self.warmup:
            scale = self.t / self.warmup
        else:
            prog = (self.t - self.warmup) / max(1, self.total - self.warmup)
            scale = self.min_ratio + (1 - self.min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
        for i, (g, b) in enumerate(zip(self.opt.param_groups, self.base)):
            s = scale
            if i == self.uplift_gi and self.uplift_anneal > 0:           # fast per-group anneal -> 0 at uplift_anneal
                if self.t <= self.warmup:
                    s = self.t / self.warmup
                elif self.t < self.uplift_anneal:
                    p = (self.t - self.warmup) / max(1, self.uplift_anneal - self.warmup)
                    s = 0.5 * (1 + math.cos(math.pi * p))                # 1 -> 0 over (warmup, anneal]
                else:
                    s = 0.0
            g["lr"] = b * s
        return scale
