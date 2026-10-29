"""Gradient checkpointing for the VGGT-Omega aggregator (monkey-patch, no source edits).

The released VGGT-Omega is inference-only and stores ALL 24-block activations when we backprop, which
caps the frames-per-GPU. IGGT's aggregator checkpoints every block during training (verified:
iggt/models/aggregator.py uses torch.utils.checkpoint). We replicate that by wrapping each
frame/inter-frame block so its activations are recomputed in backward instead of stored.

Apply AFTER loading pretrained weights (wrapping renames state_dict keys to `<list>.<i>.block.*`),
and only on the STUDENT aggregator (not the frozen teacher).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class _CkptBlock(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, *args, **kwargs):
        # checkpoint only when actually training with grad (no-op for teacher/eval/inference)
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.block, *args, use_reentrant=False, **kwargs)
        return self.block(*args, **kwargs)


def enable_gradient_checkpointing(aggregator: nn.Module, verbose: bool = True) -> int:
    n = 0
    for name in ("frame_blocks", "inter_frame_blocks"):
        ml = getattr(aggregator, name, None)
        if ml is None:
            continue
        for i in range(len(ml)):
            if not isinstance(ml[i], _CkptBlock):
                ml[i] = _CkptBlock(ml[i])
                n += 1
    if verbose:
        print(f"[grad-ckpt] wrapped {n} aggregator blocks (recompute activations in backward)")
    return n
