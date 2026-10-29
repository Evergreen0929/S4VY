"""VGGT-Omega backbone wrapper.

Exposes:
  - the cached multi-layer tokens  (`aggregated_tokens_list`, `patch_token_start`) for our heads,
  - optionally the original depth/camera predictions for joint geometry supervision,
all from a *single* aggregator forward (no recompute).

Backbone facts (verified against codebase/vggt-omega/vggt_omega):
  * `Aggregator(patch_size=16, embed_dim=1024).forward(images[B,S,3,H,W]) -> (list[Tensor|None], patch_token_start)`
  * cached layers = [4, 11, 17, 23]; non-cached entries are None.
  * cached token tensor: (B, S, 1 + num_register(16) + Np, 2*embed_dim=2048).  patch_token_start = 17.
  * ResNet mean/std normalization is applied *inside* the aggregator, so feed images in [0,1].
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..paths import VGGT_OMEGA_CKPT, ensure_vggt_omega_on_path

ensure_vggt_omega_on_path()
from vggt_omega.models.vggt_omega import VGGTOmega  # noqa: E402

# Layers whose tokens the aggregator caches (everything else is None in the returned list).
CACHED_LAYER_IDX = [4, 11, 17, 23]
PATCH_TOKEN_START = 17  # 1 camera token + 16 register tokens
EMBED_DIM = 1024
TOKEN_DIM = 2 * EMBED_DIM  # 2048, frame/inter-frame concat


class VGGTOmegaBackbone(nn.Module):
    """Thin wrapper around VGGTOmega that returns the cached tokens (and optional geometry preds)."""

    def __init__(
        self,
        ckpt_path: Optional[str] = VGGT_OMEGA_CKPT,
        enable_depth: bool = True,
        enable_camera: bool = True,
        patch_size: int = 16,
        embed_dim: int = EMBED_DIM,
        load_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.token_dim = 2 * embed_dim
        self.patch_token_start = PATCH_TOKEN_START

        self.net = VGGTOmega(
            patch_size=patch_size,
            embed_dim=embed_dim,
            enable_camera=enable_camera,
            enable_depth=enable_depth,
            enable_alignment=False,
        )
        if load_pretrained and ckpt_path:
            sd = torch.load(ckpt_path, map_location="cpu")
            sd = sd.get("model", sd) if isinstance(sd, dict) else sd
            missing, unexpected = self.net.load_state_dict(sd, strict=False)
            if missing or unexpected:
                print(f"[VGGTOmegaBackbone] loaded ckpt with missing={len(missing)} unexpected={len(unexpected)}")

    # --- convenience accessors ----------------------------------------------------------------
    @property
    def aggregator(self) -> nn.Module:
        return self.net.aggregator

    def freeze(self) -> None:
        for p in self.net.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,
        run_depth: bool = False,
        run_camera: bool = False,
        amp: bool = True,
    ) -> dict:
        """images: (B, S, 3, H, W) in [0,1].

        Returns dict with:
          tokens_list        : list[Tensor|None] cached per-layer tokens, shape (B,S,T,2048)
          patch_token_start  : int (17)
          depth, depth_conf  : optional (B,S,H,W,1)/(B,S,H,W) if run_depth
          pose_enc           : optional (B,S,9) if run_camera
        """
        if images.dim() == 4:
            images = images.unsqueeze(0)
        amp_dtype = torch.bfloat16 if (amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if (amp and torch.cuda.is_available()) else _nullctx()
        with ctx:
            tokens_list, patch_token_start = self.net.aggregator(images)

        out = {"tokens_list": tokens_list, "patch_token_start": patch_token_start}

        # Geometry heads run in fp32 (they assert fp32 output), reusing the same tokens.
        if run_depth and self.net.dense_head is not None:
            with torch.autocast(device_type="cuda", enabled=False):
                depth, depth_conf = self.net.dense_head(tokens_list, images=images, patch_token_start=patch_token_start)
            out["depth"], out["depth_conf"] = depth, depth_conf
        if run_camera and self.net.camera_head is not None:
            with torch.autocast(device_type="cuda", enabled=False):
                out["pose_enc"] = self.net.camera_head(tokens_list, patch_token_start=patch_token_start)
        return out


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
