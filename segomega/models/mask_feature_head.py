"""Pixel decoder that turns the backbone's multi-scale patch tokens into a per-frame, high-resolution
mask-feature map.  Mask logits are then  <query_embed, mask_feature>  (dot product), exactly like
Mask2Former's pixel decoder and SegVGGT's `einsum('bnd,bshwd->bnshw', ...)`.

Architecture is a faithful clone of VGGT-Omega's DenseHead DPT trunk (so we inherit its proven
multi-scale fusion), but the final projection emits `mask_dim` channels at 1/4 input resolution
instead of depth + a pixel-shuffle to full res.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ..paths import ensure_vggt_omega_on_path

ensure_vggt_omega_on_path()
# Reuse the exact DPT building blocks from the released backbone.
from vggt_omega.models.heads.dense_head import (  # noqa: E402
    _make_dense_resize_layer,
    _make_fusion_block,
    _make_scratch,
)
from vggt_omega.models.heads.utils import create_uv_grid, position_grid_to_embed  # noqa: E402


class MaskFeatureHead(nn.Module):
    """tokens_list -> per-frame mask feature map (B, S, mask_dim, H/4, W/4)."""

    def __init__(
        self,
        dim_in: int = 2048,
        patch_size: int = 16,
        features: int = 256,
        mask_dim: int = 256,
        out_channels: list[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: list[int] = [4, 11, 17, 23],
        out_ratio: int = 4,   # H/4 (segmenter baseline). 2 = H/2 (finer boundaries, more memory).
        anyup_enable: bool = False,      # replace the pyramid's UPSAMPLING resize layers with AnyUp
        anyup_ckpt: str | None = None,   # (RGB-guided). Pyramid resolutions / refinenet / output res all unchanged.
        anyup_qchunk: int | None = 8192,
        anyup_bf16: bool = True,         # run the upsampler under bf16 autocast (compute/activation heavy)
        anyup_grad_ckpt: bool = False,   # grad-checkpoint the upsampler only if bf16 alone still OOMs.
        uplift_pyr: bool = False,        # like but UPLiFT (not AnyUp) at the L0/L1 upsampling spots
        uplift_cfg: str | None = None,
        uplift_guidance_ds: int = 1,     # run UPLiFT at outsize/ds (cheaper RGB guidance) then bilinear to full
        uplift_compile: bool = False,    # torch.compile the UPLiFT modules
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.intermediate_layer_idx = intermediate_layer_idx
        self.mask_dim = mask_dim
        self.out_ratio = out_ratio   # final mask_feat at input/out_ratio (4 -> H/4)
        self.anyup_enable = anyup_enable
        self.uplift_pyr = uplift_pyr
        self.anyup_qchunk = anyup_qchunk
        self.anyup_bf16 = anyup_bf16
        self.uplift_guidance_ds = max(1, int(uplift_guidance_ds))
        self.uplift_compile = uplift_compile
        self.anyup_grad_ckpt = anyup_grad_ckpt

        self.norm = nn.LayerNorm(dim_in, eps=1e-5)
        self.projects = nn.ModuleList(
            [nn.Conv2d(dim_in, oc, kernel_size=1) for oc in out_channels]
        )
        # patch grid is at H/patch_size; finest fused scale = patch_grid * (patch_size/out_ratio) = H/out_ratio.
        # per-level resize factors halve down the pyramid: [ps/r, ps/2r, ps/4r, ps/8r] (segmenter: [4,2,1,0.5]).
        f0 = patch_size / out_ratio
        self.resize_factors = [f0 / (2 ** i) for i in range(4)]
        self.resize_layers = nn.ModuleList(
            [_make_dense_resize_layer(out_channels[i], self.resize_factors[i]) for i in range(4)]
        )
        self.scratch = _make_scratch(out_channels, features)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        # final projection to the mask-embedding space (H/out_ratio resolution)
        self.mask_proj = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.GroupNorm(32, features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, mask_dim, kernel_size=1),
        )

        # for each UPSAMPLING pyramid level (resize factor > 1, i.e. L0 4x and L1 2x), replace the
        # ConvTranspose resize with a trainable RGB-guided AnyUp. L2 (identity) and L3 (downsample) and
        # everything else (refinenet fusion, output H/4, decoder, loss) are UNCHANGED from segmenter.
        self.anyup_levels = [i for i, f in enumerate(self.resize_factors) if f > 1.0] if anyup_enable else []
        if anyup_enable:
            import os as _os, sys as _sys
            _root = _os.environ.get("ANYUP_ROOT", "/PATH/TO/codebase/anyup")
            if _root not in _sys.path:
                _sys.path.insert(0, _root)
            from anyup.model import AnyUp                      # pure-torch path (use_natten=False)
            self.anyups = nn.ModuleList()
            for i in self.anyup_levels:
                au = AnyUp(use_natten=False)
                if anyup_ckpt:
                    au.load_state_dict(torch.load(anyup_ckpt, map_location="cpu"), strict=True)
                self.anyups.append(au)
                self.resize_layers[i] = nn.Identity()          # AnyUp does this level's upsampling instead
            self.register_buffer("_imnet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("_imnet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # same placement as but UPLiFT (CVPR'26 linear Local Attender) instead of AnyUp,
        # one per upsampling level (L0 4x -> iters 2, L1 2x -> iters 1), bf16.
        elif uplift_pyr:
            import os as _os, sys as _sys, math as _math
            _root = _os.environ.get("UPLIFT_ROOT", "/PATH/TO/codebase/UPLiFT")
            if _root not in _sys.path:
                _sys.path.insert(0, _root)
            from omegaconf import OmegaConf
            from uplift.uplift import UPLiFT
            _cfg = OmegaConf.load(uplift_cfg)
            self.anyup_levels = [i for i, f in enumerate(self.resize_factors) if f > 1.0]
            self.uplifts = nn.ModuleList()
            for i in self.anyup_levels:
                iters = int(round(_math.log2(self.resize_factors[i])))   # 4x->2 steps, 2x->1 step
                u = UPLiFT(out_channels[i], patch_size, _cfg.uplift, low_mem=True)   # in_channels = this level's C
                u.enable_iterative(iters)
                if uplift_compile:                                        # compile (dynamic shapes from variable S/res)
                    u = torch.compile(u, dynamic=True)
                self.uplifts.append(u)
                self.resize_layers[i] = nn.Identity()
            self.register_buffer("_imnet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("_imnet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _apply_pos_embed(self, x: torch.Tensor, width: int, height: int, ratio: float = 0.1) -> torch.Tensor:
        pos = create_uv_grid(x.shape[-1], x.shape[-2], aspect_ratio=width / height, dtype=x.dtype, device=x.device)
        pos = position_grid_to_embed(pos, x.shape[1])
        pos = (pos * ratio).permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos

    def _scratch_forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        l1, l2, l3, l4 = feats
        l1 = self.scratch.layer1_rn(l1)
        l2 = self.scratch.layer2_rn(l2)
        l3 = self.scratch.layer3_rn(l3)
        l4 = self.scratch.layer4_rn(l4)
        out = self.scratch.refinenet4(l4, size=l3.shape[2:])
        out = self.scratch.refinenet3(out, l3, size=l2.shape[2:])
        out = self.scratch.refinenet2(out, l2, size=l1.shape[2:])
        out = self.scratch.refinenet1(out, l1, size=l1.shape[2:])
        return out

    def _anyup_resize(self, level_pos: int, x: torch.Tensor, images: torch.Tensor,
                      B: int, S: int, H: int, W: int, th: int, tw: int) -> torch.Tensor:
        """RGB-guided upsample of pyramid level `x` (B*S,C,ph,pw) -> (B*S,C,th,tw), replacing a
        ConvTranspose. Guidance = input RGB at the target (th,tw). Checkpointed in training."""
        au = self.anyups[level_pos]
        img_g = images.reshape(B * S, 3, H, W).float()
        img_g = (img_g - self._imnet_mean) / self._imnet_std
        if img_g.shape[-2:] != (th, tw):
            img_g = F.interpolate(img_g, size=(th, tw), mode="bilinear", align_corners=False)

        def _au(a, b):
            return au(a, b, output_size=(th, tw), q_chunk_size=self.anyup_qchunk)
        # bf16 autocast ~halves AnyUp's time & activation memory (its LFU 5D op + per-pixel attention are
        # the cost, not its tiny weights). Return fp32 so the rest of the (fp32) head stays consistent.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.anyup_bf16):
            if self.anyup_grad_ckpt and self.training:
                out = torch.utils.checkpoint.checkpoint(_au, img_g, x, use_reentrant=False)
            else:
                out = _au(img_g, x)
        return out.float()

    def _uplift_resize(self, level_pos: int, x: torch.Tensor, images: torch.Tensor,
                       B: int, S: int, H: int, W: int, th: int, tw: int) -> torch.Tensor:
        """same as _anyup_resize but UPLiFT (RGB-guided, bf16). Returns fp32 for the fp32 pyramid."""
        u = self.uplifts[level_pos]
        ds = self.uplift_guidance_ds
        gh, gw = max(1, th // ds), max(1, tw // ds)          # run UPLiFT at reduced res, then bilinear to full
        img_g = images.reshape(B * S, 3, H, W).float()
        img_g = (img_g - self._imnet_mean) / self._imnet_std
        if img_g.shape[-2:] != (gh, gw):
            img_g = F.interpolate(img_g, size=(gh, gw), mode="bilinear", align_corners=False)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.anyup_bf16):
            out = u(img_g, x, outsize=(gh, gw))
        if (gh, gw) != (th, tw):
            out = F.interpolate(out.float(), size=(th, tw), mode="bilinear", align_corners=False)
        return out.float()

    def forward(self, tokens_list: list, images: torch.Tensor, patch_token_start: int) -> torch.Tensor:
        B, S, _, H, W = images.shape
        ph, pw = H // self.patch_size, W // self.patch_size
        feats = []
        for i, layer_idx in enumerate(self.intermediate_layer_idx):
            x = tokens_list[layer_idx]
            if x is None:
                raise ValueError(f"backbone did not cache layer {layer_idx}")
            x = x[:, :, patch_token_start:].float()             # (B,S,Np,2048)
            x = x.reshape(B * S, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(B * S, x.shape[-1], ph, pw)
            x = self.projects[i](x)
            x = self._apply_pos_embed(x, W, H)
            if i in self.anyup_levels:                          # upsampling level -> learned RGB-guided upsampler
                fac = self.resize_factors[i]; pos = self.anyup_levels.index(i)
                th, tw = int(round(ph * fac)), int(round(pw * fac))
                x = (self._uplift_resize(pos, x, images, B, S, H, W, th, tw) if self.uplift_pyr
                     else self._anyup_resize(pos, x, images, B, S, H, W, th, tw))
            else:                                               # identity / downsample -> segmenter module
                x = self.resize_layers[i](x)
            feats.append(x)
        fused = self._scratch_forward(feats)                    # (B*S, features, H/out_ratio, W/out_ratio)
        fused = self._apply_pos_embed(fused, W, H)
        m = self.mask_proj(fused)                               # (B*S, mask_dim, H/out_ratio, W/out_ratio)
        mh, mw = m.shape[-2:]
        return m.reshape(B, S, self.mask_dim, mh, mw)
