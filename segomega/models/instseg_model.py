"""Full model: VGGT-Omega backbone  ->  mask-feature pixel decoder + class-agnostic query decoder.

Returns instance predictions (per-frame masks shared by 3D-consistent queries) and, optionally,
the backbone's depth/camera predictions for joint geometry supervision (IGGT/SegVGGT style).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .backbone import VGGTOmegaBackbone
from .mask_feature_head import MaskFeatureHead
from .query_decoder import QueryInstanceDecoder


class SegOmega(nn.Module):
    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        num_queries: int = 400,
        num_dec_layers: int = 9,
        mask_dim: int = 256,
        dec_dim: int = 256,
        patch_size: int = 16,
        embed_dim: int = 1024,
        supervise_depth: bool = True,
        supervise_camera: bool = False,
        use_teacher: bool = True,        # frozen VGGT-Omega teacher for geometry anti-forgetting (SegVGGT-style)
        grad_checkpointing: bool = True, # checkpoint student aggregator blocks -> fit more frames/GPU
        freeze_backbone: bool = False,
        load_pretrained: bool = True,
        mask_feat_ratio: int = 4,        # H/4 mask features (segmenter baseline). 2 = H/2 (finer, more memory).
        anyup_enable: bool = False,      # replace the head's UPSAMPLING resize layers with AnyUp (RGB-guided)
        anyup_ckpt: Optional[str] = None,
        anyup_qchunk: Optional[int] = 8192,
        hr_upsampler: Optional[str] = None,   # end-branch upsampler on the final mask_feat ("uplift" | None)
        uplift_cfg: Optional[str] = None,     # path to a UPLiFT config yaml (architecture); shared by the UPLiFT options
        uplift_iters: int = 2,                # 2x steps: 2 -> H/4->full (128->512), 1 -> 128->256
        uplift_bf16: bool = True,
        uplift_pyr: bool = False,             # UPLiFT (bf16) at the head's L0/L1 upsampling spots
        uplift_guidance_ds: int = 1,          # UPLiFT at outsize/ds + bilinear to full
        uplift_compile: bool = False,         # torch.compile the UPLiFT modules
        enable_prompt: bool = False,          # SAM-style prompt encoder -> promptable object query (frozen base)
        prompt_arch: str = "v1",              # "v1lp" adds the learnable prompt pooling (the released recipe)
        prompt_init_gain_content: float = 1.0,  # v1 init-scale fix: x on content_mlp final layer (25 = 's basin)
        prompt_init_gain_pos: float = 1.0,      # v1 init-scale fix: x on pos_mlp final layer (17 = 's basin)
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.use_teacher = use_teacher
        # When distilling from a teacher we must run the student's depth+camera heads to have something
        # to distill into; so enable them whenever a teacher is present.
        self.supervise_depth = supervise_depth or use_teacher
        self.supervise_camera = supervise_camera or use_teacher

        self.backbone = VGGTOmegaBackbone(
            ckpt_path=ckpt_path,
            enable_depth=self.supervise_depth,
            enable_camera=self.supervise_camera,
            patch_size=patch_size,
            embed_dim=embed_dim,
            load_pretrained=load_pretrained,
        )
        if freeze_backbone:
            self.backbone.freeze()

        # gradient checkpointing on the STUDENT aggregator (after pretrained load; teacher left intact)
        if grad_checkpointing:
            from .grad_ckpt import enable_gradient_checkpointing
            enable_gradient_checkpointing(self.backbone.net.aggregator)

        # Frozen teacher: a pristine copy of VGGT-Omega whose depth/pose outputs supervise the
        # finetuned student, preserving the original geometry tasks (cf. SegVGGT's frozen-VGGT teacher).
        self.teacher = None
        if use_teacher:
            self.teacher = VGGTOmegaBackbone(
                ckpt_path=ckpt_path, enable_depth=True, enable_camera=True,
                patch_size=patch_size, embed_dim=embed_dim, load_pretrained=load_pretrained,
            )
            self.teacher.freeze()

        token_dim = 2 * embed_dim
        # AnyUp replaces the mask-feature head's UPSAMPLING resize layers (RGB-guided). Pyramid
        # resolutions / refinenet / output res / decoder / loss all unchanged vs segmenter.
        self.mask_feature_head = MaskFeatureHead(dim_in=token_dim, patch_size=patch_size, mask_dim=mask_dim,
                                                 out_ratio=mask_feat_ratio,
                                                 anyup_enable=anyup_enable, anyup_ckpt=anyup_ckpt,
                                                 anyup_qchunk=anyup_qchunk,
                                                 uplift_pyr=uplift_pyr, uplift_cfg=uplift_cfg,
                                                 uplift_guidance_ds=uplift_guidance_ds,
                                                 uplift_compile=uplift_compile)
        self.query_decoder = QueryInstanceDecoder(
            token_dim=token_dim, mask_dim=mask_dim, dec_dim=dec_dim,
            num_queries=num_queries, num_layers=num_dec_layers,
        )
        # promptable branch (only trained when enable_prompt; base stays frozen)
        self.prompt_encoder = None
        self.prompt_arch = prompt_arch
        if enable_prompt:
            from .prompt_encoder import PromptEncoder
            self.prompt_encoder = PromptEncoder(feat_dim=mask_dim, dec_dim=dec_dim,
                                                init_gain_content=prompt_init_gain_content,
                                                init_gain_pos=prompt_init_gain_pos,
                                                learnable_pool=(prompt_arch == "v1lp"))

        # ---- UPLiFT end-branch (linear Local Attender). Upsamples the FINAL H/4 mask_feat
        # directly (128 -> H/4*2^iters) RGB-guided; decoder + base losses unchanged. The crisp HR
        # logits <mask_embed_final, mask_feat_hr> feed a matched-only HR loss (in SetCriterion). ----
        self.hr_up = None
        self.uplift_iters = uplift_iters
        self.uplift_bf16 = uplift_bf16
        if hr_upsampler == "uplift":
            import os as _os, sys as _sys
            _root = _os.environ.get("UPLIFT_ROOT", "/PATH/TO/codebase/UPLiFT")
            if _root not in _sys.path:
                _sys.path.insert(0, _root)
            from omegaconf import OmegaConf
            from uplift.uplift import UPLiFT
            _cfg = OmegaConf.load(uplift_cfg)
            # low_mem=True: LocalAttender loops over offsets instead of materializing a giant
            # [B,C,D,H,I,W,I] tensor (which OOMs at 512^2). Required at our resolutions.
            self.hr_up = UPLiFT(mask_dim, patch_size, _cfg.uplift, low_mem=True)   # in_channels=256, random init
            self.hr_up.enable_iterative(uplift_iters)
            self.register_buffer("_imnet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("_imnet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor, amp: bool = True, prompts=None) -> dict:
        """images: (B,S,3,H,W) in [0,1]. prompts: optional list[B] of per-object prompt dicts;
        when given, a prompt query is encoded and decoded alongside the 400 (unidirectional), and
        preds gets prompt_mask_logits/prompt_objectness (+ aux)."""
        if images.dim() == 4:
            images = images.unsqueeze(0)
        B, S, _, H, W = images.shape
        ph, pw = H // self.patch_size, W // self.patch_size

        bb = self.backbone(images, run_depth=self.supervise_depth, run_camera=self.supervise_camera, amp=amp)
        tokens_list, pts = bb["tokens_list"], bb["patch_token_start"]

        # mask features + query decoding run in fp32 for stable mask logits
        with torch.autocast(device_type="cuda", enabled=False):
            mask_feat = self.mask_feature_head(tokens_list, images, pts)        # (B,S,mask_dim,h,w)
            pq = pq_pos = pq_valid = None
            if prompts is not None and self.prompt_encoder is not None:
                pq, pq_pos, pq_valid = self.prompt_encoder(mask_feat, prompts)
            preds = self.query_decoder(tokens_list, pts, mask_feat, (ph, pw),
                                       prompt_q=pq, prompt_q_pos=pq_pos, prompt_valid=pq_valid)
            # UPLiFT upsamples the final mask_feat 128 -> 128*2^iters (RGB-guided), bf16 for speed.
            if self.hr_up is not None:
                h, w = mask_feat.shape[-2:]
                oh, ow = h * (2 ** self.uplift_iters), w * (2 ** self.uplift_iters)
                img_g = images.reshape(B * S, 3, H, W).float()
                img_g = (img_g - self._imnet_mean) / self._imnet_std
                if img_g.shape[-2:] != (oh, ow):
                    img_g = torch.nn.functional.interpolate(img_g, size=(oh, ow), mode="bilinear", align_corners=False)
                feat_lo = mask_feat.reshape(B * S, mask_feat.shape[2], h, w).float()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.uplift_bf16):
                    feat_hr = self.hr_up(img_g, feat_lo, outsize=(oh, ow))
                # keep bf16 (no fp32 cast-back): the HR feature only feeds the matched-only HR loss, so
                # halving this big (S,256,512,512) tensor saves memory + time. Loss runs in bf16.
                preds["mask_feat_hr"] = feat_hr.reshape(B, S, mask_feat.shape[2], oh, ow)

        preds["mask_feat_hw"] = mask_feat.shape[-2:]
        if "depth" in bb:
            preds["depth"], preds["depth_conf"] = bb["depth"], bb["depth_conf"]
        if "pose_enc" in bb:
            preds["pose_enc"] = bb["pose_enc"]

        # frozen-teacher targets for geometry anti-forgetting (same convention/scale as the student head)
        if self.use_teacher and self.training:
            self.teacher.eval()
            with torch.no_grad():
                t = self.teacher(images, run_depth=True, run_camera=True, amp=amp)
            preds["teacher_pose_enc"] = t.get("pose_enc")
            preds["teacher_depth"] = t.get("depth")
        return preds
