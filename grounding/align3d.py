#!/usr/bin/env python
"""Segmenter adapter over SegVGGT's scene loader.

Builds S4VY on top of a VGGT-Omega backbone and runs it over frames loaded by SegVGGT, so masks
come back in the same full-FOV frame convention that `map_pred_inst_to_gt_pointcloud` expects for
lifting 2D masks onto the scene point cloud. SegVGGT is an external dependency: point SEGVGGT_ROOT
at a checkout of it.
"""
import os, sys, numpy as np
import _env, torch, torch.nn.functional as F, torch.nn as nn

SEGV = os.environ.get("SEGVGGT_ROOT", "/PATH/TO/codebase/SegVGGT")
sys.path.insert(0, SEGV); os.chdir(SEGV)
from eval.eval_instance_seg import load_single_scene_data, map_pred_inst_to_gt_pointcloud  # noqa: E402

DSF = int(os.environ.get("DSF", "1")); MASK_THR = float(os.environ.get("MASK_THR", "0.4"))
SCORE_THR = float(os.environ.get("SCORE_THR", "0.3"))


def build_omega():
    sys.path.insert(0, os.environ.get("SEG4D_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, _env.require("VGGT_OMEGA_ROOT", "VGGT-Omega code checkout"))
    from segomega.models import SegOmega, inject_lora, set_lora_enabled
    m = SegOmega(ckpt_path=_env.require("VGGT_OMEGA_CKPT", "VGGT-Omega backbone weights"), num_queries=400, num_dec_layers=9,
                 mask_dim=256, dec_dim=256, supervise_depth=True, supervise_camera=True,
                 use_teacher=False, grad_checkpointing=True, load_pretrained=True, mask_feat_ratio=4).to(os.environ.get("SEG_DEV", "cuda")).eval()
    m.backbone.freeze(); inject_lora(m.backbone.net.aggregator, rank=256, alpha=256.0); m=m.to(os.environ.get("SEG_DEV", "cuda")).eval()
    sd = torch.load(os.environ["INFER_CKPT"], map_location="cpu"); m.load_state_dict(sd.get("model", sd), strict=False)
    set_lora_enabled(m, True)
    from segomega.engine.decode import decode_instances
    return m, decode_instances


@torch.no_grad()
def omega_predict(model, decode, images_np):
    # images_np: (S,H,W,3) uint8/float from SegVGGT loader (full FOV). Resize to /16 valid size.
    S = images_np.shape[0]
    t = torch.from_numpy(images_np).float().div(255.0).permute(0, 3, 1, 2)   # (S,3,H,W)
    t = F.interpolate(t, size=(384, 512), mode="bilinear", align_corners=False).to(os.environ.get("SEG_DEV", "cuda"))
    fp = model.query_decoder.frame_pos
    if S > fp.num_embeddings:
        w = fp.weight.data; w_new = F.interpolate(w.t()[None], size=S, mode="linear", align_corners=False)[0].t().contiguous()
        ne = nn.Embedding(S, w.shape[1]).to(w.device); ne.weight.data = w_new; model.query_decoder.frame_pos = ne
    out = model(t)
    scores, masks = decode(out["objectness"], out["mask_logits"], score_thr=SCORE_THR)[0]
    return masks.float().cpu().numpy(), scores.cpu().numpy()   # masks (n,S,h,w)
