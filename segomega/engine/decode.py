"""Turn raw decoder outputs into instance masks.

Queries are already shared across frames -- one query is one 3D-consistent instance -- so decoding is
just a per-query objectness threshold plus a mask threshold, and the result is a 4D masklet.

The benchmark evaluation harness that produces the paper's numbers is not part of this release.
"""
from __future__ import annotations

import torch


def decode_instances(objectness, mask_logits, score_thr=0.5, mask_thr=0.5, min_pixels=10, return_soft=False):
    """-> per batch element: (scores[m], masks[m,S,h,w]). masks bool by default; if return_soft, the soft
    sigmoid probs of the kept instances (for crisp viz: upsample SOFT, threshold at display res)."""
    B, N = objectness.shape[:2]
    scores_all = objectness.softmax(-1)[..., 0]                 # P(object)
    out = []
    for b in range(B):
        keep = scores_all[b] > score_thr
        prob = mask_logits[b].sigmoid()
        masks = (prob > mask_thr) & keep.view(N, 1, 1, 1)
        sizes = masks.flatten(1).sum(-1)
        keep = keep & (sizes > min_pixels)
        out.append((scores_all[b][keep], (prob[keep] if return_soft else masks[keep])))
    return out


def hr_mask_logits(preds):
    """Full per-query high-res mask logits from the UPLiFT end-branch feature,
    <mask_embed_final, mask_feat_hr>. Returns (B,N,S,Hh,Wh) or None when no HR branch. Visualization only."""
    if "mask_feat_hr" not in preds or "mask_embed_final" not in preds:
        return None
    return torch.einsum("bnd,bsdhw->bnshw", preds["mask_embed_final"].float(), preds["mask_feat_hr"].float())


def viz_idmap(preds, H, W, S, score_thr=0.5):
    """Display id-map from the BEST mask available (UPLiFT HR if present, else H/4), upsampling
    SOFT probabilities with bilinear (fixes the blocky binarize-then-upsample look in saved panels)."""
    from ..utils.viz import pred_masks_to_idmap
    ml = hr_mask_logits(preds)
    if ml is None:
        ml = preds["mask_logits"]
    scores, soft = decode_instances(preds["objectness"], ml, score_thr=score_thr, return_soft=True)[0]
    return pred_masks_to_idmap(soft, scores, H, W, S)


