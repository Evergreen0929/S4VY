"""Turn raw decoder outputs into instance masks.

Queries are already shared across frames -- one query is one 3D-consistent instance -- so decoding is
just a per-query objectness threshold plus a mask threshold, and the result is a 4D masklet.

The benchmark evaluation harness that produces the paper's numbers is not part of this release.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


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




# --- shared post-processing -------------------------------------------------
# These live here rather than in each entry point: resolution handling and edge
# snapping decide how good the released masks look, and three copies drifted apart.

def model_input_hw(h, w, long_side=512, mult=16):
    """Input size the backbone expects: longest side capped, both sides floored to `mult`."""
    if w >= h:
        wr, hr = long_side, round(h / w * long_side)
    else:
        wr, hr = round(w / h * long_side), long_side
    return max(mult, (hr // mult) * mult), max(mult, (wr // mult) * mult)


def masks_to_idmap(objectness, mask_logits, out_hw, score_thr=0.5, mask_thr=0.5, min_pixels=10):
    """Decode one batch element straight to a full-resolution instance id map (0 = background).

    The soft probabilities are upsampled and thresholded afterwards; thresholding first and
    upsampling the binary mask leaves visible stair-stepping at the output resolution.
    """
    Ho, Wo = out_hw
    scores, soft = decode_instances(objectness, mask_logits, score_thr=score_thr,
                                    mask_thr=mask_thr, min_pixels=min_pixels, return_soft=True)[0]
    S = mask_logits.shape[2]
    if soft.numel() == 0:
        return np.zeros((S, Ho, Wo), np.int32), np.zeros((0,), np.float32)
    up = (F.interpolate(soft.float().reshape(-1, 1, *soft.shape[-2:]), size=(Ho, Wo),
                        mode="bilinear", align_corners=False).reshape(soft.shape[0], S, Ho, Wo) > 0.5)
    up = up.cpu().numpy()
    order = np.argsort([m.sum() for m in up])                    # small first, large overwrites
    idmap = np.zeros((S, Ho, Wo), np.int32)
    for newid, qi in enumerate(order, start=1):
        idmap[up[qi]] = newid
    return idmap, scores.float().cpu().numpy()[order]


def _guided(I, p, r=4, eps=1e-3):
    """Gray-guided filter: snap soft mask p to the intensity edges of guide I (both HxW float[0,1])."""
    from scipy.ndimage import uniform_filter
    mI = uniform_filter(I, 2 * r + 1); mp = uniform_filter(p, 2 * r + 1)
    cov = uniform_filter(I * p, 2 * r + 1) - mI * mp
    var = uniform_filter(I * I, 2 * r + 1) - mI * mI
    a = cov / (var + eps); b = mp - a * mI
    return np.clip(uniform_filter(a, 2 * r + 1) * I + uniform_filter(b, 2 * r + 1), 0, 1)


def edge_snap(idmap, rgb):
    """Re-cut each instance in `idmap` along the image edges. -> refined id map (same shape)."""
    gray = rgb.astype(np.float32).mean(2) / 255.0
    ids = [int(i) for i in np.unique(idmap) if i > 0]
    if not ids:
        return idmap.astype(np.int32)
    ids.sort(key=lambda i: (idmap == i).sum())                   # small first, large overwrites
    best = np.full(idmap.shape, 0.5, np.float32)
    out = np.zeros(idmap.shape, np.int32)
    for i in ids:
        soft = _guided(gray, (idmap == i).astype(np.float32))
        soft = np.clip((soft - 0.30) / 0.45, 0, 1)               # mild contrast, keep a feather band
        m = soft > best
        out[m] = i; best[m] = soft[m]
    return out
