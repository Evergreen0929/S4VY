"""Lightweight instance-mask visualization (PIL + numpy only).

Builds a side-by-side panel per frame: [ RGB | GT instances | Predicted instances ], stacked over a
few frames, so you can eyeball whether masks are tight AND whether the SAME instance keeps its color
across frames (i.e. 3D-consistent). Used by the long single-GPU diagnostic run.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_PAL = None


def _palette(n: int):
    global _PAL
    if _PAL is None or len(_PAL) < n:
        rng = np.random.RandomState(42)
        p = rng.randint(40, 256, size=(max(n, 256), 3)).astype(np.uint8)
        p[0] = 0  # background = black
        _PAL = p
    return _PAL


def colorize(id_map: np.ndarray) -> np.ndarray:
    pal = _palette(int(id_map.max()) + 2)
    return pal[id_map.astype(np.int64) % len(pal)]


def overlay(rgb01: np.ndarray, id_map: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    rgb = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
    col = colorize(id_map)
    m = id_map > 0
    out = rgb.copy()
    out[m] = (alpha * col[m] + (1 - alpha) * rgb[m]).astype(np.uint8)
    return out


def pred_masks_to_idmap(masks: torch.Tensor, scores: torch.Tensor, H: int, W: int, S: int) -> np.ndarray:
    """masks:(m,S,h,w) bool, scores:(m,) -> (S,H,W) int id-map (higher score wins on overlap)."""
    idmap = np.zeros((S, H, W), dtype=np.int32)
    if masks.numel() == 0:
        return idmap
    # bilinear (not nearest) so upsampled mask edges are smooth -> no staircase/grid artifact;
    # threshold at 0.5 after interpolating the soft {0,1} mask.
    up = F.interpolate(masks.float().flatten(0, 1)[:, None], size=(H, W),
                       mode="bilinear", align_corners=False)
    up = up[:, 0].view(masks.shape[0], S, H, W) > 0.5
    order = torch.argsort(scores)  # ascending -> high score overwrites
    for rank, i in enumerate(order.tolist()):
        mi = up[i].cpu().numpy()
        idmap[mi] = rank + 1
    return idmap


def colorize_depth(depth: np.ndarray, vmin: float, vmax: float, valid: np.ndarray = None) -> np.ndarray:
    """depth HxW -> RGB uint8 via 'turbo' colormap, shared [vmin,vmax] normalization."""
    from matplotlib import cm
    d = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    rgb = (cm.get_cmap("turbo")(d)[..., :3] * 255).astype(np.uint8)
    if valid is not None:
        rgb[~valid] = 0  # mask invalid depth to black
    return rgb


def _norm_range(arr, valid=None):
    v = arr[valid] if (valid is not None and valid.any()) else arr.reshape(-1)
    return float(np.percentile(v, 2)), float(np.percentile(v, 98))


def save_depth_panel(images: torch.Tensor, pred_depth: torch.Tensor, gt_depth: torch.Tensor,
                     gt_valid: torch.Tensor, out_path: str, teacher_depth: torch.Tensor = None,
                     max_frames: int = 4):
    """images:(S,3,H,W) [0,1]; depths:(S,H,W); gt_valid:(S,H,W) bool.

    Columns: [ RGB | GT | Student (| Teacher) ]. Each depth map is colorized with ITS OWN per-map
    [2,98] percentile range (PER-COLUMN self-normalization), because VGGT-Omega depth is only up-to-scale
    -- so we compare relative STRUCTURE, not absolute scale. (A shared GT range would saturate the model
    outputs and look all-dark, which is just a scale artifact, not an error.)
    """
    S, _, H, W = images.shape
    rgb = images.permute(0, 2, 3, 1).cpu().numpy()
    gd = gt_depth.detach().float().cpu().numpy()
    gv = gt_valid.detach().cpu().numpy().astype(bool)
    sd = pred_depth.detach().float().cpu().numpy()
    td = teacher_depth.detach().float().cpu().numpy() if teacher_depth is not None else None
    fidx = list(range(0, S, max(1, S // max_frames)))[:max_frames]
    sep = np.full((H, 3, 3), 255, np.uint8)
    rows = []
    for f in fidx:
        cols = [(np.clip(rgb[f], 0, 1) * 255).astype(np.uint8)]
        vmin, vmax = _norm_range(gd[f], gv[f]); cols.append(colorize_depth(gd[f], vmin, vmax, gv[f]))
        vmin, vmax = _norm_range(sd[f]);        cols.append(colorize_depth(sd[f], vmin, vmax))
        if td is not None:
            vmin, vmax = _norm_range(td[f]);    cols.append(colorize_depth(td[f], vmin, vmax))
        row = cols[0]
        for c in cols[1:]:
            row = np.concatenate([row, sep, c], axis=1)
        rows.append(row)
    hsep = np.full((3, rows[0].shape[1], 3), 255, np.uint8)
    grid = rows[0]
    for r in rows[1:]:
        grid = np.concatenate([grid, hsep, r], axis=0)
    Image.fromarray(grid).save(out_path)
    return out_path


def save_panel(images: torch.Tensor, gt_ids: torch.Tensor, pred_idmap: np.ndarray,
               out_path: str, max_frames: int = 4):
    """images:(S,3,H,W) [0,1], gt_ids:(S,H,W) int, pred_idmap:(S,H,W) int."""
    S, _, H, W = images.shape
    fidx = list(range(0, S, max(1, S // max_frames)))[:max_frames]
    rgb = images.permute(0, 2, 3, 1).cpu().numpy()
    gt = gt_ids.cpu().numpy()
    rows = []
    for f in fidx:
        a = (np.clip(rgb[f], 0, 1) * 255).astype(np.uint8)
        b = overlay(rgb[f], gt[f])
        c = overlay(rgb[f], pred_idmap[f])
        sep = np.full((H, 3, 3), 255, np.uint8)
        rows.append(np.concatenate([a, sep, b, sep, c], axis=1))
    hsep = np.full((3, rows[0].shape[1], 3), 255, np.uint8)
    grid = rows[0]
    for r in rows[1:]:
        grid = np.concatenate([grid, hsep, r], axis=0)
    Image.fromarray(grid).save(out_path)
    return out_path
