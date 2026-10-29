"""Depth + pose metrics, aligned with IGGT's eval conventions (iggt/metrics.py, iggt/utils/vo_eval.py).

Depth (median-scale-aligned, IGGT-style) PLUS extra diagnostics that catch the depth<->pose
SCALE-DECOUPLING that breaks multi-view fusion:
  - absrel / delta1 : after GLOBAL median scale alignment (relative-structure quality, like IGGT)
  - absrel_raw      : NO alignment (exposes absolute-scale drift vs GT)
  - scale_cv        : coefficient of variation of the per-frame scale median(gt_f)/median(pred_f)
                      across the frames of ONE scene. If depth is multi-view consistent this is ~0
                      (one global scale); if the SI loss let each frame float, it blows up -> no fuse.

Pose: compare two sets of extrinsics (world->cam, OpenCV). Used student-vs-teacher (clean, same
encoding convention -> measures geometry forgetting) and best-effort student-vs-GT. Camera centers
are sim3-aligned (Umeyama) before ATE; rotations compared by geodesic angle -> RRA/AUC.
All numpy/torch only (no evo dependency).
"""
from __future__ import annotations

import numpy as np
import torch


# ----------------------------------------------------------------------------- depth
@torch.no_grad()
def depth_metrics(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> dict:
    """pred/gt/valid: (S,H,W). Returns dict of floats (percentages where noted)."""
    pred = pred.detach().float().cpu().numpy()
    gt = gt.detach().float().cpu().numpy()
    valid = valid.detach().cpu().numpy().astype(bool) & (gt > 0) & (pred > 0)
    if valid.sum() < 100:
        return {}
    pv, gv = pred[valid], gt[valid]

    # global median alignment (IGGT default)
    scale = np.median(gv) / max(np.median(pv), 1e-9)
    pa = pv * scale
    absrel = float(np.mean(np.abs(pa - gv) / gv) * 100.0)
    ratio = np.maximum(pa / gv, gv / pa)
    delta1 = float(np.mean(ratio < 1.25) * 100.0)
    delta2 = float(np.mean(ratio < 1.25 ** 2) * 100.0)
    tau = float(np.mean(ratio < 1.03) * 100.0)          # IGGT's inlier ratio @1.03 (their "tau")

    # NO alignment -> absolute-scale drift
    absrel_raw = float(np.mean(np.abs(pv - gv) / gv) * 100.0)

    # per-frame scale consistency (the multi-view-fusion diagnostic)
    S = pred.shape[0]
    per_frame = []
    for s in range(S):
        m = valid[s]
        if m.sum() < 50:
            continue
        per_frame.append(np.median(gt[s][m]) / max(np.median(pred[s][m]), 1e-9))
    scale_cv = float(np.std(per_frame) / max(np.mean(per_frame), 1e-9)) if len(per_frame) > 1 else 0.0

    return {"depth_absrel": absrel, "depth_tau": tau, "depth_delta1": delta1, "depth_delta2": delta2,
            "depth_absrel_raw": absrel_raw, "depth_scale_cv": scale_cv,
            "depth_global_scale": float(scale)}


# ----------------------------------------------------------------------------- pose helpers
def _centers_and_R(extr: np.ndarray):
    """extr: (S,3,4) or (S,4,4) world->cam. -> centers (S,3) in world, R_wc (S,3,3)."""
    R = extr[:, :3, :3]
    t = extr[:, :3, 3]
    C = -np.einsum("sij,sj->si", np.transpose(R, (0, 2, 1)), t)   # camera center in world
    return C, R


def _umeyama(src: np.ndarray, dst: np.ndarray):
    """sim3 (s,R,t) aligning src->dst (both (N,3)). Returns aligned src."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = d0.T @ s0 / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (s0 ** 2).sum() / src.shape[0]
    scale = np.trace(np.diag(D) @ S) / max(var, 1e-12)
    t = mu_d - scale * R @ mu_s
    return (scale * (R @ src.T)).T + t


def _geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """per-frame angle (deg) between rotation matrices (S,3,3)."""
    Rrel = np.einsum("sij,skj->sik", Ra, Rb)              # Ra @ Rb^T
    tr = np.clip((np.trace(Rrel, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


@torch.no_grad()
def pose_metrics(pred_extr: torch.Tensor, ref_extr: torch.Tensor, tag: str = "") -> dict:
    """pred_extr / ref_extr: (S,3,4) or (S,4,4) world->cam. tag e.g. 'vsT' / 'vsGT'."""
    pe = pred_extr.detach().float().cpu().numpy()
    re = ref_extr.detach().float().cpu().numpy()
    if pe.ndim != 3 or pe.shape[0] < 2:
        return {}
    Cp, Rp = _centers_and_R(pe)
    Cr, Rr = _centers_and_R(re)
    # rotation: global-align by removing the first frame's relative rotation is unnecessary;
    # geodesic of R_pred R_ref^T is invariant to a shared world rotation only after alignment.
    # We align rotations by the optimal global rotation: use the same Umeyama R on centers is
    # translation-only, so for rotation report the raw relative angle (both share frame 0 world).
    rot_deg = _geodesic_deg(Rp, Rr)
    Cp_a = _umeyama(Cp, Cr)
    ate = float(np.sqrt(np.mean(np.sum((Cp_a - Cr) ** 2, axis=1))))
    # relative pose error (consecutive frames), translation part after sim3 align
    if pe.shape[0] >= 2:
        dp = np.linalg.norm(np.diff(Cp_a, axis=0), axis=1)
        dr = np.linalg.norm(np.diff(Cr, axis=0), axis=1)
        rpe_trans = float(np.sqrt(np.mean((dp - dr) ** 2)))
    else:
        rpe_trans = 0.0
    out = {
        f"pose_{tag}_ate": ate,
        f"pose_{tag}_rpe_trans": rpe_trans,
        f"pose_{tag}_rot_deg_mean": float(np.mean(rot_deg)),
        f"pose_{tag}_rra15": float(np.mean(rot_deg < 15) * 100.0),
        f"pose_{tag}_rra30": float(np.mean(rot_deg < 30) * 100.0),
    }
    # AUC@30 of rotation accuracy (area under accuracy-vs-threshold up to 30 deg)
    thr = np.arange(1, 31)
    acc = np.array([np.mean(rot_deg < t) for t in thr])
    out[f"pose_{tag}_auc30"] = float(acc.mean() * 100.0)
    return out
