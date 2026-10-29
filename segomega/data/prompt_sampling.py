"""Online prompt sampling for the promptable branch.

From a batch's GT instance-id maps (inst_ids: B,S,H,W; 0=bg, 1..K), sample per-object prompts on 1+ frames:
  - point: a positive pixel inside the object's mask (optionally a negative pixel outside), normalized (x,y).
  - box  : the object's tight bbox on a frame, JITTERED (loose/tight) to simulate imprecise boxes.
Returns model-ready prompt dicts + the per-object target masklet (S,H,W). Everything is ONLINE (cheap vs the
frozen 1B fwd) -> fresh prompts each step = free augmentation. Works for exhaustive AND non-exhaustive data
(binary this-object-vs-rest needs only the object's own mask; unlabeled bg is correctly 'not this object').
"""
from __future__ import annotations

from typing import List, Dict, Tuple

import torch


def _norm_xy(col: int, row: int, W: int, H: int) -> Tuple[float, float]:
    return (col + 0.5) / W, (row + 0.5) / H


def _rand_box(W: int, H: int):
    """A fully random 'garbage' box: uniform center, side 5-50% of image."""
    bw = (0.05 + 0.45 * torch.rand(()).item()) * W
    bh = (0.05 + 0.45 * torch.rand(()).item()) * H
    cx = torch.rand(()).item() * W
    cy = torch.rand(()).item() * H
    x0 = min(max(cx - bw / 2, 0), W - 2); x1 = min(max(cx + bw / 2, x0 + 1), W - 1)
    y0 = min(max(cy - bh / 2, 0), H - 2); y1 = min(max(cy + bh / 2, y0 + 1), H - 1)
    return x0 / W, y0 / H, x1 / W, y1 / H


def sample_prompts(inst_ids: torch.Tensor, num_instances: List[int],
                   max_objs: int = 4, p_box: float = 0.5, p_neg: float = 0.3,
                   max_prompt_frames=None, min_area: int = 32, single_frame_box: bool = False,
                   noise_aug: float = 0.0):
    """inst_ids:(B,S,H,W) long. Returns (prompts, targets):
       prompts:  list[B] of list[P] dicts {'points':[(f,x,y,label)], 'boxes':[(f,x0,y0,x1,y1)]}
       targets:  list[B] of FloatTensor (P,S,H,W) binary masklet per prompted object (aligned to prompts).
       noise_aug (v1lpn robustness): on MULTI-frame prompts only, with prob=noise_aug corrupt
       floor(U(0,0.2)*nf) of the chosen frames -- point: uniform random location (label kept positive);
       box: 70% another object's box on that frame / 30% garbage box. Target masklet stays the TRUE
       object, so the model must learn to down-weight inconsistent frames."""
    B, S, H, W = inst_ids.shape
    dev = inst_ids.device
    prompts: List[List[Dict]] = []
    targets: List[torch.Tensor] = []
    for b in range(B):
        K = int(num_instances[b])
        objs, tgts = [], []
        if K == 0:
            prompts.append([]); targets.append(inst_ids.new_zeros(0, S, H, W, dtype=torch.float32)); continue
        # candidate objects with enough total area
        cand = []
        masks = []
        for k in range(K):
            m = (inst_ids[b] == k + 1)                                   # (S,H,W) bool
            if int(m.sum()) >= min_area:
                cand.append(k); masks.append(m)
        if not cand:
            prompts.append([]); targets.append(inst_ids.new_zeros(0, S, H, W, dtype=torch.float32)); continue
        sel = torch.randperm(len(cand))[:max_objs].tolist()
        for si in sel:
            m = masks[si]                                                # (S,H,W) bool
            fp = [s for s in range(S) if int(m[s].sum()) > 0]
            if not fp:
                continue
            use_box = torch.rand(()).item() < p_box
            pts, boxes = [], []
            if use_box:
                # box prompt on a RANDOM number of present frames (1..present) = MULTIBOX (mirrors multipoint;
                # nf=1 -> single-frame box, nf>1 -> multi-frame boxes). SAM box-noise per frame.
                # single_frame_box=True -> -era behavior: box on ONE random present frame (v1 diagnostic:
                # v1's blind mean-pool of many big-box gaussian tokens ~ scene-average vector -> routing dies).
                if single_frame_box:
                    chosen = [fp[torch.randint(len(fp), ()).item()]]
                else:
                    hi = len(fp) if max_prompt_frames is None else min(int(max_prompt_frames), len(fp))
                    nf = torch.randint(1, hi + 1, ()).item()
                    chosen = [fp[i] for i in torch.randperm(len(fp))[:nf].tolist()]
                bad = set()
                if noise_aug > 0 and len(chosen) > 1 and torch.rand(()).item() < noise_aug:
                    n_bad = int(torch.rand(()).item() * 0.2 * len(chosen))          # floor of U(0,20%) of frames
                    bad = set(torch.randperm(len(chosen))[:n_bad].tolist())
                for ci, f in enumerate(chosen):
                    if ci in bad:
                        if torch.rand(()).item() < 0.7:                              # 70%: ANOTHER object's box
                            others = [mo for j, mo in enumerate(masks) if j != si and int(mo[f].sum()) > 0]
                            if others:
                                mo = others[torch.randint(len(others), ()).item()]
                                oys, oxs = torch.where(mo[f])
                                boxes.append((f, int(oxs.min()) / W, int(oys.min()) / H,
                                              max(int(oxs.max()), int(oxs.min()) + 1) / W,
                                              max(int(oys.max()), int(oys.min()) + 1) / H))
                                continue
                        boxes.append((f,) + _rand_box(W, H))                         # 30% (or fallback): garbage box
                        continue
                    ys, xs = torch.where(m[f])
                    x0, x1 = int(xs.min()), int(xs.max()); y0, y1 = int(ys.min()), int(ys.max())
                    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
                    sx, sy = min(0.1 * bw, 20.0), min(0.1 * bh, 20.0)   # SAM: gaussian std 10% side, cap 20px
                    nx0 = min(max(x0 + torch.randn(()).item() * sx, 0), W - 1)
                    ny0 = min(max(y0 + torch.randn(()).item() * sy, 0), H - 1)
                    nx1 = min(max(x1 + torch.randn(()).item() * sx, nx0 + 1), W - 1)
                    ny1 = min(max(y1 + torch.randn(()).item() * sy, ny0 + 1), H - 1)
                    boxes.append((f, nx0 / W, ny0 / H, nx1 / W, ny1 / H))
            else:
                # points on a RANDOM number of the object's present frames: uniform 1..len(fp)
                # (i.e. 1..S in the limit an object spans all frames). max_prompt_frames=None -> no cap.
                hi = len(fp) if max_prompt_frames is None else min(int(max_prompt_frames), len(fp))
                nf = torch.randint(1, hi + 1, ()).item()
                chosen = [fp[i] for i in torch.randperm(len(fp))[:nf].tolist()]
                bad = set()
                if noise_aug > 0 and len(chosen) > 1 and torch.rand(()).item() < noise_aug:
                    n_bad = int(torch.rand(()).item() * 0.2 * len(chosen))          # floor of U(0,20%) of frames
                    bad = set(torch.randperm(len(chosen))[:n_bad].tolist())
                for ci, f in enumerate(chosen):
                    if ci in bad:                                        # WRONG positive: uniform random location
                        pts.append((f, torch.rand(()).item(), torch.rand(()).item(), 1))
                        continue
                    ys, xs = torch.where(m[f])
                    j = torch.randint(len(xs), ()).item()
                    x, y = _norm_xy(int(xs[j]), int(ys[j]), W, H)
                    pts.append((f, x, y, 1))                             # positive
                    if torch.rand(()).item() < p_neg:
                        bg = torch.where(~m[f])
                        if len(bg[0]) > 0:
                            jb = torch.randint(len(bg[0]), ()).item()
                            xn, yn = _norm_xy(int(bg[1][jb]), int(bg[0][jb]), W, H)
                            pts.append((f, xn, yn, 0))                   # negative
            if not pts and not boxes:
                continue
            objs.append({"points": pts, "boxes": boxes})
            tgts.append(m.float())
        prompts.append(objs)
        targets.append(torch.stack(tgts, 0) if tgts else inst_ids.new_zeros(0, S, H, W, dtype=torch.float32))
    return prompts, targets


def prompt_mask_loss(prompt_mask_logits: torch.Tensor, targets: List[torch.Tensor],
                     num_points: int = 12544, w_mask: float = 5.0, w_dice: float = 5.0):
    """prompt_mask_logits:(B,P,S,h,w); targets: list[B] of (P,S,H,W) binary. Point-sampled BCE+dice,
       NO matching (prompt p <-> target p directly), over ALL frames (absent -> empty target = presence signal)."""
    import torch.nn.functional as F
    B, P, S, h, w = prompt_mask_logits.shape
    if P == 0:
        z = prompt_mask_logits.sum() * 0.0
        return {"loss_pmask": z, "loss_pdice": z, "loss_total": z}
    H, W = targets[0].shape[-2:] if targets[0].numel() else (h, w)
    bce = prompt_mask_logits.new_zeros(()); dice = prompt_mask_logits.new_zeros(()); n = 0
    for b in range(B):
        tb = targets[b]
        if tb.numel() == 0:
            continue
        m = tb.shape[0]
        pred = prompt_mask_logits[b, :m].reshape(m * S, 1, h, w)
        pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False).reshape(m, S * H * W)
        gt = tb.reshape(m, S * H * W).to(pred.dtype)
        Pall = pred.shape[1]
        if num_points < Pall:
            idx = torch.randperm(Pall, device=pred.device)[:num_points]
            pred, gt = pred[:, idx], gt[:, idx]
        bce = bce + F.binary_cross_entropy_with_logits(pred, gt, reduction="mean")
        prob = pred.sigmoid()
        dice = dice + (1 - (2 * (prob * gt).sum(-1) + 1) / (prob.sum(-1) + gt.sum(-1) + 1)).mean()
        n += 1
    denom = max(n, 1)
    lm, ld = w_mask * bce / denom, w_dice * dice / denom
    return {"loss_pmask": lm, "loss_pdice": ld, "loss_total": lm + ld}
