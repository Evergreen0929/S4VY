"""Set-prediction criterion: Hungarian match at mask-feature resolution, supervise masks at FULL
resolution by bilinear-upsampling the matched predictions and comparing against full-res GT with
shared column point-sampling. Objectness CE as usual; aux layers are re-matched (Mask2Former).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .matcher import HungarianMatcher


def _build_gt_masks(inst_ids: torch.Tensor, num_inst: List[int], hw):
    """inst_ids:(B,S,H,W) long (0=bg, 1..K). -> list len B of (K_b, S, h, w) float at mask resolution."""
    B, S, H, W = inst_ids.shape
    h, w = hw
    out = []
    for b in range(B):
        K = num_inst[b]
        if K == 0:
            out.append(torch.zeros(0, S, h, w, device=inst_ids.device))
            continue
        ids = inst_ids[b].float().unsqueeze(1)                          # (S,1,H,W)
        ids = F.interpolate(ids, size=(h, w), mode="nearest").squeeze(1).long()  # (S,h,w)
        masks = torch.stack([(ids == k + 1).float() for k in range(K)], 0)       # (K,S,h,w)
        out.append(masks)
    return out


def _sample_points(pred, gt, num_points):
    """pred:(M,P) logits, gt:(M,P) {0,1}; sample shared columns."""
    P = pred.shape[1]
    if num_points >= P:
        return pred, gt
    idx = torch.randperm(P, device=pred.device)[:num_points]
    return pred[:, idx], gt[:, idx]



class SetCriterion(nn.Module):
    def __init__(self, matcher: HungarianMatcher = None, w_mask=5.0, w_dice=5.0, w_obj=2.0,
                 eos_coef=0.1, num_points=12544, ignore_bg_subsets=None, pseudo_weight=0.5,
                 pseudo_no_exh_flip=False):
        super().__init__()
        self.matcher = matcher or HungarianMatcher(w_obj=w_obj, w_mask=w_mask, w_dice=w_dice, num_points=num_points)
        self.w_mask, self.w_dice, self.w_obj = w_mask, w_dice, w_obj
        self.num_points = num_points
        self.pseudo_weight = pseudo_weight     # per-instance weight multiplier for soft EMA-teacher pseudo instances
        # GATED (default False = existing behavior): when True, a pseudo batch is NOT flipped to exhaustive -> unmatched
        # queries keep the partial-label treatment (NO ∅/no-object penalty). Then teacher-filtered/missed real objects
        # are simply UNSUPERVISED (not pushed to background). ∅ signal then sources only from genuinely exhaustive data.
        self.pseudo_no_exh_flip = pseudo_no_exh_flip
        self.register_buffer("obj_weight", torch.tensor([1.0, eos_coef]))
        # subsets whose bg(0) is "unlabeled/unknown" (e.g. ScanNet200 projection holes) -> treat as
        # IGNORE in the mask loss (drop those pixels), so the model isn't penalised for segmenting a
        # real-but-unlabeled object there. Objectness (query-level ∅) is unaffected.
        self.ignore_bg_subsets = set(ignore_bg_subsets or [])

    def _instance_losses(self, objectness, mask_logits, gt_hi_list, indices, HW, valid_list=None, exhaustive_list=None, weight_list=None, pseudo_ignore_list=None):
        B, N, S, h, w = mask_logits.shape
        H, W = HW
        device = mask_logits.device
        tgt = torch.ones(B, N, dtype=torch.long, device=device)
        mask_bce = mask_logits.new_zeros(())
        mask_dice = mask_logits.new_zeros(())
        wsum = 0.0                                                       # sum of per-instance weights (real=1, pseudo=lambda*conf)
        for b in range(B):
            pi, gi = indices[b]
            if pi.numel() == 0:
                continue
            pi = pi.to(device); gi = gi.to(device)
            tgt[b, pi] = 0
            m = pi.numel()
            # bilinear-upsample the matched predicted masks to FULL res, then compare to full-res GT
            pred = mask_logits[b, pi].reshape(m * S, 1, h, w)
            pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False).reshape(m, S * H * W)
            gt = gt_hi_list[b][gi].reshape(m, S * H * W).to(pred.dtype)   # gt may be SOFT (pseudo) or hard (real) in [0,1]
            Pall = pred.shape[1]
            if self.num_points < Pall:
                idx = torch.randperm(Pall, device=pred.device)[: self.num_points]
                pred, gt = pred[:, idx], gt[:, idx]
            else:
                idx = None
            if valid_list is not None:                                   # drop IGNORE pixels (e.g. scannet holes)
                vb = valid_list[b].reshape(-1)
                if idx is not None:
                    vb = vb[idx]
                if not bool(vb.any()):
                    continue                                             # all-ignore for this instance -> skip
                pred, gt = pred[:, vb], gt[:, vb]
            if weight_list is None:
                # ---- hard-GT path: UNCHANGED (per-batch-element normalization, hard GT) ----
                mask_bce = mask_bce + F.binary_cross_entropy_with_logits(pred, gt, reduction="mean")
                prob = pred.sigmoid()
                dice = 1 - (2 * (prob * gt).sum(-1) + 1) / (prob.sum(-1) + gt.sum(-1) + 1)
                mask_dice = mask_dice + dice.mean()
                wsum += 1.0
            else:
                # ---- pseudo path: per-instance weighted (real=1, soft pseudo=lambda*conf) ----
                iw = weight_list[b].to(device)[gi]
                bce = F.binary_cross_entropy_with_logits(pred, gt, reduction="none").mean(-1)   # (m,)
                prob = pred.sigmoid()
                dice = 1 - (2 * (prob * gt).sum(-1) + 1) / (prob.sum(-1) + gt.sum(-1) + 1)      # (m,)
                mask_bce = mask_bce + (iw * bce).sum()
                mask_dice = mask_dice + (iw * dice).sum()
                wsum += float(iw.sum())
        denom = max(wsum, 1.0)
        # objectness CE; for PARTIAL-annotation samples (exhaustive=False) drop the no-object penalty on
        # unmatched queries (unlabeled real objects must NOT be pushed to "no-object" -> would teach suppression).
        ce = F.cross_entropy(objectness.reshape(-1, 2), tgt.reshape(-1), weight=self.obj_weight, reduction="none").reshape(B, N)
        ow = torch.ones(B, N, device=device)
        if exhaustive_list is not None:
            for b in range(B):
                if not bool(exhaustive_list[b]):
                    ow[b, tgt[b] == 1] = 0.0                              # partial: only matched queries supervise objectness
        # THREE-BAND: an UNMATCHED query whose predicted mask lies mostly in the IGNORE region (teacher mid-conf,
        # "maybe a real object") must NOT be pushed to ∅ -> drop its objectness penalty (else we suppress it).
        if pseudo_ignore_list is not None:
            for b in range(B):
                ig = pseudo_ignore_list[b]
                if ig is None:
                    continue
                igf = F.interpolate(ig.float().unsqueeze(1), size=(h, w), mode="nearest").squeeze(1).reshape(-1)  # (S*h*w,)
                pmb = (mask_logits[b].sigmoid() > 0.5).float().reshape(N, -1)                                     # (N,S*h*w)
                ov = (pmb * igf).sum(-1) / pmb.sum(-1).clamp(min=1)                                               # frac of pred in ignore
                ow[b, (tgt[b] == 1) & (ov > 0.5)] = 0.0
        loss_obj = (ce * ow).sum() / ow.sum().clamp(min=1)
        return {"loss_mask": self.w_mask * mask_bce / denom,
                "loss_dice": self.w_dice * mask_dice / denom,
                "loss_obj": self.w_obj * loss_obj}

    def _hr_mask_loss(self, mask_embed_final, mask_feat_hr, gt_hi_list, indices, HW, valid_list=None):
        """crisp mask loss through an end-branch upsampled feature (UPLiFT), matched queries ONLY
        (memory-bounded). Logits for matched query m = <mask_embed_m, mask_feat_hr> at the upsampled res,
        then BILINEAR-upsampled to full HW and compared to the full-res GT (same regime as the base loss,
        so is supervised at the same resolution as segmenter). Reuses the H/4 match `indices`."""
        B, S, d, Hh, Wh = mask_feat_hr.shape
        H, W = HW
        device = mask_feat_hr.device
        mask_bce = mask_feat_hr.new_zeros(())
        mask_dice = mask_feat_hr.new_zeros(())
        n_inst = 0
        for b in range(B):
            pi, gi = indices[b]
            if pi.numel() == 0:
                continue
            pi = pi.to(device); gi = gi.to(device)
            me_m = mask_embed_final[b, pi].to(mask_feat_hr.dtype)        # (m,d), match HR feature dtype (bf16)
            pred = torch.einsum("md,sdhw->mshw", me_m, mask_feat_hr[b])  # (m,S,Hh,Wh)
            pred = F.interpolate(pred.reshape(pi.numel() * S, 1, Hh, Wh).float(), size=(H, W),
                                 mode="bilinear", align_corners=False).reshape(pi.numel(), S * H * W)
            gt = gt_hi_list[b][gi].reshape(gi.numel(), S * H * W).to(pred.dtype)
            Pall = pred.shape[1]
            if self.num_points < Pall:
                idx = torch.randperm(Pall, device=pred.device)[: self.num_points]
                pred, gt = pred[:, idx], gt[:, idx]
            else:
                idx = None
            if valid_list is not None:
                vb = valid_list[b].reshape(-1)
                if idx is not None:
                    vb = vb[idx]
                if not bool(vb.any()):
                    continue
                pred, gt = pred[:, vb], gt[:, vb]
            mask_bce = mask_bce + F.binary_cross_entropy_with_logits(pred, gt, reduction="mean")
            prob = pred.sigmoid()
            dice = 1 - (2 * (prob * gt).sum(-1) + 1) / (prob.sum(-1) + gt.sum(-1) + 1)
            mask_dice = mask_dice + dice.mean()
            n_inst += 1
        denom = max(n_inst, 1)
        return {"loss_mask_hr": self.w_mask * mask_bce / denom,
                "loss_dice_hr": self.w_dice * mask_dice / denom}

    def forward(self, outputs: dict, batch: dict):
        mask_logits = outputs["mask_logits"]
        hw = mask_logits.shape[-2:]
        inst_ids = batch["inst_ids"]
        HW = inst_ids.shape[-2:]
        num = batch["num_instances"]
        gt_lo = _build_gt_masks(inst_ids, num, hw)          # for matching (cheap)
        gt_hi = _build_gt_masks(inst_ids, num, HW)          # full-res targets for the loss
        B0 = inst_ids.shape[0]
        # per-pixel IGNORE (mask-loss only): explicit batch['ignore'] (video partial/letterbox) AND bg-ignore subsets
        ign = batch.get("ignore")                                        # (B,S,H,W) bool, True=ignore  | None
        subs = batch.get("subset", [None] * B0)
        valid_list = None
        if ign is not None or self.ignore_bg_subsets:
            valid_list = []
            for b in range(B0):
                v = torch.ones_like(inst_ids[b], dtype=torch.bool)
                if ign is not None: v = v & (~ign[b].to(v.device).bool())
                if self.ignore_bg_subsets and subs[b] in self.ignore_bg_subsets: v = v & (inst_ids[b] != 0)
                valid_list.append(v)
        # exhaustive flag per sample (Seg4D partial datasets set False); default True (InsScene/dense)
        exh = batch.get("exhaustive")
        exhaustive_list = ([bool(x) for x in (exh.tolist() if torch.is_tensor(exh) else exh)] if exh is not None else None)
        # ---- (opt-in): augment partial GT with SOFT EMA-teacher pseudo instances. ----
        # Gated purely by batch['pseudo_masks'] being present; teacher batches never set it -> code below is a no-op.
        # POSITIVE-ONLY augmentation: pseudo adds extra objects the teacher found (richer positive supervision on
        # partial VOS clips) but does NOT flip the sample to exhaustive -- pseudo is a CONFIDENT SUBSET, not a
        # complete labeling, so forcing unmatched queries -> no-object would suppress real-but-unlabeled objects the
        # teacher missed (severe under-segmentation of unlabeled regions). The no-object/∅ signal stays sourced from
        # the genuinely exhaustive datasets in the mix (VIPSeg / 3D). exhaustive flag left as the dataset's own.
        weight_list = None
        pm = batch.get("pseudo_masks")                                   # None | list[B] of (P,S,H,W) soft [0,1]  (real precedence applied upstream)
        if pm is not None:
            pc = batch.get("pseudo_conf")                                # list[B] of (P,) confidence
            weight_list = []
            if exhaustive_list is None:
                exhaustive_list = [True] * B0
            for b in range(B0):
                nr = int(num[b])
                pmb = pm[b] if b < len(pm) else None
                if pmb is not None and pmb.shape[0] > 0:
                    P, S = pmb.shape[0], pmb.shape[1]
                    plo = F.interpolate(pmb.reshape(P * S, 1, *HW).float(), size=hw, mode="bilinear", align_corners=False)
                    plo = (plo.reshape(P, S, *hw) > 0.5).float()
                    gt_lo[b] = torch.cat([gt_lo[b], plo.to(gt_lo[b].device)], 0)
                    gt_hi[b] = torch.cat([gt_hi[b], pmb.float().to(gt_hi[b].device)], 0)   # soft target for pseudo
                    cb = (pc[b] if pc is not None else torch.ones(P))
                    w = torch.cat([torch.ones(nr), self.pseudo_weight * cb.float().cpu()])
                    if not self.pseudo_no_exh_flip:
                        exhaustive_list[b] = True                         # pseudo-completed -> apply no-object signal (default)
                    # else: leave exhaustive as the dataset's own (partial) -> filtered/missed objects NOT pushed to ∅
                else:
                    w = torch.ones(nr)
                weight_list.append(w)
        # THREE-BAND: teacher mid-confidence (0.3-0.7) mask union -> IGNORE region. (a) mask loss skips those pixels
        # (fold into valid_list); (b) objectness ∅ skips unmatched queries lying there (passed to _instance_losses).
        pig = batch.get("pseudo_ignore")                                  # None | list[B] of (S,H,W) bool
        pig_list = None
        if pig is not None:
            pig_list = [None] * B0
            if valid_list is None:
                valid_list = [torch.ones_like(inst_ids[b], dtype=torch.bool) for b in range(B0)]
            for b in range(B0):
                if b < len(pig) and pig[b] is not None:
                    ib = pig[b].to(valid_list[b].device).bool()
                    valid_list[b] = valid_list[b] & (~ib)
                    pig_list[b] = ib
        indices = self.matcher(outputs["objectness"], mask_logits, gt_lo)
        losses = self._instance_losses(outputs["objectness"], mask_logits, gt_hi, indices, HW, valid_list, exhaustive_list, weight_list, pig_list)
        for j, aux in enumerate(outputs.get("aux_outputs", [])):
            aux_idx = self.matcher(aux["objectness"], aux["mask_logits"], gt_lo)
            for k, v in self._instance_losses(aux["objectness"], aux["mask_logits"], gt_hi, aux_idx, HW, valid_list, exhaustive_list, weight_list, pig_list).items():
                losses[f"{k}_aux{j}"] = v
        # extra crisp mask loss through the UPLiFT end-branch feature (matched queries only)
        if "mask_feat_hr" in outputs and "mask_embed_final" in outputs:
            losses.update(self._hr_mask_loss(outputs["mask_embed_final"], outputs["mask_feat_hr"],
                                             gt_hi, indices, HW, valid_list))
        losses["loss_total"] = sum(losses.values())
        return losses, indices
