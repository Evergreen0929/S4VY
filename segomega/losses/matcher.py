"""Hungarian matcher for class-agnostic, multi-frame instance masks.

Cost(query, gt_instance) = w_obj * (-P(object)) + w_mask * BCE(masks) + w_dice * Dice(masks),
with masks compared over the *whole* (S,h,w) volume (so a query is matched to a 3D-consistent GT
instance, not a per-frame blob).  Mask costs are evaluated on a random subset of points for speed
(standard Mask2Former trick).  This file (and criterion.py) reimplements what SegVGGT's eval-only
release omits, following Mask2Former conventions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def _pairwise_mask_costs(pred_logits: torch.Tensor, gt: torch.Tensor):
    """pred_logits:(N,P), gt:(G,P) in {0,1}. Returns (bce_cost(N,G), dice_cost(N,G))."""
    P = pred_logits.shape[1]
    pos = F.binary_cross_entropy_with_logits(pred_logits, torch.ones_like(pred_logits), reduction="none")
    neg = F.binary_cross_entropy_with_logits(pred_logits, torch.zeros_like(pred_logits), reduction="none")
    bce = (pos @ gt.t() + neg @ (1 - gt).t()) / P                       # (N,G)

    prob = pred_logits.sigmoid()
    num = 2 * (prob @ gt.t())                                            # (N,G)
    den = prob.sum(-1, keepdim=True) + gt.sum(-1)[None, :]               # (N,G)
    dice = 1 - (num + 1) / (den + 1)
    return bce, dice


class HungarianMatcher:
    def __init__(self, w_obj=2.0, w_mask=5.0, w_dice=5.0, num_points=12544):
        self.w_obj, self.w_mask, self.w_dice = w_obj, w_mask, w_dice
        self.num_points = num_points

    @torch.no_grad()
    def __call__(self, objectness, mask_logits, gt_masks_list):
        """objectness:(B,N,2)  mask_logits:(B,N,S,h,w)  gt_masks_list: list len B of (G_b,S,h,w) float.

        Returns list of (pred_idx LongTensor, gt_idx LongTensor) per batch element.
        """
        B, N = objectness.shape[:2]
        results = []
        for b in range(B):
            gt = gt_masks_list[b]
            G = gt.shape[0]
            if G == 0:
                results.append((torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)))
                continue
            pred = mask_logits[b].flatten(1)                            # (N, S*h*w)
            gtf = gt.flatten(1).to(pred.dtype)                          # (G, S*h*w)
            # sample shared points
            Pall = pred.shape[1]
            if self.num_points < Pall:
                idx = torch.randperm(Pall, device=pred.device)[: self.num_points]
                pred, gtf = pred[:, idx], gtf[:, idx]
            bce, dice = _pairwise_mask_costs(pred, gtf)
            prob_obj = objectness[b].softmax(-1)[:, 0:1]                # P(object) (N,1)
            cost = self.w_mask * bce + self.w_dice * dice - self.w_obj * prob_obj
            cost = torch.nan_to_num(cost, nan=1e4, posinf=1e4, neginf=-1e4).cpu().numpy()
            r, c = linear_sum_assignment(cost)
            results.append((torch.as_tensor(r, dtype=torch.long), torch.as_tensor(c, dtype=torch.long)))
        return results
