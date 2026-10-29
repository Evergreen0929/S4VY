"""Optional branch: online EMA teacher + soft pseudo-label generation for non-exhaustive
(partial-label) 4D data. Fully gated by cfg.train.ema_pseudo -- unused by teacher.

EMA teacher = EMA of the student's TRAINABLE params (LoRA + heads); the frozen backbone is shared, so
the teacher forward just swaps the trainable params into the same model (no second 1B backbone in memory).
Teacher runs no_grad + bf16, only on non-exhaustive batches, BEFORE the student fwd/bwd (peaks don't stack).

generate_pseudo(): from the teacher's confident instances, keep those NOT overlapping any real GT object
(real-GT precedence, IoU<thr) as SOFT pseudo instances (mask probs + confidence). These are fed to
SetCriterion via batch['pseudo_masks']/['pseudo_conf'] -> soft, confidence-weighted supervision + the
gained no-object (varnothing) signal. Duplicates of real objects are dropped (the criterion's exhaustive
no-object term then suppresses the leftover duplicate queries)."""
import os
import torch
import torch.nn.functional as F


class EMATeacher:
    def __init__(self, model, momentum=0.999, init_ckpt=None, teacher_framepos=False):
        self.m = momentum
        # teacher_framepos: force decoder frame_pos ON during the teacher forward. TRUE only when the teacher was
        # TRAINED WITH frame_pos (warm-start from teacher). A self-EMA teacher inherits the student's (no) frame_pos
        # -> must be FALSE, else it uses the never-trained frame_pos.weight and emits garbage pseudo.
        self.tfp = teacher_framepos
        self.params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]   # LoRA + heads
        if init_ckpt:
            # WARM START: init teacher from a CONVERGED ckpt (e.g. teacher) instead of the cold student.
            # A cold self-teacher produced degenerate near-full-frame pseudo masks -> collapse; a healthy
            # teacher (teacher) yields clean object-scale pseudo from step 0.
            import torch as _t
            sd = _t.load(init_ckpt, map_location="cpu"); sd = sd.get("model", sd)
            miss = [n for n, _ in self.params if n not in sd]
            self.ema = {n: (sd[n].detach().to(p.device, p.dtype).clone() if n in sd else p.detach().clone())
                        for n, p in self.params}
            print(f"[ema] WARM-START teacher from {init_ckpt}: matched {len(self.params)-len(miss)}/{len(self.params)}"
                  f"{' (MISSING '+str(len(miss))+')' if miss else ''}", flush=True)
        else:
            self.ema = {n: p.detach().clone() for n, p in self.params}

    @torch.no_grad()
    def reset_from(self, model):
        """Snapshot the CURRENT student as the teacher start (delayed self-teacher: called once at ema_update_start
        so the teacher begins from a competent student, not the cold init -> avoids cold-teacher pseudo collapse
        WITHOUT needing an external warm-start ckpt)."""
        for n, p in model.named_parameters():
            if n in self.ema:
                self.ema[n].copy_(p.detach())
        print(f"[ema] reset teacher <- current student ({len(self.ema)} params)", flush=True)

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.ema:
                self.ema[n].mul_(self.m).add_(p.detach(), alpha=1 - self.m)

    @torch.no_grad()
    def forward_pseudo(self, model, images):
        """Swap EMA trainable params in, no_grad bf16 forward, swap student params back. Returns teacher out."""
        backup = {n: p.detach().clone() for n, p in self.params}
        for n, p in self.params:
            p.data.copy_(self.ema[n])
        was_train = model.training
        model.eval()
        # If the teacher was TRAINED WITH decoder frame_pos (warm-start from teacher) while the STUDENT trains with it
        # dropped (SEG_NO_FRAMEPOS=1), force frame_pos back ON for the teacher forward ONLY -- else the teacher teacher
        # runs off-distribution. For a self-EMA teacher (tfp=False) this is skipped (teacher has no trained frame_pos).
        _nofp = _fmode = None
        if self.tfp:
            _nofp = os.environ.pop("SEG_NO_FRAMEPOS", None)
            _fmode = os.environ.pop("SEG_FRAMEPOS_MODE", None)   # default "embed" -> teacher uses its learned nn.Embedding
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(images)
        finally:
            if _nofp is not None: os.environ["SEG_NO_FRAMEPOS"] = _nofp
            if _fmode is not None: os.environ["SEG_FRAMEPOS_MODE"] = _fmode
        for n, p in self.params:
            p.data.copy_(backup[n])
        if was_train:
            model.train()
        return out


@torch.no_grad()
def generate_pseudo(teacher_out, inst_ids, num_inst, obj_thr=0.5, iou_thr=0.5, min_area=25, topk=25,
                    max_area_frac=0.5, three_band=False, ignore_lo=0.3):
    """teacher_out: {objectness (1,N,2), mask_logits (1,N,S,h,w)}; inst_ids (1,S,H,W) real GT.
    -> (pseudo_masks (P,S,H,W) soft float, conf (P,), ignore_mask (S,H,W) bool | None) on GT device.
    Keeps confident teacher instances that do NOT overlap any real GT object (real precedence).
    THREE-BAND (three_band=True): obj>obj_thr = POSITIVE pseudo (supervise); ignore_lo<obj<=obj_thr = IGNORE
    (their mask union returned -> student neither supervised nor ∅-penalised there); obj<=ignore_lo -> nothing
    (student ∅ suppresses over-emission). three_band=False keeps the legacy 2-band behaviour (ignore_mask=None)."""
    obj = teacher_out["objectness"][0].float().softmax(-1)[:, 0]         # (N,) P(object)
    ml = teacher_out["mask_logits"][0].float()                           # (N,S,h,w)
    N, S, h, w = ml.shape
    dev = inst_ids.device
    H, W = inst_ids.shape[-2:]
    # --- IGNORE band (three_band only): union of mid-confidence teacher masks -> "maybe a real object" regions ---
    ign_mask = None
    if three_band:
        ign_keep = (obj > ignore_lo) & (obj <= obj_thr)
        if ign_keep.any():
            ib = (ml[torch.where(ign_keep)[0]].sigmoid() > 0.5)          # (Q,S,h,w)
            iarea = ib.flatten(1).sum(1); ifrac = iarea.float() / float(S * h * w)
            ib = ib[(iarea > min_area) & (ifrac < max_area_frac)]        # drop specks / full-frame blobs
            if ib.numel() > 0:
                u = F.interpolate(ib.any(0).float().unsqueeze(1), size=(H, W), mode="nearest").squeeze(1)
                ign_mask = (u > 0.5).to(dev)                             # (S,H,W) bool
    keep = obj > obj_thr
    if keep.sum() == 0:
        return None, None, ign_mask
    idx = torch.where(keep)[0]
    prob_lo = ml[idx].sigmoid()                                          # (P,S,h,w) at mask res (for cheap IoU)
    binm = (prob_lo > 0.5)
    area = binm.flatten(1).sum(1)
    frac = area.float() / float(S * h * w)                               # avg per-frame coverage
    # reject BOTH tiny specks (< min_area) AND degenerate near-full-frame blobs (>max_area_frac) -- the latter
    # are not objects and, kept as pseudo, collapse the student toward whole-image masks (bug: fg~0.96).
    ok = (area > min_area) & (frac < max_area_frac)
    idx, prob_lo, binm = idx[ok], prob_lo[ok], binm[ok]
    if idx.numel() == 0:
        return None, None, ign_mask
    # real GT masks at mask res (h,w) for IoU vs pseudo
    K = int(num_inst)
    if K > 0:
        rids = F.interpolate(inst_ids[0].float().unsqueeze(1), size=(h, w), mode="nearest").squeeze(1).long()  # (S,h,w)
        real = torch.stack([(rids == k + 1) for k in range(K)], 0)       # (K,S,h,w) bool
        rflat = real.flatten(1).float()                                  # (K, S*h*w)
        pflat = binm.flatten(1).float()                                  # (P, S*h*w)
        inter = pflat @ rflat.t()                                        # (P,K)
        union = pflat.sum(1, keepdim=True) + rflat.sum(1) - inter        # (P,K)
        iou = inter / union.clamp(min=1)
        drop = (iou.max(1).values > iou_thr) if K > 0 else torch.zeros(len(idx), dtype=torch.bool, device=ml.device)
        keep2 = ~drop
        idx, prob_lo = idx[keep2], prob_lo[keep2]
    if idx.numel() == 0:
        return None, None, ign_mask
    conf = obj[idx]
    if idx.numel() > topk:                                               # cap: most-confident pseudo only
        sel = torch.topk(conf, topk).indices
        idx, conf = idx[sel], conf[sel]
    # upsample the SOFT masks of the kept pseudo to full res (for the criterion's soft target)
    up = F.interpolate(ml[idx].sigmoid().reshape(-1, 1, h, w), size=(H, W), mode="bilinear", align_corners=False)
    pseudo = up.reshape(idx.numel(), S, H, W).to(dev)
    return pseudo, conf.to(dev), ign_mask
