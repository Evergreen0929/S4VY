#!/usr/bin/env python
"""box, : box -> box -> PICK  K -> s4vy_prompt masklet -> J&F


"""
import os
import numpy as np
import torch
from PIL import Image

FR_CAP = int(os.environ.get("FR_CAP", "48"))
PROMPT_K = int(os.environ.get("PROMPT_K", "5"))
PICK = os.environ.get("PICK", "area")


def pick_boxes(items, k=None, pick=None):
    """box K  prompt encoder presence_frac ,  area( §3)"""
    k = PROMPT_K if k is None else k
    pick = PICK if pick is None else pick
    if k <= 0 or len(items) <= k: return items
    if pick == "area":
        a = [(it[3] - it[1]) * (it[4] - it[2]) for it in items]
        return [items[int(i)] for i in sorted(np.argsort(-np.asarray(a))[:k])]
    return [items[i] for i in np.linspace(0, len(items) - 1, k).round().astype(int)]


def boxes_to_items(wins, bxall):
    """box -> box ( grounder_infer )"""
    bxwin = {}
    for win, bxs in zip(wins, bxall):
        for j, f in enumerate(win):
            bxwin.setdefault(int(f), []).append(bxs[j] if j < len(bxs) else None)
    items = []
    for f in sorted(bxwin):
        bs = [b for b in bxwin[f] if b is not None]
        if not bs: continue
        items.append([int(f)] + [float(np.median([b[k] for b in bs])) for k in range(4)])
    return items


def score_box_path(pm_model, imgs, items, gt, in_w, in_h, up, jf, tag="", k=None, return_mask=False):
    """items(box) -> masklet -> (J, F, n_box[, mask])gt  (T,H,W)  GT, up/jf

    mask,  4 """
    _ret = (lambda j, f, n, m=None: (j, f, n, m) if return_mask else (j, f, n))
    if not items: return _ret(0.0, 0.0, 0)
    S = len(imgs)
    prompts = pick_boxes(items, k=k)
    pf = sorted({int(it[0]) for it in prompts})
    rest = [f for f in range(S) if f not in set(pf)]
    room = max(1, FR_CAP - len(pf))
    chunks = [rest[i:i + room] for i in range(0, len(rest), room)] or [[]]
    mk = None
    for ch in chunks:
        fs = sorted(set(pf) | set(ch))
        loc = {fr: i for i, fr in enumerate(fs)}
        im = torch.stack([torch.from_numpy(
                np.asarray(Image.fromarray(imgs[fr]).resize((in_w, in_h)), dtype=np.float32) / 255.
             ).permute(2, 0, 1) for fr in fs]).unsqueeze(0).to(os.environ.get("SEG_DEV", "cuda"))
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pr = pm_model(im, prompts=[[{"points": [],
                     "boxes": [(loc[int(it[0])], *[float(x) for x in it[1:5]]) for it in prompts]}]])
            mkf = (pr["prompt_mask_logits"][0, 0].float() > 0).cpu().numpy()
        except Exception as e:
            print(f"{tag}: PROMPT FAIL {type(e).__name__}: {e}", flush=True)
            return _ret(0.0, 0.0, len(prompts))
        if mk is None: mk = np.zeros((len(gt),) + mkf.shape[1:], bool)
        for i, fr in enumerate(fs):
            if fr < len(gt): mk[fr] |= mkf[i]
    if mk is None: return _ret(0.0, 0.0, len(prompts))
    J, Fv = jf(up(mk, gt.shape[1:]), gt)
    return _ret(round(J or 0.0, 4), round(Fv or 0.0, 4), len(prompts), mk)
