#!/usr/bin/env python
"""Shared critic card layout (v3.1), identical across all three data sources.

Two differences from the earlier layout:
  1. A and B share one frame set -- the intersection of frames where both candidates are
     present, uniformly subsampled to NFRAME in temporal order.
  2. Both candidates are drawn on that same frame set, so the critic compares like with
     like rather than two independently chosen views.
"""
import os

import cv2
import numpy as np
from PIL import Image

MAG = np.array([255, 0, 255], np.float32)
NFRAME = int(os.environ.get("NFRAME", "5"))
PERFRAME = os.environ.get("PERFRAME", "0") == "1"


def pick_frames(mA_by, mB_by, k=None):
    """A/B  id:  k

    """
    k = k or NFRAME
    okA = {int(f) for f, m in mA_by.items() if m is not None and m.any()}
    okB = {int(f) for f, m in mB_by.items() if m is not None and m.any()}
    cand = sorted(okA & okB)
    if len(cand) < k: cand = sorted(okA | okB)
    if len(cand) < k: cand = sorted({int(f) for f in list(mA_by) + list(mB_by)})
    if not cand: return [0] * k
    if len(cand) < k: return [cand[min(i, len(cand) - 1)] for i in range(k)]
    return [cand[i] for i in np.linspace(0, len(cand) - 1, k).round().astype(int)]


def n_common(mA_by, mB_by):
    okA = {int(f) for f, m in mA_by.items() if m is not None and m.any()}
    okB = {int(f) for f, m in mB_by.items() if m is not None and m.any()}
    return len(okA & okB)


def render_perframe(get_img, frames, mask_by, out_dir, key, tag):
    """ [ | ]  -> {out_dir}/img/{key}__{tag}{j}.jpg"""
    for j, f in enumerate(frames):
        im = np.asarray(get_img(int(f)))
        msk = mask_by.get(int(f))
        o = im.astype(np.float32).copy()
        if msk is not None:
            m = cv2.resize(np.asarray(msk).astype(np.float32), (im.shape[1], im.shape[0]),
                           interpolation=cv2.INTER_LINEAR) > 0.5
            o[m] = o[m] * 0.6 + MAG * 0.4
            cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(o, cont, -1, tuple(int(c) for c in MAG), 2)
        Image.fromarray(np.concatenate([im, o.astype(np.uint8)], axis=1)).save(
            f"{out_dir}/img/{key}__{tag}{j}.jpg", quality=90)
    return [int(f) for f in frames]


def dump_masks(out_dir, key, mA_by, mB_by):
    """ masklet  packbits , """
    os.makedirs(f"{out_dir}/masks", exist_ok=True)
    fr = sorted({int(f) for f in list(mA_by) + list(mB_by)})
    if not fr: return
    sh = None
    for d in (mA_by, mB_by):
        for m in d.values():
            if m is not None: sh = np.asarray(m).shape; break
        if sh: break
    if sh is None: return
    def stack(d):
        return np.packbits(np.stack([(np.asarray(d[f]).astype(bool) if d.get(f) is not None
                                      else np.zeros(sh, bool)) for f in fr]).reshape(len(fr), -1), axis=1)
    def _sh(d):
        for m in d.values():
            if m is not None: return np.asarray(np.asarray(m).shape, np.int32)
        return np.array(sh, np.int32)
    np.savez_compressed(f"{out_dir}/masks/{key}.npz", A=stack(mA_by), B=stack(mB_by),
                        frames=np.array(fr, np.int32), shape=np.array(sh, np.int32),
                        shapeA=_sh(mA_by), shapeB=_sh(mB_by))
