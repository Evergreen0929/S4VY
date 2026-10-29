#!/usr/bin/env python
"""Critic card layout: side-by-side A/B per frame, frames chosen by A/B disagreement.

Earlier layouts asked the model to pair frame j with frame k+j as the same timestamp, or
dropped the pairing entirely. Compositing both candidates into one image per frame removes
that burden, and weighting frame choice by disagreement puts the decision where the two
candidates actually differ.

Env: SRC, DST, NFRAME (8), SHARD / NSHARD, FRAMESEL (disagreement).
"""
import glob
import json
import os
import _env
import sys

import cv2
import numpy as np
from PIL import Image

AG = os.environ.get("GROUNDING_REPO", os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, AG)
D = os.environ.get("GROUNDING_DATA_ROOT", "/PATH/TO/data")
SRC = _env.require("SRC", "source card directory"); DST = _env.require("DST", "destination directory")
K = int(os.environ.get("NFRAME", "8"))
SEL = os.environ.get("FRAMESEL", "disagree")
SHARD = int(os.environ.get("SHARD", "0")); NSHARD = int(os.environ.get("NSHARD", "1"))
NORAW = os.environ.get("NORAW", "0") == "1"
os.makedirs(f"{DST}/img", exist_ok=True)
MAG = np.array([255, 0, 255], np.float32)
CYA = np.array([0, 255, 255], np.float32)
SC2D = os.environ.get("SC2D", os.environ.get("SCANNET200_2D_ROOT", "/PATH/TO/codebase/SegVGGT/data_processing/scannet200/scenes_2d"))
SPPP_2D = f"{D}/scannetpp_scenes_2d"
VID2D = os.environ.get("VID2D", f"{D}/vid_scenes_2d")


def frame_reader(key, scene, ds):
    if ds in ("mevis", "lvos", "refdavis"):
        vid = key.split("_", 1)[1].split("__")[0]
        return lambda f: np.asarray(Image.open(f"{VID2D}/{vid}/color/{int(f)}.jpg").convert("RGB"))
    if ds == "scannetpp":
        def _g(f):
            im = Image.open(f"{SPPP_2D}/{scene}/color/{int(f)}.jpg").convert("RGB")
            if im.width > 512: im = im.resize((512, int(im.height * 512 / im.width)))
            return np.asarray(im)
        return _g
    return lambda f: np.asarray(Image.open(f"{SC2D}/{scene}/color/{int(f)}.jpg").convert("RGB"))


def pick(frames, mA, mB, k):
    """ A/B () k ,

    """
    n = len(frames)
    dis = (mA ^ mB).reshape(n, -1).sum(1).astype(np.int64)
    if SEL == "uniform" or dis.max() == 0:
        idx = np.linspace(0, n - 1, min(k, n)).round().astype(int)
    else:
        idx = []
        for b in range(k):
            lo, hi = int(b * n / k), int((b + 1) * n / k)
            if hi <= lo: continue
            j = lo + int(np.argmax(dis[lo:hi]))
            if dis[j] > 0: idx.append(j)
        rest = [int(j) for j in np.argsort(-dis) if int(j) not in idx]
        idx = sorted(idx + rest[:max(0, k - len(idx))])
        if not idx: idx = list(np.linspace(0, n - 1, min(k, n)).round().astype(int))
    idx = [int(x) for x in idx][:k]
    while len(idx) < k: idx.append(idx[-1] if idx else 0)
    return [int(frames[i]) for i in idx], [int(dis[i]) for i in idx]


def overlay(im, m, col):
    o = im.astype(np.float32).copy()
    if m is not None:
        mm = cv2.resize(np.asarray(m).astype(np.float32), (im.shape[1], im.shape[0]),
                        interpolation=cv2.INTER_LINEAR) > 0.5
        o[mm] = o[mm] * 0.6 + col * 0.4
        cont, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(o, cont, -1, tuple(int(c) for c in col), 2)
    return o.astype(np.uint8)


def main():
    rows = [json.loads(l) for f in sorted(glob.glob(f"{SRC}/meta_*.jsonl")) for l in open(f)]
    rows = [r for i, r in enumerate(rows) if i % NSHARD == SHARD]
    meta = open(f"{DST}/meta_{SHARD:02d}.jsonl", "a")
    print(f"[pairframe] {SRC} -> {DST} K={K} SEL={SEL} this shard {len(rows)}", flush=True)
    ok = fail = 0
    for i, r in enumerate(rows):
        key = r["key"]
        if os.path.exists(f"{DST}/img/{key}__Q{K-1}.jpg"): continue
        mp = f"{SRC}/masks/{key}.npz"
        if not os.path.exists(mp): fail += 1; continue
        try:
            z = np.load(mp); sh = tuple(int(x) for x in z["shape"])
            fr = z["frames"] if "frames" in z.files else np.arange(z["A"].shape[0], dtype=np.int32)
            def un(a):
                bits = a.shape[1] * 8
                s2 = sh if int(np.prod(sh)) <= bits else (96, 128)
                return np.unpackbits(a, axis=1)[:, :int(np.prod(s2))].reshape(len(fr), *s2).astype(bool)
            mA, mB = un(z["A"]), un(z["B"])
            if mA.shape != mB.shape:
                mB = np.stack([cv2.resize(x.astype(np.float32), (mA.shape[2], mA.shape[1])) > .5 for x in mB])
            frs, dis = pick(list(fr), mA, mB, K)
            loc = {int(f): j for j, f in enumerate(fr)}
            gi = frame_reader(key, r["scene"], r.get("ds", "scanrefer"))
            for j, f in enumerate(frs):
                im = gi(f); a = overlay(im, mA[loc[f]], MAG); b = overlay(im, mB[loc[f]], CYA)
                pan = ([a, b], [b, a]) if NORAW else ([im, a, b], [im, b, a])
                Image.fromarray(np.concatenate(pan[0], axis=1)).save(f"{DST}/img/{key}__P{j}.jpg", quality=90)
                Image.fromarray(np.concatenate(pan[1], axis=1)).save(f"{DST}/img/{key}__Q{j}.jpg", quality=90)
        except Exception as e:
            print(f"  {key}: FAIL {type(e).__name__}: {e}", flush=True); fail += 1; continue
        o = dict(r); o["fmt"] = (f"pfab{K}" if NORAW else f"pairframe{K}")
        o["viewsA"] = frs; o["viewsB"] = frs; o["disagree"] = dis
        meta.write(json.dumps(o) + "\n"); meta.flush(); ok += 1
        if (i + 1) % 200 == 0: print(f"  [{i+1}/{len(rows)}] ok={ok} fail={fail}", flush=True)
    print(f"[pairframe] done ok={ok} fail={fail}", flush=True)
    meta.close()


if __name__ == "__main__":
    main()
