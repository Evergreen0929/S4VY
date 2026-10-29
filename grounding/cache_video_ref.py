#!/usr/bin/env python
"""s4vy cache for VIDEO referring-segmentation data (MeViS / Ref-DAVIS / LVOS), for joint v3.

Mirrors cache_scene_masks but drops every 3D step: these datasets have no poses or depth, so there is no
projection, no point cloud and no 3D IoU. What is left is exactly the part MFRAME actually trains on --
s4vy's per-frame instance masks, its query embeddings, the injected registers, and the GT bridge:

    target[oid] = argmax_j  video-level 2D IoU( s4vy instance j , GT mask of oid )

The bridge is TEXT-INDEPENDENT, so this (GPU-expensive) pass is run once and the referring expressions are
joined in later by the manifest builder. That matters for LVOS, whose expressions are pseudo-labels being
regenerated in parallel -- relabelling must not force a re-cache.

Env: DS=mevis|refdavis|lvos  SPLIT=train|valid  SHARD/NSHARD  MAXF=96  OUT
"""
import os, json, glob
import _env
os.environ.setdefault("BALANCED_RES", "512")
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from align3d import build_omega
from cache_scene import MW, MH, _last
from cache_scene_masks import seg_forward_keepmasks, iou_video

D = _env.data_root()
RV = f"{D}/RefVOS"
DS = os.environ.get("DS", "mevis"); SPLIT = os.environ.get("SPLIT", "train")
MAXF = int(os.environ.get("MAXF", "96"))
MINPIX = int(os.environ.get("MINPIX", "20"))          # same spirit as VIS_T: below this an object is 'absent'
SHARD = int(os.environ.get("SHARD", "0")); NSHARD = int(os.environ.get("NSHARD", "1"))
OUT = os.environ.get("OUT", f"{D}/vidref_cache_{DS}_{SPLIT}")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- per-dataset adapters
def _sub(n, k):
    """Evenly sub-sample k of n frame indices (LVOS averages 729 frames; the cache keeps MAXF)."""
    return list(range(n)) if n <= k else np.linspace(0, n - 1, k).round().astype(int).tolist()


class MeViS:
    root = f"{RV}/mevis/{SPLIT}"
    def videos(self):
        return sorted(json.load(open(f"{self.root}/meta_expressions.json"))["videos"])
    def load(self, vid):
        meta = json.load(open(f"{self.root}/meta_expressions.json"))["videos"][vid]
        names = meta["frames"]; idx = _sub(len(names), MAXF)
        ims = np.stack([np.array(Image.open(f"{self.root}/JPEGImages/{vid}/{names[i]}.jpg").convert("RGB")) for i in idx])
        from pycocotools import mask as mu
        md = _MEVIS_MD[0]
        gt = {}
        for eid, e in meta["expressions"].items():
            if len(e["obj_id"]) != 1: continue                      # single-target only (multi-object excluded)
            oid = int(e["obj_id"][0]); aid = str(e["anno_id"][0])
            if oid in gt or aid not in md: continue
            rles = md[aid]
            m = np.stack([(mu.decode(rles[i]).astype(bool) if i < len(rles) and rles[i] else
                           np.zeros(ims.shape[1:3], bool)) for i in idx])
            gt[oid] = m
        return ims, gt, idx
    def exprs(self, vid):
        meta = json.load(open(f"{self.root}/meta_expressions.json"))["videos"][vid]
        out = {}
        for e in meta["expressions"].values():
            if len(e["obj_id"]) != 1: continue
            out.setdefault(int(e["obj_id"][0]), []).append(e["exp"])
        return out


class RefDAVIS:
    root = f"{RV}/refdavis/DAVIS"
    def videos(self):
        s = "train" if SPLIT == "train" else "val"
        return sorted(l.strip() for l in open(f"{self.root}/ImageSets/2017/{s}.txt") if l.strip())
    def load(self, vid):
        fs = sorted(glob.glob(f"{self.root}/JPEGImages/480p/{vid}/*.jpg")); idx = _sub(len(fs), MAXF)
        ims = np.stack([np.array(Image.open(fs[i]).convert("RGB")) for i in idx])
        ans = sorted(glob.glob(f"{self.root}/Annotations_unsupervised/480p/{vid}/*.png"))
        a = np.stack([np.array(Image.open(ans[i])) for i in idx])
        return ims, {int(o): (a == o) for o in np.unique(a) if o != 0}, idx
    def exprs(self, vid):
        out = {}
        for f in ("Davis17_annot1", "Davis17_annot2"):                # two annotators -> two expressions/object
            for l in open(f"{RV}/refdavis/davis_text_annotations/{f}.txt"):
                p = l.strip().split(None, 2)
                if len(p) == 3 and p[0] == vid:
                    out.setdefault(int(p[1]), []).append(p[2].strip().strip('"'))
        return out


class LVOS:
    def __init__(self): self.root = f"{RV}/lvos/{SPLIT}"
    def videos(self): return sorted(os.listdir(f"{self.root}/JPEGImages"))
    def load(self, vid):
        fs = sorted(glob.glob(f"{self.root}/JPEGImages/{vid}/*.jpg"))
        ans = sorted(glob.glob(f"{self.root}/Annotations/{vid}/*.png"))
        keep = {os.path.basename(p)[:-4] for p in ans}                # LVOS annotates every 5th frame only
        fs = [p for p in fs if os.path.basename(p)[:-4] in keep]
        idx = _sub(len(fs), MAXF)
        ims = np.stack([np.array(Image.open(fs[i]).convert("RGB")) for i in idx])
        a = np.stack([np.array(Image.open(ans[i])) for i in idx])
        return ims, {int(o): (a == o) for o in np.unique(a) if o != 0}, idx
    def exprs(self, vid):
        p = f"{RV}/lvos/labels_gpt_{SPLIT}/{vid}.txt"                 # written by the GPT-5.6 relabel pass
        if not os.path.exists(p): return {}
        out = {}
        for l in open(p):
            q = l.strip().split(None, 2)
            if len(q) == 3: out.setdefault(int(q[1]), []).append(q[2].strip().strip('"'))
        return out


_MEVIS_MD = [None]
ADAPTERS = {"mevis": MeViS, "refdavis": RefDAVIS, "lvos": LVOS}


def main():
    ad = ADAPTERS[DS]()
    if DS == "mevis":
        _MEVIS_MD[0] = json.load(open(f"{ad.root}/mask_dict.json"))    # 392MB, loaded once per worker
    vids = [v for i, v in enumerate(ad.videos()) if i % NSHARD == SHARD]
    model, _ = build_omega()
    model.query_decoder.decoder_norm.register_forward_hook(lambda m, i, o: _last.__setitem__("q", o.detach()))
    model.backbone.register_forward_hook(lambda m, i, o: _last.__setitem__("reg", o["tokens_list"]))
    for vid in vids:
        of = f"{OUT}/{vid}.npz"
        if os.path.exists(of): continue                                # idempotent: re-runs only fill gaps
        try:
            with torch.no_grad():
                ims, gt, idx = ad.load(vid)
                if not gt: print(f"{vid}: no GT object, skip", flush=True); continue
                S = ims.shape[0]
                masks_full, masks2d, scores, qemb, register = seg_forward_keepmasks(model, ims)
                del masks_full
                n = masks2d.shape[0]
                if n == 0: np.savez_compressed(of, empty=True); continue
                idmap = np.full((S, MH, MW), -1, np.int16)
                rows = []
                ex = ad.exprs(vid)
                for oid, m in gt.items():
                    g = F.interpolate(torch.from_numpy(m).float()[:, None], size=(MH, MW), mode="area")[:, 0] > 0.5
                    g = g.numpy()
                    idmap[g] = oid
                    i2 = np.array([iou_video(masks2d[j], g) for j in range(n)]) if g.any() else np.zeros(n)
                    px = g.reshape(S, -1).sum(1)
                    rows.append(dict(scene=vid, ds=DS, split=SPLIT, oid=int(oid), n=n,
                                     target=int(i2.argmax()), best_iou2d=round(float(i2.max()), 3),
                                     present=int((px > MINPIX).sum()), S=int(S),
                                     pres_rate=round(float((px > MINPIX).mean()), 3),
                                     px_med=int(np.median(px)), descs=ex.get(int(oid), [])))
                np.savez_compressed(of, qemb=qemb.astype(np.float32), scores=scores.astype(np.float32),
                                    register=register.astype(np.float16), idmap=idmap,
                                    pred_masks=np.packbits(masks2d.reshape(n, -1), axis=1),
                                    pred_shape=np.array([n, S, MH, MW]), n=n,
                                    frame_idx=np.array(idx), rows=json.dumps(rows))
                print(f"{vid}: S={S} n={n} objs={len(rows)} iou2d={np.mean([r['best_iou2d'] for r in rows]):.3f} "
                      f"pres={np.mean([r['pres_rate'] for r in rows]):.3f}", flush=True)
        except Exception as e:
            print(f"{vid}: FAIL {type(e).__name__}: {e}", flush=True)
            import gc; e.__traceback__ = None; del e; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__": main()
