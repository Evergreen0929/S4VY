#!/usr/bin/env python
"""Render critic cards for ScanNet++ scenes.

For each dumped query the two mask candidates (query-matching stream and box stream) are drawn over
the same frames, so the critic sees a matched pair and only the mask differs.

Env: DUMP_DIR (holds __box.json / __qmask.npz), OUT, POOL (jsonl of scenes), SHARD/NSHARD, PROMPT_K, PICK.
"""
import os, sys, json, glob
import _env
os.environ.setdefault("SEG_TAG", "s4vy"); os.environ.setdefault("NO_INST_PROJ", "1")
os.environ.setdefault("MFRAME", "1"); os.environ.setdefault("COT", "1"); os.environ.setdefault("BOX_OUT", "1")
D = _env.data_root()
os.environ.setdefault("MODEL_PATH", os.environ.get("QWEN3_VL_2B", ""))
import numpy as np, torch, cv2
from PIL import Image
from prompt_infer import load_model, IN_W, IN_H
from cache_scene import load_scene
from align3d import map_pred_inst_to_gt_pointcloud
from gen_spp_cot import scene_points
SPPP_2D = f"{D}/scannetpp_scenes_2d"


def scene_frames(sc, idx):
    """Load the scene's RGB frames at the indices the dump refers to."""
    out = []
    for k in idx:
        im = Image.open(f"{SPPP_2D}/{sc}/color/{int(k)}.jpg").convert("RGB")
        if im.width > 512: im = im.resize((512, int(im.height * 512 / im.width)))
        out.append(np.asarray(im))
    return np.stack(out)

DUMP_DIR = os.environ.get("DUMP_DIR", f"{D}/sppv3tr_dump")
OUT = os.environ.get("OUT", f"{D}/critic_cards_scene")
import perframe_cards as PF
POOL = os.environ.get("POOL", f"{D}/spp_train.jsonl")
SPPC = f"{D}/scannetpp_cache_v2"
PK = int(os.environ.get("PROMPT_K", "5")); PICK = os.environ.get("PICK", "area")
FR_CAP = int(os.environ.get("FR_CAP", "48"))
SHARD = int(os.environ.get("SHARD", "0")); NSHARD = int(os.environ.get("NSHARD", "1"))
MAG = np.array([255, 0, 255], np.float32)
pm_model = load_model()


def aabb_iou(a, b):
    if not len(a) or not len(b): return 0.0
    lo = np.maximum(a.min(0), b.min(0)); hi = np.minimum(a.max(0), b.max(0))
    inter = float(np.prod(np.clip(hi - lo, 0, None)))
    va = float(np.prod(a.max(0) - a.min(0))); vb = float(np.prod(b.max(0) - b.min(0)))
    return inter / max(va + vb - inter, 1e-9)


def render(imgs, fs, mask, path):
    """ prompt_infer.render_card , mask: (len(fs),h,w)"""
    area = mask.reshape(len(fs), -1).sum(1)
    top = np.argsort(-area)[:3]
    if area[top[0]] == 0: top = np.linspace(0, len(fs) - 1, min(3, len(fs))).astype(int)
    rows = []
    for ti in top:
        im = imgs[int(ti)]
        m = cv2.resize(mask[int(ti)].astype(np.float32), (im.shape[1], im.shape[0]),
                       interpolation=cv2.INTER_LINEAR) > 0.5
        o = im.astype(np.float32).copy()
        o[m] = o[m] * 0.6 + MAG * 0.4
        cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(o, cont, -1, tuple(int(c) for c in MAG), 2)
        rows.append(np.concatenate([im, o.astype(np.uint8)], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path, quality=90)
    return [int(fs[int(t)]) for t in top]


def pick_boxes(items):
    if PK <= 0 or len(items) <= PK: return items
    if PICK == "area":
        a = [(it[3] - it[1]) * (it[4] - it[2]) for it in items]
        return [items[int(i)] for i in sorted(np.argsort(-np.asarray(a))[:PK])]
    return [items[i] for i in np.linspace(0, len(items) - 1, PK).round().astype(int)]


@torch.no_grad()
def main():
    desc = {}
    for l in open(POOL):
        r = json.loads(l); desc[f"{r['scene']}__{r['oid']}__{r['ann']}"] = r
    keys = sorted(k for k in desc if os.path.exists(f"{DUMP_DIR}/{k}__box.json"))
    byscene = {}
    for k in keys: byscene.setdefault(k.split("__")[0], []).append(k)
    scenes = sorted(byscene)
    scenes = [s for i, s in enumerate(scenes) if i % NSHARD == SHARD]
    meta = open(f"{OUT}/meta_{SHARD:02d}.jsonl", "a")
    print(f"[sppcards] {len(scenes)} scenes / {sum(len(byscene[s]) for s in scenes)} query PICK={PICK} K={PK}", flush=True)
    for si, sc in enumerate(scenes):
        try:
            z = np.load(f"{SPPC}/{sc}.npz", allow_pickle=True)
            instmask = z["instmask"]; S = int(z["register"].shape[0])
            geo = load_scene(sc)
            pts = scene_points(sc)
            imgs_all = scene_frames(sc, range(S))
        except Exception as e:
            print(f"{sc}: LOAD FAIL {type(e).__name__}: {e}", flush=True); continue
        for key in byscene[sc]:
            _sent = f"{OUT}/img/{key}__B{PF.NFRAME-1}.jpg" if PF.PERFRAME else f"{OUT}/img/{key}__B.jpg"
            if os.path.exists(_sent): continue
            r = desc[key]; oid = int(r["oid"])
            gt = (instmask == oid)
            if gt.sum() == 0: continue
            qf = f"{DUMP_DIR}/{key}__qmask.npz"
            _noA = not os.path.exists(qf)
            if _noA and (r.get("_ref_miou") or 0) > 0:
                print(f"{key}: A  {r.get('_ref_miou')}  qmask,  dump", flush=True); continue
            if _noA:
                mA = None; arA = None
            else:
                _z = np.load(qf); sh = _z["shape"]
                mA = np.unpackbits(_z["mask"], axis=1)[:, :int(sh[1]) * int(sh[2])].reshape(*sh).astype(bool)
                arA = mA.reshape(len(mA), -1).sum(1)
            if mA is None:
                fsA = [0]; fnA = np.array(fsA, int); liftA = np.zeros(gt.shape, bool)
            else:
                fsA = sorted(int(x) for x in np.argsort(-arA)[:3] if arA[int(x)] > 0) or [0]
                fnA = np.array(fsA, int)
                _all = np.arange(len(mA), dtype=int)
                liftA = map_pred_inst_to_gt_pointcloud(geo["points"], geo["super_points"],
                            geo["intrinsics"][_all], geo["extrinsics"][_all], geo["depth_maps"][_all],
                            mA[_all][None].astype(np.float32), depth_tolerance=0.1).astype(bool)[0]
            items = [it for it in json.load(open(f"{DUMP_DIR}/{key}__box.json")).get("boxes", [])
                     if int(it[0]) < S and (it[3] - it[1]) > 1e-3 and (it[4] - it[2]) > 1e-3]
            if not items: continue
            prompts = pick_boxes(items)
            pf = sorted({int(it[0]) for it in prompts})
            rest = [f for f in range(S) if f not in set(pf)]
            room = max(1, FR_CAP - len(pf))
            mB_by = {}; liftB = None; ok = True
            for ch in ([rest[i:i + room] for i in range(0, len(rest), room)] or [[]]):
                fs = sorted(set(pf) | set(ch)); loc = {fr: i for i, fr in enumerate(fs)}
                im = torch.stack([torch.from_numpy(np.asarray(
                        Image.fromarray(imgs_all[fr]).resize((IN_W, IN_H)), dtype=np.float32) / 255.
                     ).permute(2, 0, 1) for fr in fs]).unsqueeze(0).cuda()
                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pr = pm_model(im, prompts=[[{"points": [], "boxes": [
                            (loc[int(it[0])], *[float(x) for x in it[1:5]]) for it in prompts]}]])
                    mk = (pr["prompt_mask_logits"][0, 0].float() > 0).cpu().numpy()
                except Exception as e:
                    print(f"{key}: PROMPT FAIL {type(e).__name__}: {e}", flush=True); ok = False; break
                fn = np.array(fs, int)
                l = map_pred_inst_to_gt_pointcloud(geo["points"], geo["super_points"],
                        geo["intrinsics"][fn], geo["extrinsics"][fn], geo["depth_maps"][fn],
                        mk[None].astype(np.float32), depth_tolerance=0.1).astype(bool)[0]
                liftB = l if liftB is None else (liftB | l)
                for i, fr in enumerate(fs): mB_by[fr] = mk[i]
            if not ok or liftB is None: continue
            iou = lambda l: (float(aabb_iou(pts[l], pts[gt])),
                             float((l & gt).sum()) / max(float((l | gt).sum()), 1.0))
            bA, mIA = (0.0, 0.0) if mA is None else iou(liftA)
            bB, mIB = iou(liftB)
            if r.get("_ref_miou") is not None: mIA = float(r["_ref_miou"])
            if r.get("_ref_biou") is not None: bA = float(r["_ref_biou"])
            if r.get("_ref_box_miou") is not None: mIB = float(r["_ref_box_miou"])
            if r.get("_ref_box_biou") is not None: bB = float(r["_ref_box_biou"])
            arB = {f: m.sum() for f, m in mB_by.items()}
            if PF.PERFRAME:
                mA_by = ({int(f): np.zeros_like(m) for f, m in mB_by.items()} if mA is None
                         else {int(f): mA[int(f)] for f in range(len(mA))})
                _frs = PF.pick_frames(mA_by, mB_by)
                PF.dump_masks(OUT, key, mA_by, mB_by)
                vA = PF.render_perframe(lambda f: imgs_all[f], _frs, mA_by, OUT, key, "A")
                vB = PF.render_perframe(lambda f: imgs_all[f], _frs, mB_by, OUT, key, "B")
                _nc = PF.n_common(mA_by, mB_by)
            else:
                fsB = sorted(sorted(arB, key=lambda f: -arB[f])[:3]) or fsA
                _mAr = (np.zeros((len(fnA),) + mB_by[fsB[0]].shape, bool) if mA is None else mA[fnA])
                vA = render(imgs_all[fnA], fsA, _mAr, f"{OUT}/img/{key}__A.jpg")
                vB = render(imgs_all[np.array(fsB, int)], fsB,
                            np.stack([mB_by[f] for f in fsB]), f"{OUT}/img/{key}__B.jpg")
                _nc = -1
            meta.write(json.dumps({
                "key": key, "scene": sc, "oid": oid, "ann": r["ann"], "ds": "scannetpp",
                "cat": r.get("name", "") or "", "desc": r["desc"],
                "A_biou": round(bA, 4), "A_miou": round(mIA, 4),
                "B_biou": round(bB, 4), "B_miou": round(mIB, 4),
                "metric": "iou3d",
                "fmt": (f"perframe{PF.NFRAME}" if PF.PERFRAME else "card3"),
                "S": int(S), "n_common": _nc,
                "label": "A" if bA >= bB else "B", "margin": round(abs(bA - bB), 4),
                "viewsA": vA, "viewsB": vB}) + "\n"); meta.flush()
        print(f"[{si+1}/{len(scenes)}] {sc}: {len(byscene[sc])} q", flush=True)
    meta.close()


if __name__ == "__main__": main()
