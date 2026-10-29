#!/usr/bin/env python
"""Closed-loop stage 1 (GPU): per query, ONE forward produces both candidates:
  P = GPT K=5 POINTS -> prompt masklet (Tool A, new best 0.472)
  X = GPT K=5 boxes  -> prompt masklet (backup candidate, same forward P=2)
  B = MF-CoT's OWN dumped answer (inst3d + 2D masks from ITS forward; {key}__mf.npz) - no re-derivation
Frame set = capped-64 hi selection ∪ picks, uniform fill to 64. NO GT leakage into frames.
Outputs per query:
  {OUT}/cards/{key}__A.jpg / __B.jpg   verify cards (top-3 pred-area frames, [raw|magenta] rows)
  {OUT}/state/{key}.json               fs, per-candidate 2D area frames, 3D miou/biou vs GT (for scoring)
  {OUT}/masks/{key}.npz                maskA/maskB (T,96,128 bool) for retries
"""
import os, sys, json, glob
import _env
import numpy as np
from PIL import Image
import cv2

AG = os.environ.get("GROUNDING_REPO", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AG)
import align3d
from align3d import load_single_scene_data, map_pred_inst_to_gt_pointcloud
from clip_embed import gt_masks, aabb_iou, PTS

D = _env.data_root()
SR = f"{D}/scanrefer/annotations/scanrefer/ScanRefer_filtered.json"
IDM = f"{D}/gt_idmap_cache"
SC2D = os.environ.get("SCANNET200_2D_ROOT", "/PATH/TO/codebase/SegVGGT/data_processing/scannet200/scenes_2d")
VRANGES = f"{D}/vidi_tr_ranges"
K5 = f"{D}/critic_openloop/resp_k5"
K5PT = f"{D}/critic_openloop/resp_k5pt"
SELHI = f"{D}/critic_frames/resp_hi"
MFC = os.environ.get("MFC_DIR", f"{D}/scanrefer_mfcot_cl_N2")
OUT = os.environ.get("CLOUT", f"{D}/closedloop")
for d in ("cards", "state", "masks"): os.makedirs(f"{OUT}/{d}", exist_ok=True)
VG = _env.require("VGGT_OMEGA_CKPT", "VGGT-Omega backbone weights")
CKPT = os.environ.get("PROMPT_CKPT", "/PATH/TO/checkpoints/S4VY/s4vy_prompt/prompt_final.pt")
NF = 64
IN_W, IN_H = 512, 384
MAG = np.array([255, 0, 200], np.float32)


def sample_queries():
    sr = json.load(open(SR))
    desc = {(x["scene_id"], int(x["object_id"]), str(x.get("ann_id"))): 1 for x in sr}
    rng = np.random.RandomState(0)
    pool = []
    for f in sorted(glob.glob(f"{VRANGES}/*.json")):
        sc = os.path.basename(f)[:-5]
        rs = json.load(open(f))
        if not rs or rs[0]["S"] > 240: continue
        rs = [r for r in rs if (sc, r["oid"], str(r["ann_id"])) in desc]
        for i in rng.permutation(len(rs))[:2]:
            pool.append({"scene": sc, "oid": rs[i]["oid"], "ann_id": str(rs[i]["ann_id"]), "S": rs[i]["S"]})
    rng.shuffle(pool)
    return pool[:160]


def load_model():
    import torch
    sys.path.insert(0, os.environ.get("SEG4D_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from segomega.models.instseg_model import SegOmega
    from segomega.engine.optimizer import setup_trainable
    m = SegOmega(ckpt_path=VG, num_queries=400, num_dec_layers=9, mask_dim=256, dec_dim=256,
                 supervise_depth=False, supervise_camera=False, use_teacher=False,
                 grad_checkpointing=True, freeze_backbone=False, load_pretrained=True,
                 mask_feat_ratio=4, enable_prompt=True, prompt_arch="v1lp")
    m = setup_trainable(m, mode="lora", lora_rank=256, lora_alpha=256.0).cuda()
    sd = torch.load(CKPT, map_location="cpu")
    missing, unexpected = m.load_state_dict(sd.get("model", sd), strict=False)
    assert not missing and not unexpected
    return m.eval()


def capped_sel(scene, oid, ann, S, cap=64):
    rf = f"{SELHI}/{scene}__{oid}__{ann}.json"
    sel = []
    if os.path.exists(rf):
        r = json.load(open(rf))
        if isinstance(r.get("frames"), list):
            sel = sorted(set(int(f) for f in r["frames"] if 0 <= int(f) < S))
    if len(sel) > cap:
        sel = [sel[i] for i in np.linspace(0, len(sel) - 1, cap).round().astype(int)]
    return sel


def frame_set(sel, picks, S):
    base = sorted(set(sel) | set(picks))
    if not base: base = list(np.linspace(0, S - 1, min(NF, S)).round().astype(int))
    if len(base) > NF:
        keep = set(picks)
        rest = [f for f in base if f not in keep]
        take = [rest[i] for i in np.linspace(0, len(rest) - 1, NF - len(keep)).round().astype(int)]
        base = sorted(keep | set(take))
    if len(base) < min(NF, S):
        extra = [f for f in np.linspace(0, S - 1, min(NF, S)).round().astype(int) if f not in base]
        base = sorted(set(base) | set(extra[:min(NF, S) - len(base)]))
    return base


def render_card(sc, fs, mask, path):
    """mask: (T,96,128) bool. top-3 frames by pred area -> rows of [raw | magenta overlay]."""
    area = mask.reshape(len(fs), -1).sum(1)
    top = np.argsort(-area)[:3]
    if area[top[0]] == 0: top = np.linspace(0, len(fs) - 1, min(3, len(fs))).astype(int)
    rows = []
    for ti in top:
        im = np.array(Image.open(f"{SC2D}/{sc}/color/{fs[int(ti)]}.jpg").convert("RGB"))
        m = cv2.resize(mask[int(ti)].astype(np.float32), (im.shape[1], im.shape[0]),
                       interpolation=cv2.INTER_LINEAR) > 0.5
        o = im.astype(np.float32).copy()
        o[m] = o[m] * 0.6 + MAG * 0.4
        cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(o, cont, -1, tuple(int(c) for c in MAG), 2)
        rows.append(np.concatenate([im, o.astype(np.uint8)], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path, quality=90)
    return [int(fs[int(t)]) for t in top]


def main():
    import torch
    qs = sample_queries()
    mfc = {}
    for f in glob.glob(f"{MFC}/*.json"):
        sc = os.path.basename(f)[:-5]
        for r in (json.load(open(f)) or []):
            mfc[f"{sc}__{r['oid']}__{r.get('ann_id')}"] = r
    print(f"[cl-stage1] {len(qs)} queries, mfcot cache={len(mfc)}", flush=True)
    m = load_model()
    by_scene = {}
    for q in qs: by_scene.setdefault(q["scene"], []).append(q)
    n = 0
    for si, (sc, qlist) in enumerate(sorted(by_scene.items())):
        if all(os.path.exists(f"{OUT}/state/{q['scene']}__{q['oid']}__{q['ann_id']}.json") for q in qlist):
            n += len(qlist); continue
        sample = load_single_scene_data("data_processing/scannet200", sc, "scannet200", 1)
        if sample is None: continue
        coord = np.fromfile(f"{PTS}/{sc}.bin", np.float32).reshape(-1, 6)[:, :3]
        gtm, _ = gt_masks(sc)
        for q in qlist:
            key = f"{q['scene']}__{q['oid']}__{q['ann_id']}"
            sf = f"{OUT}/state/{key}.json"
            if os.path.exists(sf): n += 1; continue
            g = json.load(open(f"{K5}/{key}.json")) if os.path.exists(f"{K5}/{key}.json") else {}
            gp = json.load(open(f"{K5PT}/{key}.json")) if os.path.exists(f"{K5PT}/{key}.json") else {}
            mf = mfc.get(key)
            gt3d = gtm.get(str(q["oid"]))
            if "error" in g or gt3d is None or gt3d.sum() == 0: continue
            S = min(q["S"], len(sample["intrinsics"]))
            W, H = g.get("W", 640), g.get("H", 478)
            boxes = []
            for b in g.get("boxes", []):
                bb = b.get("bbox"); f = int(b.get("frame", -1))
                if bb is None or not isinstance(bb, list) or len(bb) != 4 or not (0 <= f < S): continue
                x0, x1 = sorted((max(0, min(float(bb[0]), W)) / W, max(0, min(float(bb[2]), W)) / W))
                y0, y1 = sorted((max(0, min(float(bb[1]), H)) / H, max(0, min(float(bb[3]), H)) / H))
                if x1 - x0 > 1e-3 and y1 - y0 > 1e-3: boxes.append((f, x0, y0, x1, y1))
            pts = []
            for b in gp.get("points", []):
                pt = b.get("point"); f = int(b.get("frame", -1))
                if pt is None or not isinstance(pt, list) or len(pt) != 2 or not (0 <= f < S): continue
                pts.append((f, max(0, min(float(pt[0]), W)) / W, max(0, min(float(pt[1]), H)) / H))
            if not boxes or not pts: continue
            sel = capped_sel(q["scene"], q["oid"], q["ann_id"], S)
            fs = frame_set(sel, sorted(set([b[0] for b in boxes] + [t[0] for t in pts])), S)
            loc = {f: i for i, f in enumerate(fs)}
            prompt_pt = {"points": [(loc[f], x, y, 1) for f, x, y in pts if f in loc], "boxes": []}
            prompt = {"boxes": [(loc[f],) + t[1:] for t in boxes if (f := t[0]) in loc]}
            if not prompt_pt["points"]: continue
            imgs = torch.stack([torch.from_numpy(np.array(
                Image.open(f"{SC2D}/{sc}/color/{f}.jpg").convert("RGB").resize((IN_W, IN_H)),
                dtype=np.float32) / 255.0).permute(2, 0, 1) for f in fs]).unsqueeze(0).cuda()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pr = m(imgs, prompts=[[prompt_pt, prompt]])
            mP = (pr["prompt_mask_logits"][0, 0].float() > 0).cpu().numpy()          # point masklet
            mX = (pr["prompt_mask_logits"][0, 1].float() > 0).cpu().numpy()          # box masklet
            fsn = np.array(fs, int)
            lift = map_pred_inst_to_gt_pointcloud(sample["points"], sample["super_points"],
                        sample["intrinsics"][fsn], sample["extrinsics"][fsn], sample["depth_maps"][fsn],
                        np.stack([mP, mX]).astype(np.float32), depth_tolerance=0.1).astype(bool)
            row = {"key": key, "fs": [int(x) for x in fs],
                   "boxes_frames": [int(b[0]) for b in boxes], "pts_frames": [int(t[0]) for t in pts]}
            for name, p3, m2 in (("P", lift[0], mP), ("X", lift[1], mX)):
                inter = float((p3 & gt3d).sum()); union = float((p3 | gt3d).sum())
                row[f"{name}_miou3d"] = inter / max(union, 1.0)
                row[f"{name}_biou3d"] = aabb_iou(coord[p3], coord[gt3d])
                row[f"{name}_card_frames"] = render_card(sc, fs, m2, f"{OUT}/cards/{key}__{name}.jpg")
            # Tool B: MF-CoT's own dumped answer -- 3D mask + its forward's 2D masks, verbatim
            mfz = f"{MFC}/{key}__mf.npz"
            row["B_miou3d"], row["B_biou3d"] = 0.0, 0.0
            mB = None
            if os.path.exists(mfz):
                z = np.load(mfz)
                b3d = np.unpackbits(z["inst3d"])[:int(z["N"])].astype(bool)
                inter = float((b3d & gt3d).sum()); union = float((b3d | gt3d).sum())
                row["B_miou3d"] = inter / max(union, 1.0)
                row["B_biou3d"] = aabb_iou(coord[b3d], coord[gt3d])
                if z["m2"].size:
                    sh = tuple(int(x) for x in z["m2shape"])
                    mB2 = np.unpackbits(z["m2"])[:sh[0]*sh[1]*sh[2]].reshape(sh).astype(bool)
                    fcap = [int(x) for x in z["fcap"]]
                    row["B_card_frames"] = render_card(sc, fcap, mB2, f"{OUT}/cards/{key}__B.jpg")
                    mB = (mB2, fcap)
            np.savez_compressed(f"{OUT}/masks/{key}.npz", P=np.packbits(mP), X=np.packbits(mX),
                                shape=np.array(mP.shape),
                                **({"Bm2": np.packbits(mB[0]), "Bshape": np.array(mB[0].shape),
                                    "Bfcap": np.array(mB[1], np.int32)} if mB else {}))
            json.dump(row, open(sf, "w")); n += 1
        if (si + 1) % 15 == 0: print(f"  scene {si+1}/{len(by_scene)} n={n}", flush=True)
    R = [json.load(open(f)) for f in glob.glob(f"{OUT}/state/*.json")]
    print(f"\n=== CLOSED-LOOP STAGE1 (n={len(R)}) ===")
    for k in ("P_miou3d", "X_miou3d", "B_miou3d"):
        v = np.array([r[k] for r in R])
        print(f"  {k}: mean={v.mean():.3f} acc@.25={100*(v>=.25).mean():.1f} acc@.5={100*(v>=.5).mean():.1f}")
    both = np.array([max(r["P_miou3d"], r["X_miou3d"], r["B_miou3d"]) for r in R])
    print(f"  oracle-of-three: mean={both.mean():.3f} acc@.25={100*(both>=.25).mean():.1f} acc@.5={100*(both>=.5).mean():.1f}")


if __name__ == "__main__":
    main()
