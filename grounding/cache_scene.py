#!/usr/bin/env python
"""ScanNet++ (SpaceFormer V4) MF-CoT training cache — BALANCED res (BALANCED_RES=512 -> 448x592).
Frames+poses+depth from InsScene-processed_scannetpp_v2; GT 3D masks + captions from SpaceFormer V4.
Chain: InsScene frames -> segmenter forward (balanced) -> map 2D masks to SpaceFormer GT cloud
       -> per-caption target = best-IoU segmenter instance; desc = SpaceFormer caption.
Coord convention NAILED: extrinsic = world2cam = inv(trajectory); depth png / 1000 = metres.
Env: SC (scene), BALANCED_RES=512, MAXF (frame cap, default 96), NPTS (cloud downsample, default 200000).
Single-scene test: SC=00777c41d4 python cache_scene.py"""
import os, json, tarfile, io, glob
import _env
os.environ.setdefault("BALANCED_RES", "512")
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from align3d import build_omega, map_pred_inst_to_gt_pointcloud, SCORE_THR

D = _env.data_root()
INS = os.environ.get("INSSCENE_SPP_ROOT", f"{D}/InsScene-15K/processed_scannetpp_v2/processed_scannetpp_v2")
SF = os.environ.get("SPACEFORMER_SPP_ROOT", "/PATH/TO/datasets/spaceformer/v4/scannetpp_v4gt_qwen")
OUT = os.environ.get("OUT", f"{D}/scannetpp_train_cache")
MAXF = int(os.environ.get("MAXF", "96")); NPTS = int(os.environ.get("NPTS", "200000"))
os.makedirs(OUT, exist_ok=True)
_last = {}

def _bal_hw(images_np, R=int(os.environ.get("BALANCED_RES", "512")), p=16):
    if R <= 0: return (384, 512)
    import math; H, W = int(images_np.shape[1]), int(images_np.shape[2])
    ar = max(0.5, min(2.0, W / H)); tok = (R / p) ** 2
    hp = round(math.sqrt(tok / ar)); wp = round(hp * ar); return (hp * p, wp * p)

MW, MH = 128, 96                                             # gt_idmap 2D resolution (matches ScanNet200 caches)

def project_gt_idmap(points, instmask, ints, exts, deps, tol=0.1):
    """GT 3D points -> per-frame 2D idmap (S,MH,MW) int16 = GT obj_idx per pixel (-1 empty). z-buffer: nearest wins.
    Depth-consistency (|proj_z - depth_map| < tol) for visibility. Mirrors ScanNet200 gt_idmap contract."""
    S = len(ints); dh, dw = deps.shape[1], deps.shape[2]
    idmap = -np.ones((S, MH, MW), np.int16)
    ph = np.hstack([points[:, :3].astype(np.float64), np.ones((len(points), 1))])
    for f in range(S):
        pc = (exts[f] @ ph.T).T[:, :3]; ic = ints[f] @ pc.T
        z = ic[2].copy(); z[z == 0] = 1e-6; u = ic[0] / z; v = ic[1] / z
        valid = (z > 0) & (u >= 0) & (u < dw) & (v >= 0) & (v < dh)
        if not valid.any(): continue
        ui = u[valid].astype(int); vi = v[valid].astype(int); zv = z[valid]
        dmap = deps[f][vi.clip(0, dh-1), ui.clip(0, dw-1)]
        vis = (dmap > 0) & (np.abs(zv - dmap) < tol)                        # depth-consistent = actually visible
        if not vis.any(): continue
        uu = (u[valid][vis] * MW / dw).astype(int).clip(0, MW-1); vv = (v[valid][vis] * MH / dh).astype(int).clip(0, MH-1)
        oid = instmask[valid][vis]; zz = zv[vis]
        order = np.argsort(-zz)                                             # far first -> near overwrites (z-buffer)
        flat = idmap[f].ravel(); px = (vv * MW + uu)[order]; ov = oid[order]
        m2 = ov >= 0; flat[px[m2]] = ov[m2]
    return idmap

@torch.no_grad()
def seg_forward(model, images_np):
    S = images_np.shape[0]
    t = torch.from_numpy(images_np).float().div(255.).permute(0, 3, 1, 2)
    t = F.interpolate(t, size=_bal_hw(images_np), mode="bilinear", align_corners=False).cuda()
    fp = model.query_decoder.frame_pos
    if S > fp.num_embeddings:
        import torch.nn as nn
        w = fp.weight.data; wn = F.interpolate(w.t()[None], size=S, mode="linear", align_corners=False)[0].t().contiguous()
        ne = nn.Embedding(S, w.shape[1]).to(w.device); ne.weight.data = wn; model.query_decoder.frame_pos = ne
    out = model(t)
    tl = [x for x in _last["reg"] if x is not None][-1]      # backbone tokens_list[-1] (1,S,T,2048) -> per-frame 17 register tokens
    register = tl[0, :, :17].float().cpu().numpy()           # (S,17,2048), matches s4vy_register_cache contract
    qf = _last["q"][0]; obj = out["objectness"][0]; ml = out["mask_logits"][0]
    scores = obj.softmax(-1)[..., 0]; keep = scores > SCORE_THR
    masks = (ml.sigmoid() > 0.5) & keep.view(-1, 1, 1, 1)
    keep = keep & (masks.flatten(1).sum(-1) > 10)
    return masks[keep].float().cpu().numpy(), scores[keep].cpu().numpy(), qf[keep].float().cpu().numpy(), register

def load_scene(sc):
    """Return sample dict mirroring load_single_scene_data, from InsScene frames + SpaceFormer cloud."""
    md = np.load(f"{INS}/{sc}/scene_iphone_metadata.npz", allow_pickle=True)
    traj = md["trajectories"].astype(np.float64); K = md["intrinsics"].astype(np.float64); fnames = md["images"]
    S = len(traj); idx = np.arange(S) if S <= MAXF else np.linspace(0, S - 1, MAXF).round().astype(int)
    imgs, deps, exts, ints = [], [], [], []
    for i in idx:
        fn = str(fnames[i]).replace(".png", ".jpg")
        ip = f"{INS}/{sc}/images/{fn}"; dp = f"{INS}/{sc}/depth/{fn.replace('.jpg', '.png')}"
        if not (os.path.exists(ip) and os.path.exists(dp)): continue
        im = np.array(Image.open(ip).convert("RGB")); dm = np.array(Image.open(dp)).astype(np.float32) / 1000.0
        imgs.append(im); deps.append(dm)
        exts.append(np.linalg.inv(traj[i]))          # world2cam = inv(cam2world)
        ints.append(K[i])
    imgs = np.stack(imgs); deps = np.stack(deps)
    # SpaceFormer cloud + GT masks + captions
    tf = tarfile.open(f"{SF}/{sc}/scene.tar")
    rd = lambda n: tf.extractfile([x for x in tf.getnames() if x.endswith(n)][0]).read()
    coord = np.load(io.BytesIO(rd("pcd.npz")))["coord"].astype(np.float64)
    pi = np.load(io.BytesIO(rd("point_indices.npz"))); packed = pi["packed"]; lengths = pi["lengths"]
    caps = json.load(io.BytesIO(rd("captions_v3.json"))); tf.close()
    N0 = len(coord); sub = np.arange(N0) if N0 <= NPTS else np.sort(np.random.default_rng(0).choice(N0, NPTS, replace=False))
    coord_ds = coord[sub]
    # instmask: per downsampled point -> captioned-instance id+1 (0 = none). obj_idx = index into caps['results']
    inv = -np.ones(N0, np.int64); off = np.concatenate([[0], np.cumsum(lengths)])
    obj2len = {r["obj_idx"]: i for i, r in enumerate(caps["results"])}   # obj_idx -> position in lengths
    for r in caps["results"]:
        oi = r["obj_idx"]
        if oi < len(lengths): inv[packed[off[oi]:off[oi + 1]]] = oi
    instmask = inv[sub]                               # (NPTS,) obj_idx or -1
    return dict(images=imgs, intrinsics=np.stack(ints), extrinsics=np.stack(exts), depth_maps=deps,
                points=coord_ds, super_points=np.arange(len(coord_ds)), instmask=instmask,
                caps=[(r["obj_idx"], r["name"], r["captions"]) for r in caps["results"]])

def main():
    model, _ = build_omega()
    model.query_decoder.decoder_norm.register_forward_hook(lambda m, i, o: _last.__setitem__("q", o.detach()))
    model.backbone.register_forward_hook(lambda m, i, o: _last.__setitem__("reg", o["tokens_list"]))   # capture per-frame register (for <REG> injection)
    scenes = [os.environ["SC"]] if os.environ.get("SC") else [s.strip() for s in open(os.environ["SCENE_LIST"])]
    TOL = float(os.environ.get("TOL", "0.1"))
    for sc in scenes:
        if os.path.exists(f"{OUT}/{sc}.npz"):
            try:
                z = np.load(f"{OUT}/{sc}.npz", allow_pickle=True)
                if "register" in z or "empty" in z: print(f"{sc}: skip (done)", flush=True); continue   # re-run only if missing register/idmap
            except Exception: pass
        try:
            s = load_scene(sc)
            masks2d, scores, qemb, register = seg_forward(model, s["images"])   # +register (S,17,2048)
            if len(qemb) == 0: np.savez_compressed(f"{OUT}/{sc}.npz", empty=True); continue
            inst3d = map_pred_inst_to_gt_pointcloud(s["points"], s["super_points"], s["intrinsics"],
                s["extrinsics"], s["depth_maps"], masks2d, depth_tolerance=TOL)   # (n, NPTS) bool
            idmap = project_gt_idmap(s["points"], s["instmask"], s["intrinsics"], s["extrinsics"], s["depth_maps"], tol=TOL)  # (S,MH,MW) GT obj_idx per pixel
            n, N = inst3d.shape
            # per-caption target = best-IoU segmenter instance vs that caption's GT points
            rows = []
            for oi, name, captions in s["caps"]:
                gt = (s["instmask"] == oi)
                if gt.sum() == 0: continue
                iou = (inst3d & gt).sum(1) / np.maximum((inst3d | gt).sum(1), 1)
                tgt = int(iou.argmax()); bi = float(iou.max())
                rows.append(dict(scene=sc, oid=int(oi), name=name, desc=captions[0], target=tgt, n=n, best_iou=round(bi, 3)))
            np.savez_compressed(f"{OUT}/{sc}.npz", qemb=qemb.astype(np.float32), scores=scores.astype(np.float32),
                register=register.astype(np.float16), idmap=idmap.astype(np.int16),   # +register +idmap for MFRAME training
                inst3d=np.packbits(inst3d, axis=1), n=n, N=N, instmask=s["instmask"].astype(np.int64),
                rows=json.dumps(rows))
            print(f"{sc}: n={n} N={N} caps={len(rows)} best_iou_mean={np.mean([r['best_iou'] for r in rows]):.3f}", flush=True)
        except Exception as e:
            import traceback; print(f"{sc}: FAIL {type(e).__name__}: {e}", flush=True); traceback.print_exc()

if __name__ == "__main__":
    main()
