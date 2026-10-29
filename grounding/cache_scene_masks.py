import os
import _env
#!/usr/bin/env python
"""ScanNet++ cache v2 — replaces the 3D-projection GT with InsScene's NATIVE per-frame instance masks.

WHY (measured against the v1 cache):
  * v1's per-frame GT came from project_gt_idmap: SpaceFormer's laser cloud, downsampled to 200k points, splatted
    per frame under a |proj_z - depth| < 0.1 test. Paired against InsScene's own per-frame masks on 388
    (frame, object) pairs, that projection recovers a MEDIAN 0.625 of the native mask area; 41.8% of objects
    keep under half and 19.1% under a tenth. Rendered, the masks are dotted speckle, not solid regions.
  * ScanNet200 survives the same lossy projection (median 206 px/object, only 4.5% fall under VIS_T=20), but
    ScanNet++ objects are small (median 73 px) so 22.5% drop below the threshold -> the per-frame
    present/absent label, which is MFRAME's core supervision, silently flips to absent.

WHAT CHANGES:
  1. GT per frame = InsScene `refined_ins_ids` (dense, native resolution) instead of the point splat.
  2. The SpaceFormer obj_idx -> InsScene instance id BRIDGE is built by voting the old projection over ALL
     frames (single-frame agreement is 0.77; aggregating over frames is far tighter). Vote share is stored so
     low-confidence bridges can be dropped downstream.
  3. `target` (the qemb label) is chosen by video-level 2D IoU against that dense GT, not 3D IoU on the
     downsampled cloud. The 3D number is kept as `best_iou` for comparison; the new one is `best_iou2d`.
  4. s4vy's per-frame predicted masks are SAVED (packbits at the 96x128 idmap resolution) so eval no longer
     needs a segmenter re-run -- v1 computed masks2d and threw it away.

Env: SC | SCENE_LIST, BALANCED_RES=512, MAXF=96, NPTS=200000, OUT. Single scene: SC=00777c41d4 python cache_scene_masks.py"""
import os, json, tarfile, io, glob
os.environ.setdefault("BALANCED_RES", "512")
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from align3d import build_omega, map_pred_inst_to_gt_pointcloud, SCORE_THR
from cache_scene import _bal_hw, project_gt_idmap, load_scene, MW, MH, INS, SF, NPTS, MAXF, _last

_D = _env.data_root()
OUT = os.environ.get("OUT", f"{_D}/scannetpp_cache_v2")
MIN_VOTE = float(os.environ.get("MIN_VOTE", "0.5"))     # bridge accepted only if the winning native id takes this share
os.makedirs(OUT, exist_ok=True)


def frame_names(sc):
    """The EXACT frame list (and order) that cache_scene.load_scene keeps -> index k == register/idmap index."""
    md = np.load(f"{INS}/{sc}/scene_iphone_metadata.npz", allow_pickle=True)
    fn = md["images"]; S0 = len(md["trajectories"])
    idx = np.arange(S0) if S0 <= MAXF else np.linspace(0, S0 - 1, MAXF).round().astype(int)
    out = []
    for i in idx:
        f = str(fn[i]).replace(".png", ".jpg")
        if os.path.exists(f"{INS}/{sc}/images/{f}") and os.path.exists(f"{INS}/{sc}/depth/{f.replace('.jpg', '.png')}"):
            out.append(f)
    return out


def native_ins(sc, names):
    """InsScene per-frame instance ids -> (S,MH,MW) int32, NEAREST-downsampled to the idmap grid."""
    out = np.full((len(names), MH, MW), -1, np.int32)
    for k, f in enumerate(names):
        p = f"{INS}/{sc}/refined_ins_ids/{f}.npy"
        if not os.path.exists(p): continue
        a = np.load(p).astype(np.int32)
        out[k] = np.array(Image.fromarray(a).resize((MW, MH), Image.NEAREST))
    return out


def build_bridge(idmap_proj, ins2d, oids):
    """SpaceFormer obj_idx -> (native id, vote share), voted over EVERY frame of the old projection.
    A native id can only be claimed once (SpaceFormer sometimes splits finer than InsScene); the higher
    vote share wins and the loser stays unbridged (it falls back to the projection)."""
    votes = {}
    for o in oids:
        m = (idmap_proj == o)
        if not m.any(): continue
        v, c = np.unique(ins2d[m], return_counts=True)
        ok = v >= 0
        if not ok.any(): continue
        v, c = v[ok], c[ok]
        j = int(c.argmax()); votes[int(o)] = (int(v[j]), float(c[j] / c.sum()), int(c.sum()))
    claimed, bridge = {}, {}
    for o, (nid, share, tot) in sorted(votes.items(), key=lambda x: -x[1][1]):
        if share < MIN_VOTE: continue
        if nid in claimed: continue                    # already taken by a higher-confidence object
        claimed[nid] = o; bridge[o] = (nid, round(share, 3), tot)
    return bridge


@torch.no_grad()
def seg_forward_keepmasks(model, images_np):
    """Same forward as cache_scene.seg_forward, but the per-frame masks are KEPT (downsampled to MHxMW)."""
    S = images_np.shape[0]
    t = torch.from_numpy(images_np).float().div(255.).permute(0, 3, 1, 2)
    t = F.interpolate(t, size=_bal_hw(images_np), mode="bilinear", align_corners=False).to(os.environ.get("SEG_DEV", "cuda"))
    fp = model.query_decoder.frame_pos
    if S > fp.num_embeddings:
        import torch.nn as nn
        w = fp.weight.data; wn = F.interpolate(w.t()[None], size=S, mode="linear", align_corners=False)[0].t().contiguous()
        ne = nn.Embedding(S, w.shape[1]).to(w.device); ne.weight.data = wn; model.query_decoder.frame_pos = ne
    out = model(t)
    tl = [x for x in _last["reg"] if x is not None][-1]
    register = tl[0, :, :17].float().cpu().numpy()
    qf = _last["q"][0]; obj = out["objectness"][0]; ml = out["mask_logits"][0]        # (Q,S,h,w)
    scores = obj.softmax(-1)[..., 0]; keep = scores > SCORE_THR
    masks = (ml.sigmoid() > 0.5) & keep.view(-1, 1, 1, 1)
    keep = keep & (masks.flatten(1).sum(-1) > 10)
    masks = masks[keep]                                                              # (n,S,h,w) full res
    small = F.interpolate(masks.float(), size=(MH, MW), mode="area") > 0.5           # (n,S,MH,MW) idmap grid
    return (masks.float().cpu().numpy(), small.cpu().numpy(), scores[keep].cpu().numpy(), qf[keep].float().cpu().numpy(), register)


def iou_video(a, b):
    """video-level 2D IoU between two (S,MH,MW) bool masks (sum over frames -> temporal-aware, robust to
    a single bad frame, and it is exactly the quantity MFRAME's per-frame supervision cares about)."""
    inter = np.logical_and(a, b).sum(); uni = np.logical_or(a, b).sum()
    return float(inter / uni) if uni > 0 else 0.0


def main():
    model, _ = build_omega()
    model.query_decoder.decoder_norm.register_forward_hook(lambda m, i, o: _last.__setitem__("q", o.detach()))
    model.backbone.register_forward_hook(lambda m, i, o: _last.__setitem__("reg", o["tokens_list"]))
    scenes = [os.environ["SC"]] if os.environ.get("SC") else [s.strip() for s in open(os.environ["SCENE_LIST"]) if s.strip()]
    TOL = float(os.environ.get("TOL", "0.1"))
    for sc in scenes:
        if os.path.exists(f"{OUT}/{sc}.npz"):
            try:
                z = np.load(f"{OUT}/{sc}.npz", allow_pickle=True)
                if "idmap_native" in z or "empty" in z: print(f"{sc}: skip (done)", flush=True); continue
            except Exception: pass
        try:
            s = load_scene(sc)
            names = frame_names(sc)
            masks_full, masks2d, scores, qemb, register = seg_forward_keepmasks(model, s["images"])
            if len(qemb) == 0: np.savez_compressed(f"{OUT}/{sc}.npz", empty=True); continue
            S = masks2d.shape[1]
            if len(names) != S:
                print(f"{sc}: FAIL frame-name/register mismatch {len(names)} vs {S}", flush=True); continue
            inst3d = map_pred_inst_to_gt_pointcloud(s["points"], s["super_points"], s["intrinsics"],
                                                    s["extrinsics"], s["depth_maps"], masks_full, depth_tolerance=TOL)
            del masks_full
            idmap_proj = project_gt_idmap(s["points"], s["instmask"], s["intrinsics"], s["extrinsics"], s["depth_maps"], tol=TOL)
            ins2d = native_ins(sc, names)                                            # (S,MH,MW) native ids
            oids = [oi for oi, _, _ in s["caps"]]
            bridge = build_bridge(idmap_proj, ins2d, oids)
            idmap_native = np.full((S, MH, MW), -1, np.int16)                        # dense GT, SpaceFormer obj_idx valued
            for o, (nid, share, _) in bridge.items(): idmap_native[ins2d == nid] = o
            n = masks2d.shape[0]
            rows = []
            for oi, name, captions in s["caps"]:
                gt3d = (s["instmask"] == oi)
                iou3 = ((inst3d & gt3d).sum(1) / np.maximum((inst3d | gt3d).sum(1), 1)) if gt3d.sum() else np.zeros(n)
                br = bridge.get(int(oi))
                gt2 = (idmap_native == oi)
                if br is not None and gt2.any():
                    i2 = np.array([iou_video(masks2d[j], gt2) for j in range(n)])
                    tgt, bi2 = int(i2.argmax()), float(i2.max())
                else:                                                                # unbridged -> fall back to the projection
                    gt2 = (idmap_proj == oi)
                    i2 = np.array([iou_video(masks2d[j], gt2) for j in range(n)]) if gt2.any() else np.zeros(n)
                    tgt, bi2 = int(i2.argmax()), float(i2.max())
                rows.append(dict(scene=sc, oid=int(oi), name=name, desc=captions[0], descs=list(captions),
                                 target=tgt, n=n, best_iou2d=round(bi2, 3),
                                 best_iou=round(float(iou3.max()), 3) if gt3d.sum() else 0.0,
                                 target3d=int(iou3.argmax()) if gt3d.sum() else -1,
                                 bridged=br is not None, vote=(br[1] if br else 0.0),
                                 px_native=int(gt2.reshape(S, -1).sum(1).max()) if gt2.any() else 0))
            np.savez_compressed(f"{OUT}/{sc}.npz",
                qemb=qemb.astype(np.float32), scores=scores.astype(np.float32),
                register=register.astype(np.float16),
                idmap=idmap_native,                                                  # PRIMARY GT (native, dense)
                idmap_proj=idmap_proj.astype(np.int16),                              # legacy projection, kept for audit
                idmap_native=idmap_native,                                           # explicit alias / done-marker
                pred_masks=np.packbits(masks2d.reshape(n, -1), axis=1), pred_shape=np.array([n, S, MH, MW]),
                inst3d=np.packbits(inst3d, axis=1), n=n, N=inst3d.shape[1],
                instmask=s["instmask"].astype(np.int64),
                bridge=json.dumps({str(k): v for k, v in bridge.items()}), rows=json.dumps(rows))
            nb = sum(r["bridged"] for r in rows)
            print(f"{sc}: n={n} caps={len(rows)} bridged={nb}/{len(rows)} "
                  f"iou2d={np.mean([r['best_iou2d'] for r in rows]):.3f} iou3d={np.mean([r['best_iou'] for r in rows]):.3f} "
                  f"px_native_med={np.median([r['px_native'] for r in rows]):.0f}", flush=True)
        except Exception as e:
            import traceback; print(f"{sc}: FAIL {type(e).__name__}: {e}", flush=True); traceback.print_exc()


if __name__ == "__main__":
    main()
