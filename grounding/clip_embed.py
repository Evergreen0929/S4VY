#!/usr/bin/env python
"""segmenter + CLIP grounding on ScanRefer val. Reuses align3d (segmenter forward + superpoint
map_pred) for 3D instance masks, and open_clip per-instance embeddings (like clip_ground)
to pick the instance matching each ScanRefer description. Writes per-scene query results;
aggregate separately. GT instance from aggregation.json (object_id)."""
import os, sys, json, re
import _env
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

WORK = os.environ.get("GROUNDING_REPO", os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, WORK)
import align3d as AE   # sets up SEGV path + imports; chdir(SEGV)
from align3d import build_omega, omega_predict, load_single_scene_data, map_pred_inst_to_gt_pointcloud

D = _env.data_root()
SR = os.environ.get("SCANREFER_ANNOTATIONS", f"{D}/scanrefer/annotations/scanrefer/ScanRefer_filtered.json")
RAW = os.environ.get("SCANNET_RAW_VAL_ROOT", f"{D}/scannet_raw_val/scans")
RAW_TR = os.environ.get("SCANNET_RAW_TRAIN_ROOT", f"{D}/scannet_raw_train/scans")
PTS = os.environ.get("SCANNET200_POINTS_ROOT", "/PATH/TO/codebase/SegVGGT/data_processing/scannet200/points")
VAL = os.environ.get("SCANNET_VAL_LIST", "/PATH/TO/datasets/scannet/scannetv2_val.txt")
OUT = os.environ.get("SR_OUT", f"{D}/scanrefer_clip")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"

_clip = None
def get_clip():
    global _clip
    if _clip is None:
        import open_clip
        m, _, pp = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        _clip = (m.to(DEV).eval(), pp, open_clip.get_tokenizer("ViT-B-32"))
    return _clip


@torch.no_grad()
def inst_embeds(masks2d, images_np, max_f=4, min_pix=300, margin=0.15):
    m, pp, _ = get_clip()
    n, S, h, w = masks2d.shape; H, W = images_np.shape[1:3]
    embs = {}
    for i in range(n):
        areas = masks2d[i].reshape(S, -1).sum(1)
        order = [f for f in np.argsort(-areas)[:max_f] if areas[f] >= min_pix]
        crops = []
        for f in order:
            mk = masks2d[i, f]
            if mk.shape != (H, W):
                mk = np.array(Image.fromarray(mk.astype(np.uint8)).resize((W, H), Image.NEAREST)).astype(bool)
            ys, xs = np.where(mk)
            if len(ys) == 0: continue
            dy = int((ys.max()-ys.min())*margin); dx = int((xs.max()-xs.min())*margin)
            y0, y1 = max(0, ys.min()-dy), min(H, ys.max()+dy); x0, x1 = max(0, xs.min()-dx), min(W, xs.max()+dx)
            c = images_np[f, y0:y1, x0:x1]
            if c.size: crops.append(pp(Image.fromarray(c.astype(np.uint8))))
        if crops:
            e = F.normalize(m.encode_image(torch.stack(crops).to(DEV)).float(), dim=-1).mean(0)
            embs[i] = F.normalize(e, dim=-1).cpu()
    return embs


@torch.no_grad()
def text_embed(prompts):
    m, _, tok = get_clip()
    return F.normalize(m.encode_text(tok(prompts).to(DEV)).float(), dim=-1).cpu()


def gt_masks(sc):
    # resolve the split per scene: the same call is now used for TRAIN scenes too (critic training data), and
    # hard-coding the val directory made every train scene fail with FileNotFoundError.
    root = RAW if os.path.exists(f"{RAW}/{sc}/{sc}.aggregation.json") else RAW_TR
    seg = np.array(json.load(open(f"{root}/{sc}/{sc}_vh_clean_2.0.010000.segs.json"))["segIndices"], np.int64)
    agg = json.load(open(f"{root}/{sc}/{sc}.aggregation.json"))
    return {str(g["objectId"]): np.isin(seg, np.array(g["segments"])) for g in agg["segGroups"]}, len(seg)


def aabb_iou(a, b):
    if len(a) == 0 or len(b) == 0: return 0.0
    lo = np.maximum(a.min(0), b.min(0)); hi = np.minimum(a.max(0), b.max(0))
    inter = np.prod(np.clip(hi-lo, 0, None)); va = np.prod(a.max(0)-a.min(0)); vb = np.prod(b.max(0)-b.min(0))
    return float(inter/max(va+vb-inter, 1e-9))


def main():
    valset = set(re.findall(r"scene[0-9]{4}_[0-9]{2}", open(VAL).read()))
    sr = [x for x in json.load(open(SR)) if x["scene_id"] in valset]
    by = {}
    for x in sr: by.setdefault(x["scene_id"], []).append(x)
    scenes = sorted(by)
    sl = os.environ.get("SCENE_LIST")
    if sl: keep = set(s.strip() for s in open(sl)); scenes = [s for s in scenes if s in keep]
    model, decode = build_omega()
    for sc in scenes:
        of = f"{OUT}/{sc}.json"
        if os.path.exists(of): continue
        try:
            sample = load_single_scene_data(AE.__dict__.get("root", "data_processing/scannet200"), sc, "scannet200", 1)
        except Exception:
            sample = load_single_scene_data("data_processing/scannet200", sc, "scannet200", 1)
        if sample is None: continue
        masks2d, scores = omega_predict(model, decode, sample["images"])   # (n,S,h,w)
        if masks2d.shape[0] == 0:
            json.dump([], open(of, "w")); continue
        inst3d = map_pred_inst_to_gt_pointcloud(sample["points"], sample["super_points"],
                    sample["intrinsics"], sample["extrinsics"], sample["depth_maps"], masks2d, depth_tolerance=0.1)  # (n,N)
        coord = np.fromfile(f"{PTS}/{sc}.bin", np.float32).reshape(-1, 6)[:, :3]
        embs = inst_embeds(masks2d, sample["images"])
        gtm, Nseg = gt_masks(sc)
        ek = sorted(embs)
        if not ek: json.dump([], open(of, "w")); continue
        E = torch.stack([embs[k] for k in ek])
        qs = by[sc]; descs = [q["description"] for q in qs]
        T = text_embed(descs)                                  # (Q,D)
        sims = (T @ E.t()).numpy()                             # (Q,n_emb)
        res = []
        for qi, q in enumerate(qs):
            gt = gtm.get(q["object_id"])
            if gt is None or gt.sum() == 0 or Nseg != inst3d.shape[1]: continue
            k = ek[int(sims[qi].argmax())]; pm = inst3d[k]
            inter = (pm & gt).sum(); union = (pm | gt).sum()
            res.append({"oid": q["object_id"], "name": q["object_name"],
                        "miou": round(float(inter/max(union, 1)), 3), "biou": round(aabb_iou(coord[pm], coord[gt]), 3)})
        json.dump(res, open(of, "w"))
        print(f"{sc}: {len(res)} queries scored", flush=True)


if __name__ == "__main__":
    main()
