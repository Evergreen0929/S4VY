#!/usr/bin/env python
"""E5 prep: per (scene,oid,qid) natural-language <think> CoT distilled from VGGT geometry + a VLM teacher.
For each query already in the R1-pred cache (r1pred_cache -> gtfr = the frames the agent will ground on):
  1) re-run segmenter on imgs[gtfr] -> candidate masks -> lift to 3D scene cloud (ScanNet world, z-up) -> per-candidate
     centroid / size / height-above-floor / 3D distances & vertical relations to the TARGET (deterministic geometry).
  2) ask the VLM teacher (Qwen3-VL-235B, OpenAI-compatible endpoint $COT_API_BASE) to write a SHORT first-person
     reasoning that cites appearance (from the frames) AND the geometry facts, concluding the target candidate.
     The gold target index is GIVEN -> the teacher rationalizes the known answer (no wrong conclusions).
Stores {cot: "<text>", target, gtfr, n} -> cot_cache/{scene}__{oid}__{qid}.json
Scene-sharded. Needs align3d.build_omega (the segmenter) plus the teacher endpoint up."""
import os, sys, json, base64, io, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import _env
from PIL import Image
from align3d import build_omega, load_single_scene_data, map_pred_inst_to_gt_pointcloud, SCORE_THR
D = _env.data_root()
ROOT = "data_processing/scannet200"
R1PRED = os.environ.get("R1PRED", f"{D}/r1pred_cache")
OUT = os.environ.get("OUT", f"{D}/cot_cache")
MANIFEST = os.environ.get("MANIFEST", "")
SHARD = int(os.environ.get("SHARD", "0")); NSHARD = int(os.environ.get("NSHARD", "1"))
TEACHER_IMGS = int(os.environ.get("TEACHER_IMGS", "4"))      # how many gtfr frames to show the teacher
API_BASE = os.environ.get("COT_API_BASE", "http://127.0.0.1:8000/v1")
API_MODEL = os.environ.get("COT_API_MODEL", "qwen3vl")       # vLLM --served-model-name
_h = {}


@torch.no_grad()
def segmenter_forward(model, images_np):
    t = torch.from_numpy(images_np).float().div(255.).permute(0, 3, 1, 2)
    t = F.interpolate(t, size=(384, 512), mode="bilinear", align_corners=False).cuda()
    fp = model.query_decoder.frame_pos
    if t.shape[0] > fp.num_embeddings:
        w = fp.weight.data; wn = F.interpolate(w.t()[None], size=t.shape[0], mode="linear", align_corners=False)[0].t().contiguous()
        ne = nn.Embedding(t.shape[0], w.shape[1]).to(w.device); ne.weight.data = wn; model.query_decoder.frame_pos = ne
    out = model(t)
    scores = out["objectness"][0].softmax(-1)[..., 0]; keep = scores > SCORE_THR
    ml = out["mask_logits"][0]; masks = (ml.sigmoid() > 0.5) & keep.view(-1, 1, 1, 1)
    keep = keep & (masks.flatten(1).sum(-1) > 10)
    return masks[keep].float().cpu().numpy()


def geom_facts(points, inst3d, tgt, topk=4):
    """Deterministic VGGT-geometry facts in ScanNet world coords (z = up). Avoid viewpoint-dependent
    left/right (the teacher infers those from the images); report distance + height + vertical relations + size."""
    points = points[:, :3]                                                             # sample["points"] is (N,6) xyz+rgb
    cen = np.stack([points[m].mean(0) if m.any() else np.zeros(3) for m in inst3d])    # (n,3)
    ext = np.stack([(points[m].max(0) - points[m].min(0)) if m.any() else np.zeros(3) for m in inst3d])
    floor = float(points[:, 2].min()); tc = cen[tgt]
    base = (float(points[inst3d[tgt]][:, 2].min()) - floor) if inst3d[tgt].any() else 0.0   # true bottom, not centroid
    L = [f"[Geometry from 3D reconstruction, metres] Target object (candidate #{tgt}): "
         f"size {ext[tgt,0]:.1f}(w) x {ext[tgt,1]:.1f}(d) x {ext[tgt,2]:.1f}(h); "
         f"its bottom is {base:.1f} m above the floor, centre {tc[2]-floor:.1f} m above the floor."]
    d = np.linalg.norm(cen - tc, axis=1)
    for j in np.argsort(d)[1:topk+1]:
        dz = cen[j, 2] - tc[2]
        v = "higher up" if dz > 0.3 else "lower down" if dz < -0.3 else "at a similar height"
        L.append(f"- Another object (candidate #{j}, size {ext[j,0]:.1f}x{ext[j,1]:.1f}x{ext[j,2]:.1f} m) "
                 f"is {d[j]:.1f} m from the target, {v}.")
    return "\n".join(L)


def _b64(arr):
    im = Image.fromarray(arr.astype("uint8")); b = io.BytesIO(); im.save(b, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def teacher_cot(frames_np, desc, facts, tgt):
    import requests
    sys_p = ("You are a spatial-reasoning expert. You are given several frames of an indoor scene, a referring "
             "description, 3D geometry facts, and the INDEX of the correct candidate object. Write ONE short "
             "first-person reasoning paragraph (2-4 sentences) explaining how the visual appearance in the frames "
             "AND the 3D geometry together identify the target. Cite at least one concrete geometric fact "
             "(distance / height / size) and one appearance cue. Do NOT mention the candidate index or that you "
             "were told the answer. End by stating which object is the target. Output ONLY the reasoning text.")
    usr = [{"type": "image_url", "image_url": {"url": _b64(f)}} for f in frames_np]
    usr.append({"type": "text", "text": f'Referring description: "{desc}"\n\n{facts}\n\n'
                f"The correct object is candidate #{tgt}. Write the reasoning now."})
    r = requests.post(f"{API_BASE}/chat/completions", timeout=180, json={
        "model": API_MODEL, "temperature": 0.4, "max_tokens": 220,
        "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": usr}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


@torch.no_grad()
def main():
    m9, decode = build_omega()
    m9.backbone.register_forward_hook(lambda m, i, o: _h.__setitem__("tl", o["tokens_list"]))
    records = [json.loads(l) for l in open(MANIFEST) if l.strip()]
    for i, r in enumerate(records): r["_qid"] = i
    scenes = sorted(set(r["scene"] for r in records))
    myscenes = set(scenes[SHARD::NSHARD])
    items = sorted([r for r in records if r["scene"] in myscenes], key=lambda r: r["scene"])
    print(f"[cot] shard {SHARD}/{NSHARD}: {len(myscenes)} scenes / {len(items)} queries -> {OUT}", flush=True)
    cur = None; sc_cache = {}
    for r in items:
        sc, oid, desc, qid = r["scene"], int(r["oid"]), r["desc"], r["_qid"]
        key = f"{sc}__{oid}__{qid}"
        if os.path.exists(f"{OUT}/{key}.json"): continue
        pf = f"{R1PRED}/{key}.npz"
        if not os.path.exists(pf): continue                          # only queries that have an R1-pred entry
        zp = np.load(pf)
        if "empty" in zp.files: continue
        try:
            if sc != cur:
                sc_cache = {"s": load_single_scene_data(ROOT, sc, "scannet200", 1)}; cur = sc
            sample = sc_cache["s"]
            if sample is None: continue
            gtfr = zp["gtfr"]; tgt = int(zp["target"]); imgs = sample["images"][gtfr]
            masks = segmenter_forward(m9, imgs)
            if tgt >= masks.shape[0]: continue                       # target index must exist in re-run instance set
            inst3d = map_pred_inst_to_gt_pointcloud(sample["points"], sample["super_points"],
                        sample["intrinsics"][gtfr], sample["extrinsics"][gtfr], sample["depth_maps"][gtfr],
                        masks, depth_tolerance=0.1)
            facts = geom_facts(sample["points"], inst3d, tgt)
            if os.environ.get("COT_DRY") == "1":                          # validate geometry without the teacher/vLLM
                print(f"{key}: n={masks.shape[0]} tgt={tgt}\n  desc: {desc}\n  {facts}", flush=True); continue
            shown = imgs[np.linspace(0, len(imgs)-1, min(len(imgs), TEACHER_IMGS)).round().astype(int)]
            cot = teacher_cot(shown, desc, facts, tgt)
            json.dump({"cot": cot, "target": tgt, "gtfr": gtfr.tolist(), "n": int(masks.shape[0]), "facts": facts},
                      open(f"{OUT}/{key}.json", "w"))
            print(f"{key}: n={masks.shape[0]} cot_len={len(cot)}", flush=True)
        except Exception as e:
            print(f"{key}: FAIL {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__": main()
