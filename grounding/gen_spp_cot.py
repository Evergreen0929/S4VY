#!/usr/bin/env python
"""ScanNet++ Turn-1 CoT: the SAME recipe as build_cot_data.py (s4vy geometry -> Qwen3-VL-235B rewrite),
but CLIENT-SIDE ONLY -- no segmenter forward is needed because cache_scene.py already stored the s4vy
instance masks (`inst3d`, packbits) and the caption rows. Point coords are re-derived from the SpaceFormer
cloud with the SAME deterministic downsample (default_rng(0), NPTS) that produced inst3d's columns.

  1) geom facts = build_cot_data.geom_facts(points, inst3d, target) -- metric size / height-above-floor /
     distance+vertical relation to the 4 nearest other candidates, ScanNet world z-up (verified for ScanNet++:
     z spans 0-3.8 m = storey height).
  2) teacher = Qwen3-VL-235B-A22B-FP8 over an OpenAI-compatible vLLM endpoint, 4 uniformly-spread frames,
     gold target index given -> rationalizes the known answer (identical prompt to build_cot_data.teacher_cot).

KEY: output is keyed `{scene}__{oid}.json`, NOT by manifest row index. ScanRefer's cache uses `__{qid}` (=
enumerate index), which is the documented row-index trap: re-mining or filtering the manifest silently
invalidates every key. ScanNet++ has exactly one row per (scene,oid), so (scene,oid) is a stable key and the
cache survives any later manifest change (e.g. a best_iou filter or adding caption paraphrases).

CPU-only (GPU-free): run many shards next to the vLLM replicas. Env: SHARD/NSHARD (over scenes), PAR (threads),
COT_API_BASE (comma-separated -> round-robin over replicas), TEACHER_IMGS, OUT, SCENE_LIST."""
import os, sys, json, glob, io, tarfile, threading
import _env
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault("MANIFEST", "/dev/null")   # build_cot_data reads it at module level; we only want two helpers
from build_cot_data import geom_facts, _b64                      # reuse the EXACT geometry + image encoding

D = _env.data_root()
SF = os.environ.get("SPACEFORMER_SPP_ROOT", "/PATH/TO/datasets/spaceformer/scannetpp")
SPPC = os.environ.get("SPPC", f"{D}/scannetpp_cache_v2")   # v2 = native-mask GT + 2D-IoU targets; v1 was f"{D}/scannetpp_train_cache"
SPPP_2D = f"{D}/scannetpp_scenes_2d"
OUT = os.environ.get("OUT", f"{D}/scannetpp_cot_cache")
NPTS = int(os.environ.get("NPTS", "200000"))                     # MUST match cache_scene.py (inst3d columns)
TEACHER_IMGS = int(os.environ.get("TEACHER_IMGS", "4"))
SHARD = int(os.environ.get("SHARD", "0")); NSHARD = int(os.environ.get("NSHARD", "1"))
PAR = int(os.environ.get("PAR", "16"))
API_BASES = [b.strip() for b in os.environ.get("COT_API_BASE", "http://127.0.0.1:8000/v1").split(",") if b.strip()]
API_MODEL = os.environ.get("COT_API_MODEL", "qwen3vl")
IMG_W = int(os.environ.get("IMG_W", "512"))                      # match the trainer's frame downscale
_rr = [0]; _rrlock = threading.Lock()


def teacher_cot(frames_np, desc, facts, tgt):
    """Identical system/user prompt to build_cot_data.teacher_cot; round-robins over vLLM replicas."""
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
    with _rrlock:
        base = API_BASES[_rr[0] % len(API_BASES)]; _rr[0] += 1
    r = requests.post(f"{base}/chat/completions", timeout=300, json={
        "model": API_MODEL, "temperature": 0.4, "max_tokens": 220,
        "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": usr}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def scene_points(sc):
    """Re-derive the EXACT downsampled cloud whose points index inst3d's columns (cache_scene.load_scene)."""
    tf = tarfile.open(f"{SF}/{sc}/scene.tar")
    coord = np.load(io.BytesIO(tf.extractfile([x for x in tf.getnames() if x.endswith("pcd.npz")][0]).read()))["coord"].astype(np.float64)
    tf.close()
    N0 = len(coord)
    sub = np.arange(N0) if N0 <= NPTS else np.sort(np.random.default_rng(0).choice(N0, NPTS, replace=False))
    return coord[sub]


def scene_frames(sc):
    """Uniform spread over the symlinked frames (k order == register order == what the trainer/eval sees)."""
    d = f"{SPPP_2D}/{sc}/color"
    S = len(glob.glob(f"{d}/*.jpg"))
    if S == 0: return None
    idx = np.linspace(0, S - 1, min(S, TEACHER_IMGS)).round().astype(int)
    out = []
    for k in idx:
        im = Image.open(f"{d}/{int(k)}.jpg").convert("RGB")
        if im.width > IMG_W: im = im.resize((IMG_W, int(im.height * IMG_W / im.width)))
        out.append(np.asarray(im))
    return out


def do_scene(sc):
    made = skipped = failed = 0
    try:
        z = np.load(f"{SPPC}/{sc}.npz", allow_pickle=True)
        if "rows" not in z or "inst3d" not in z: return (0, 0, 0)
        rows = json.loads(str(z["rows"]))
        todo = [r for r in rows if not os.path.exists(f"{OUT}/{sc}__{r['oid']}.json")]
        if not todo: return (0, len(rows), 0)
        n, N = int(z["n"]), int(z["N"])
        inst3d = np.unpackbits(z["inst3d"], axis=1)[:, :N].astype(bool)      # (n, NPTS) s4vy instance masks
        points = scene_points(sc)
        if len(points) != N:
            print(f"{sc}: SKIP point/inst3d mismatch {len(points)} vs {N}", flush=True); return (0, 0, len(todo))
        frames = scene_frames(sc)
        if frames is None:
            print(f"{sc}: SKIP no frames", flush=True); return (0, 0, len(todo))
    except Exception as e:
        print(f"{sc}: LOAD FAIL {type(e).__name__}: {e}", flush=True); return (0, 0, 0)

    def one(r):
        tgt = int(r["target"])
        if not (0 <= tgt < n): return "fail"
        try:
            facts = geom_facts(points, inst3d, tgt)
            cot = teacher_cot(frames, r["desc"], facts, tgt)
            if not cot: return "fail"
            json.dump({"cot": cot, "target": tgt, "n": n, "facts": facts, "best_iou": r.get("best_iou"),
                       "name": r.get("name"), "desc": r["desc"]}, open(f"{OUT}/{sc}__{r['oid']}.json", "w"))
            return "ok"
        except Exception as e:
            print(f"{sc}__{r['oid']}: FAIL {type(e).__name__}: {e}", flush=True); return "fail"

    with ThreadPoolExecutor(max_workers=PAR) as ex:
        for res in ex.map(one, todo):
            if res == "ok": made += 1
            else: failed += 1
    skipped = len(rows) - len(todo)
    return (made, skipped, failed)


def main():
    if os.environ.get("SCENE_LIST"):
        scenes = [s.strip() for s in open(os.environ["SCENE_LIST"]) if s.strip()]
    else:
        scenes = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{SPPC}/*.npz"))
    mine = scenes[SHARD::NSHARD]
    print(f"[sppcot] shard {SHARD}/{NSHARD}: {len(mine)} scenes | PAR={PAR} | replicas={API_BASES} -> {OUT}", flush=True)
    M = S = F = 0
    for i, sc in enumerate(mine):
        m, s, f = do_scene(sc); M += m; S += s; F += f
        print(f"[{i+1}/{len(mine)}] {sc}: +{m} (skip {s}, fail {f}) | total made {M} skip {S} fail {F}", flush=True)
    print(f"[sppcot] DONE shard {SHARD}: made {M}, already-had {S}, failed {F}", flush=True)


if __name__ == "__main__": main()
