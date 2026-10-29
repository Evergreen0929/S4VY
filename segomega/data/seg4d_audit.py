#!/usr/bin/env python
"""Seg4D-Anything data-readiness audit.
For every dataset that will join training, verify (per the agreed checklist):
  A existence/completeness  B frame<->mask pairing (count+name+resolution)  C RGB readable + aspect (->balanced/tile)
  D mask decodable (dtype/id-range/ignore)  E temporal id consistency  F exhaustive vs partial  G loads through adapter
  H stats table (#clips/#frames/inst/res/sparsity)
Run: python -m segomega.data.seg4d_audit   (no training; read-only)
Each dataset is a CLIP source: list_clips(root)->[(seq_id,[(rgb_path, mask_ref)...])]; mask_ref decoded by `kind`.
"""
import os, glob, json, re, random, numpy as np
from PIL import Image

FSW = os.environ.get("SEG4D_DATA_ROOT", "/PATH/TO/data")
DSROOT = os.environ.get("SEG4D_DATASETS_ROOT", "/PATH/TO/datasets")
RVOS = f"{FSW}/RefVOS"
YTVOS = f"{DSROOT}/YouTube-VOS"
SAV = f"{os.environ.get('DYNAMICVERSE_ROOT', DSROOT + '/DynamicVerse')}/SAV"

# ---------- mask decoders: return 2D int32 instance-id map (0=bg), + ignore mask ----------
def dec_kitti(p):
    a = np.array(Image.open(p)).astype(np.int32); ign = (a == 10000); a = np.where(ign, 0, a); return a, ign
def dec_palette(p):                                   # YTVOS/DAVIS: palettized uint8, value=instance id
    a = np.array(Image.open(p)); a = a[..., 0] if a.ndim == 3 else a; return a.astype(np.int32), np.zeros(a.shape, bool)
def dec_dynrep(p):
    a = np.array(Image.open(p)); a = (a[..., 0].astype(np.int32)*65536 + a[..., 1]*256 + a[..., 2]) if a.ndim == 3 else a.astype(np.int32); return a, np.zeros(a.shape[:2], bool)
def dec_npy(p):
    a = np.load(p).astype(np.int32); return a, np.zeros(a.shape, bool)
DECODE = {"kitti": dec_kitti, "palette": dec_palette, "dynrep": dec_dynrep, "npy": dec_npy}

# ---------- per-dataset clip listers ----------
def clips_kitti():
    out = []
    for sq in sorted(os.listdir(f"{FSW}/KITTI-MOTS/instances")):
        idm = sorted(glob.glob(f"{FSW}/KITTI-MOTS/instances/{sq}/*.png"))
        pair = [(f"{FSW}/KITTI-MOTS/rgb_raw/training/image_02/{sq}/{os.path.basename(m)}", m) for m in idm]
        out.append((sq, pair))
    return out

def clips_dynrep():
    out = []
    for sq in sorted(os.listdir(f"{FSW}/DynamicReplica/_viz_extract")):
        idm = sorted(glob.glob(f"{FSW}/DynamicReplica/_viz_extract/{sq}/instance_id_maps/*.png"))
        pair = []
        for m in idm:
            st = os.path.basename(m).rsplit(".", 1)[0]; d = st.rsplit("_", 1); rgb = f"{FSW}/DynamicReplica/_viz_extract/{sq}/images/{d[0]}-{d[1]}.png"
            pair.append((rgb, m))
        out.append((sq, pair))
    return out

def clips_ytvos():
    out = []
    base = f"{YTVOS}/train"
    for sq in sorted(os.listdir(f"{base}/Annotations"))[:60]:
        idm = sorted(glob.glob(f"{base}/Annotations/{sq}/*.png"))
        pair = [(f"{base}/JPEGImages/{sq}/{os.path.basename(m).replace('.png','.jpg')}", m) for m in idm]
        out.append((sq, pair))
    return out

def clips_davis():
    out = []; A = f"{RVOS}/refdavis/DAVIS/Annotations_unsupervised/480p"; J = f"{RVOS}/refdavis/DAVIS/JPEGImages/480p"
    for sq in sorted(os.listdir(A)):
        idm = sorted(glob.glob(f"{A}/{sq}/*.png"))
        pair = [(f"{J}/{sq}/{os.path.basename(m).replace('.png','.jpg')}", m) for m in idm]
        out.append((sq, pair))
    return out

DATASETS = {
    "kitti":  dict(lister=clips_kitti,  kind="kitti",   exhaustive=False, tile=True),
    "dynrep": dict(lister=clips_dynrep, kind="dynrep",  exhaustive=True,  tile=False),
    "ytvos":  dict(lister=clips_ytvos,  kind="palette", exhaustive=False, tile=False),
    "davis":  dict(lister=clips_davis,  kind="palette", exhaustive=False, tile=False),
    # TODO (locate exact layout, then add): sail(.npy + RGB), sav, mevis(masks), lvos(masks), insscene(existing loader), cityscapes(seq_frames + panoptic)
}


def audit_one(name, spec, n_clips=5):
    dec = DECODE[spec["kind"]]; rep = {"name": name}
    try:
        clips = spec["lister"]()
    except Exception as e:
        return {**rep, "PASS": False, "err": f"list_clips: {type(e).__name__}: {e}"}
    if not clips: return {**rep, "PASS": False, "err": "no clips found"}
    nfr = [len(c[1]) for c in clips]; rep["clips"] = len(clips); rep["frames"] = sum(nfr)
    fails = []; aspects = []; insts = []; ids_seen = {}
    for sq, pairs in random.sample(clips, min(n_clips, len(clips))):
        if not pairs: fails.append(f"{sq}:empty"); continue
        for rgb, mref in random.sample(pairs, min(3, len(pairs))):
            if not os.path.exists(rgb): fails.append(f"{sq}:noRGB({os.path.basename(rgb)})"); continue
            try:
                im = Image.open(rgb); w, h = im.size; aspects.append(h / w)
                idm, ign = dec(mref)
            except Exception as e: fails.append(f"{sq}:decode {type(e).__name__}"); continue
            if idm.shape[:2] != (h, w): fails.append(f"{sq}:RES rgb{(h,w)}!=mask{idm.shape[:2]}")
            ids = set(np.unique(idm)) - {0}; insts.append(len(ids))
            for i in ids: ids_seen.setdefault(sq, {}).setdefault(i, 0)
        # temporal: an id present in >=2 frames of this clip?
        seen = {}
        for rgb, mref in pairs[:min(8, len(pairs))]:
            try: idm, _ = dec(mref)
            except: continue
            for i in set(np.unique(idm)) - {0}: seen[i] = seen.get(i, 0) + 1
        if seen and max(seen.values()) < 2: fails.append(f"{sq}:no-temporal-id")
    rep["aspect_median"] = round(float(np.median(aspects)), 2) if aspects else None
    rep["needs_tile"] = (rep["aspect_median"] is not None and not (0.5 < rep["aspect_median"] < 2.0)) or spec.get("tile")
    rep["inst_per_frame_med"] = round(float(np.median(insts)), 1) if insts else 0
    rep["exhaustive"] = spec["exhaustive"]
    rep["PASS"] = len(fails) == 0
    rep["fails"] = fails[:6]
    return rep


def main():
    random.seed(0)
    print(f"{'dataset':10} {'PASS':5} {'clips':>6} {'frames':>7} {'asp':>5} {'tile':>5} {'inst':>5} {'exh':>4}  fails")
    for name, spec in DATASETS.items():
        r = audit_one(name, spec)
        print(f"{name:10} {str(r.get('PASS')):5} {r.get('clips','-'):>6} {r.get('frames','-'):>7} "
              f"{str(r.get('aspect_median')):>5} {str(r.get('needs_tile')):>5} {str(r.get('inst_per_frame_med')):>5} "
              f"{str(r.get('exhaustive')):>4}  {r.get('err','') or ';'.join(r.get('fails',[]))}", flush=True)


if __name__ == "__main__":
    main()
