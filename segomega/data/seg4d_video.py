#!/usr/bin/env python
"""Seg4D-Anything unified video dataset: extends InsSceneDataset with 4D/video subsets
(KITTI-MOTS, DynamicReplica, SAIL-VOS, YouTube-VOS, Ref-DAVIS, LVOS, SA-V, Cityscapes-VPS).

Design decisions (agreed with user):
- resize: aspect in OPEN (0.5,2.0) -> official VGGT 'balanced' (aspect-preserving, ~512^2 tokens);
  aspect outside -> OVERLAPPED TILING: pick one square-ish tile (random horizontal offset) and use it
  CONSISTENTLY across all frames of the clip (keeps temporal + real resolution). KITTI/Cityscapes only.
- sampling: dense/static -> random frames; sparse-dynamic (KITTI/SAIL) -> LARGE window (mult*max_frames
  span) + presence-aware so the clip actually contains objects and tracks them over long range.
- partial-annotation datasets (VOS/YTVOS/KITTI) set exhaustive=False + an ignore mask; the criterion
  then skips the no-object penalty for unmatched queries on those samples.
Output adds to the InsScene sample dict: 'ignore' (S,H,W bool), 'exhaustive' (bool).
"""
import os, glob, json, re, numpy as np, torch
from PIL import Image
from .insscene_dataset import InsSceneDataset

import sys
_VG = os.environ.get("VGGT_OMEGA_ROOT", "/PATH/TO/codebase/vggt-omega")
if _VG not in sys.path: sys.path.insert(0, _VG)
from vggt_omega.utils.load_fn import _balanced_target_shape   # official balanced resize

# Dataset roots. Override each with the matching env var; see docs/DATA.md for what belongs where.
FSW = os.environ.get("SEG4D_DATA_ROOT", "/PATH/TO/data")          # our own preprocessed corpora
DSROOT = os.environ.get("SEG4D_DATASETS_ROOT", "/PATH/TO/datasets")  # third-party datasets, as downloaded
RVOS = f"{FSW}/RefVOS"
YTVOS = f"{DSROOT}/YouTube-VOS"
DVERSE = os.environ.get("DYNAMICVERSE_ROOT", f"{DSROOT}/DynamicVerse")  # unified 4D corpus (rgb/ + mask/ per seq)
SAV = f"{DVERSE}/SAV"
SAVX = f"{FSW}/SAV-extra/seqs"     # SA-V shards preprocessed to rgb/ + mask/ (RGB-coded)
OFFICIAL = os.environ.get("SEG4D_OFFICIAL_VIDEO_ROOT", f"{FSW}/official_video")  # official releases, replacing the DynamicVerse repacks
DL3DV = f"{DSROOT}/dl3dv-10k"                    # DL3DV-10K: <group>/<scene>/images_8/*.png -- NO seg GT (pure-pseudo, the frozen-teacher variant)
MATTERPORT2D = f"{DSROOT}/matterport3d/matterport_2d"  # Matterport3D 2D: <scene>/color/*.jpg -- 2D seg GT absent (pure-pseudo)
VIPSEG = f"{OFFICIAL}/VIPSeg/VIPSeg"   # official VIPSeg (CVPR22): imgs/<seq>/<f>.jpg + panomasks/<seq>/<f>.png (uint16 panoptic, EXHAUSTIVE)
UVO_SEQS = f"{OFFICIAL}/UVO/seqs"      # official UVO-Dense -> rgb/+mask/ (rgbid), exhaustive class-agnostic; see scripts/uvo_dense_to_seqs.py
KITTI_STEP = f"{OFFICIAL}/KITTI-STEP/kitti-step/panoptic_maps"  # KITTI-STEP panoptic_maps/<split>/<seq>/<f>.png (R=trainId, G*256+B=inst, 255=void); EXHAUSTIVE 19-class
CVPS_SEQS = f"{FSW}/Cityscapes-VPS/cvps_seqs"       # OLD thing-only (8 classes from instances RLE); deprecated for training
CVPS_EXH_SEQS = f"{FSW}/Cityscapes-VPS/cvps_exh_seqs"  # EXHAUSTIVE panoptic (thing inst + stuff classes), see scripts/cvps_panoptic_to_seqs.py
JRDB_SEQS = f"{OFFICIAL}/JRDB-PanoTrack/seqs"  # JRDB-PanoTrack PERSPECTIVE panoptic -> rgb/+mask/ (rgbid), EXHAUSTIVE 72-class; see scripts/jrdb_panoptic_to_seqs.py
MOSE_OFFICIAL = f"{OFFICIAL}/MOSE/MOSE_release"  # official MOSE v1: train/{JPEGImages,Annotations}/<seq> (DAVIS-style palette PNG, partial VOS)
MEVIS_SEQS = f"{FSW}/RefVOS/mevis/seqs"  # MeViS motion-referring VOS -> rgb/+mask/ (rgbid, anno_id=instance), partial; see scripts/mevis_to_seqs.py


# ---------- mask decoders: path -> (2D int32 id map [0=bg], ignore bool mask) ----------
def _dec_kitti(p):
    a = np.array(Image.open(p)).astype(np.int32); ign = a == 10000; return np.where(ign, 0, a), ign
def _dec_palette(p):
    a = np.array(Image.open(p)); a = a[..., 0] if a.ndim == 3 else a; return a.astype(np.int32), np.zeros(a.shape[:2] if a.ndim>1 else a.shape, bool)
def _dec_rgbid(p):                                            # dynrep / SA-V: RGB-coded ids
    a = np.array(Image.open(p).convert("RGB")).astype(np.int32); flat = a[..., 0]*65536 + a[..., 1]*256 + a[..., 2]; return flat, np.zeros(flat.shape, bool)
def _dec_npy(p):
    a = np.load(p).astype(np.int32); return a, np.zeros(a.shape, bool)
def _dec_pano16(p):                                          # VIPSeg / Cityscapes panoptic: uint16 PNG, 0=void(ignore),
    a = np.array(Image.open(p)).astype(np.int32)             # >0 stuff(cat_id) or thing(cat_id*100+inst). value IS the
    if a.ndim == 3: a = a[..., 0]                            # temporally-consistent instance id (exhaustive: 100% non-void)
    return a, (a == 0)                                       # void(0) -> ignore mask (not bg, not penalised)
def _dec_step(p):                                            # KITTI-STEP / STEP panoptic: RGB PNG, R=trainId(255=void), inst=G*256+B
    a = np.array(Image.open(p).convert("RGB")).astype(np.int32)
    R = a[..., 0]; inst = a[..., 1] * 256 + a[..., 2]
    idm = (R + 1) * 1000 + inst                              # +1 so road(trainId 0) -> 1000, NOT 0 (=bg). stuff: (cls+1)*1000; thing: +inst
    ign = (R == 255); idm[ign] = 0                           # void -> ignore (id 0)
    return idm, ign
def _dec_nogt(p):                                           # pure-pseudo scene (dl3dv/matterport): NO seg GT.
    # all-bg id map (0 instances) + all-False ignore -> the clip carries no real GT; the EMA teacher supplies
    # all supervision. Tiny placeholder is resized to the frame by _aspect_resize_pair (zeros stay zeros).
    return np.zeros((16, 16), np.int32), np.zeros((16, 16), bool)
def _dec_pseudopng(p):                                       # offline pseudo (dump_prompt_pseudo.py): uint16 PNG id map
    a = np.array(Image.open(p)).astype(np.int32)             # ids already compact 1..K (0=bg); filtered at dump time
    return a, np.zeros_like(a, bool)                         # no ignore region (only used for PROMPT supervision)
DECODE = {"kitti": _dec_kitti, "palette": _dec_palette, "rgbid": _dec_rgbid, "npy": _dec_npy, "pano16": _dec_pano16, "step": _dec_step, "nogt": _dec_nogt, "pseudopng": _dec_pseudopng}


def _gt_over_pseudo(real, pseudo, iou_thr=0.3):
    """real,pseudo: (S,H,W) int track-id maps. Real GT tracks OVERRIDE matched pseudo tracks (by temporal IoU);
    unmatched pseudo tracks FILL where real is bg. Returns merged (S,H,W) with contiguous ids (real first)."""
    rids = [int(i) for i in np.unique(real) if i > 0]
    pids = [int(i) for i in np.unique(pseudo) if i > 0]
    if not rids:
        return pseudo
    rm = {r: (real == r) for r in rids}; rsz = {r: int(rm[r].sum()) for r in rids}
    drop = set()
    for p in pids:
        pmask = pseudo == p; psz = int(pmask.sum())
        if psz == 0:
            continue
        for r in rids:
            inter = int((pmask & rm[r]).sum())
            if inter / (psz + rsz[r] - inter + 1e-6) > iou_thr:
                drop.add(p); break
    out = np.zeros_like(pseudo); nid = 1
    for r in rids:
        out[rm[r]] = nid; nid += 1                            # real GT painted first = priority
    for p in pids:
        if p in drop:
            continue
        m = (pseudo == p) & (out == 0)                        # pseudo fills only where real is bg
        if m.any():
            out[m] = nid; nid += 1
    return out


# ---- official train/val split isolation for EXHAUSTIVE panoptic sets (keep val OUT of training) ----
_VIPSEG_SPLIT = {}
def _vipseg_split_of(seq):
    if not _VIPSEG_SPLIT:
        for sp in ("train", "val", "test"):
            p = f"{VIPSEG}/{sp}.txt"
            if os.path.exists(p):
                for l in open(p):
                    if l.strip(): _VIPSEG_SPLIT[l.strip()] = sp
    return _VIPSEG_SPLIT.get(seq, "train")    # unknown seq -> treat as train (kept in training, not eval)

_DAVIS_SPLIT = {}
def _davis_split_of(seq):
    if not _DAVIS_SPLIT:
        base = f"{RVOS}/refdavis/DAVIS/ImageSets/2017"
        for sp in ("train", "val"):
            p = f"{base}/{sp}.txt"
            if os.path.exists(p):
                for l in open(p):
                    if l.strip(): _DAVIS_SPLIT[l.strip()] = sp
    return _DAVIS_SPLIT.get(seq, "train")     # DAVIS-17: train 60 / val 30

_LVOS_SPLIT = {}
def _lvos_split_of(seq):
    if not _LVOS_SPLIT:
        for sp, tag in (("train", "train"), ("valid", "val")):
            d = f"{RVOS}/lvos/{sp}/Annotations"
            if os.path.isdir(d):
                for s in os.listdir(d): _LVOS_SPLIT[s] = tag
    return _LVOS_SPLIT.get(seq, "val")        # LVOS: train/ (downloaded) vs valid/ (held-out eval)

def _clip_split(subset, seq):
    """Official train/val/test membership so val can be held out of training. 'all' = subset has no official
    split we isolate (pseudo sets) -> kept under ANY filter. vipseg/cityscapes/kitti/jrdb/davis/lvos isolated."""
    if subset == "vipseg":
        return _vipseg_split_of(seq)
    if subset in ("cityscapes", "kitti"):     # seq id is 'train_XXXX' / 'val_XXXX'
        s = seq.split("_", 1)[0]
        return s if s in ("train", "val", "test") else "train"
    if subset == "jrdb":                       # JRDB-PanoTrack: only train labels are public -> no val to hold out
        return "train"
    if subset == "davis":
        return _davis_split_of(seq)
    if subset == "lvos":
        return _lvos_split_of(seq)
    return "all"                               # ytvos/mose/uvo/mevis/.../dl3dv/matterport -> unaffected


# ---------- clip listers: -> [(seq_id, [(rgb_path, mask_path)...])] ----------
def _clips_kitti():
    r = f"{FSW}/KITTI-MOTS"; out = []
    for sq in sorted(os.listdir(f"{r}/instances")):
        idm = sorted(glob.glob(f"{r}/instances/{sq}/*.png"))
        out.append((sq, [(f"{r}/rgb_raw/training/image_02/{sq}/{os.path.basename(m)}", m) for m in idm]))
    return out

def _clips_kitti_step():  # KITTI-STEP: panoptic_maps/<split>/<seq>/<f>.png <-> KITTI rgb image_02/<seq>/<f>.png (EXHAUSTIVE)
    rgb_root = f"{FSW}/KITTI-MOTS/rgb_raw/training/image_02"; out = []
    for split in ("train", "val"):
        sd = f"{KITTI_STEP}/{split}"
        for sq in sorted(os.listdir(sd)) if os.path.isdir(sd) else []:
            pr = [(f"{rgb_root}/{sq}/{os.path.basename(m)}", m) for m in sorted(glob.glob(f"{sd}/{sq}/*.png"))]
            pr = [(r, m) for r, m in pr if os.path.exists(r)]
            if pr: out.append((f"{split}_{sq}", pr))
    return out

def _clips_dynrep(root):  # root = extracted dir
    out = []
    for sq in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        idm = sorted(glob.glob(f"{root}/{sq}/instance_id_maps/*.png"))
        pr = []
        for m in idm:
            st = os.path.basename(m).rsplit(".", 1)[0]; d = st.rsplit("_", 1)
            pr.append((f"{root}/{sq}/images/{d[0]}-{d[1]}.png", m))
        if pr: out.append((sq, pr))
    return out

def _clips_seqdir(root):  # generic <root>/<seq>/{rgb/*.jpg|png, mask/*.png} lister (DynamicVerse / our SA-V)
    out = []
    for sq in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        pr = []
        for r in sorted(glob.glob(f"{root}/{sq}/rgb/*.jpg") + glob.glob(f"{root}/{sq}/rgb/*.png")):
            m = f"{root}/{sq}/mask/{os.path.basename(r).rsplit('.', 1)[0]}.png"
            if os.path.exists(m): pr.append((r, m))
        if pr: out.append((sq, pr))
    return out

def _clips_vipseg():  # official VIPSeg: imgs/<seq>/<f>.jpg <-> panomasks/<seq>/<f>.png (uint16 panoptic, exhaustive)
    out = []
    for sq in sorted(os.listdir(f"{VIPSEG}/imgs")) if os.path.isdir(f"{VIPSEG}/imgs") else []:
        pr = [(r, f"{VIPSEG}/panomasks/{sq}/{os.path.basename(r).rsplit('.',1)[0]}.png")
              for r in sorted(glob.glob(f"{VIPSEG}/imgs/{sq}/*.jpg"))]
        pr = [(r, m) for r, m in pr if os.path.exists(m)]
        if pr: out.append((sq, pr))
    return out

def _clips_dverse(sub):  # DynamicVerse unified: <DVERSE>/<sub>/<seq>/{rgb/*.jpg, mask/*.png} (RGB-coded ids)
    root = f"{DVERSE}/{sub}"; out = []
    for sq in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        pr = []
        for r in sorted(glob.glob(f"{root}/{sq}/rgb/*.jpg") + glob.glob(f"{root}/{sq}/rgb/*.png")):
            m = f"{root}/{sq}/mask/{os.path.basename(r).rsplit('.', 1)[0]}.png"
            if os.path.exists(m): pr.append((r, m))
        if pr: out.append((sq, pr))
    return out

def _clips_sail(root):  # root = extracted dir with <vid>/{images/*.bmp, visible/*.npy}
    out = []
    for sq in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        npys = sorted(glob.glob(f"{root}/{sq}/visible/*.npy"))
        pr = [(f"{root}/{sq}/images/{os.path.basename(n).replace('.npy','.bmp')}", n) for n in npys]
        if pr: out.append((sq, pr))
    return out

def _clips_ytvos():
    b = f"{YTVOS}/train"; out = []
    for sq in sorted(os.listdir(f"{b}/Annotations")):
        idm = sorted(glob.glob(f"{b}/Annotations/{sq}/*.png"))
        out.append((sq, [(f"{b}/JPEGImages/{sq}/{os.path.basename(m)[:-4]}.jpg", m) for m in idm]))
    return out

def _clips_mose():  # official MOSE v1: train/{JPEGImages,Annotations}/<seq>/<f>.{jpg,png} (DAVIS-style palette, partial VOS).
    b = f"{MOSE_OFFICIAL}/train"; out = []                 # train only (valid masks are first-frame-only, withheld)
    if not os.path.isdir(f"{b}/Annotations"): return out
    for sq in sorted(os.listdir(f"{b}/Annotations")):
        if sq.startswith("._"): continue                   # skip macOS AppleDouble files
        idm = sorted(glob.glob(f"{b}/Annotations/{sq}/*.png"))
        pr = [(f"{b}/JPEGImages/{sq}/{os.path.basename(m)[:-4]}.jpg", m) for m in idm]
        pr = [(r, m) for r, m in pr if os.path.exists(r)]
        if pr: out.append((sq, pr))
    return out

def _clips_davis():
    A = f"{RVOS}/refdavis/DAVIS/Annotations_unsupervised/480p"; J = f"{RVOS}/refdavis/DAVIS/JPEGImages/480p"; out = []
    for sq in sorted(os.listdir(A)):
        idm = sorted(glob.glob(f"{A}/{sq}/*.png"))
        out.append((sq, [(f"{J}/{sq}/{os.path.basename(m)[:-4]}.jpg", m) for m in idm]))
    return out

def _clips_lvos():
    out = []
    for sp in ("train", "valid"):     # valid = held-out eval; train (downloaded) = for pseudo training. _lvos_split_of tags them
        A = f"{RVOS}/lvos/{sp}/Annotations"; J = f"{RVOS}/lvos/{sp}/JPEGImages"
        for sq in sorted(os.listdir(A)) if os.path.isdir(A) else []:
            idm = sorted(glob.glob(f"{A}/{sq}/*.png"))
            out.append((sq, [(f"{J}/{sq}/{os.path.basename(m)[:-4]}.jpg", m) for m in idm]))
    return out

def _clips_sav():
    out = []
    for sq in sorted(os.listdir(SAV)) if os.path.isdir(SAV) else []:
        msk = sorted(glob.glob(f"{SAV}/{sq}/mask/*.png"))
        out.append((sq, [(f"{SAV}/{sq}/rgb/{os.path.basename(m)}", m) for m in msk]))
    return out

def _clips_cityscapes():
    F = f"{FSW}/Cityscapes-VPS/seq_frames"; out = []
    # group seq_frames by city_seq -> ordered frames; panoptic label png matched by stem
    if not os.path.isdir(F): return out
    frames = {}
    for p in sorted(glob.glob(f"{F}/*_leftImg8bit.png")):
        m = re.search(r"([a-z]+)_(\d{6})_(\d{6})_leftImg8bit", os.path.basename(p))
        if m: frames.setdefault(f"{m.group(1)}_{m.group(2)}", []).append(p)
    LAB = glob.glob(f"{FSW}/Cityscapes-VPS/**/train/*.png", recursive=True) + glob.glob(f"{FSW}/Cityscapes-VPS/**/val/*.png", recursive=True)
    lab_by = {re.search(r"([a-z]+_\d{6}_\d{6})", os.path.basename(l)).group(1): l for l in LAB if re.search(r"([a-z]+_\d{6}_\d{6})", os.path.basename(l))}
    for seq, fs in frames.items():
        pr = []
        for p in sorted(fs):
            k = re.search(r"([a-z]+_\d{6}_\d{6})", os.path.basename(p)).group(1)
            if k in lab_by: pr.append((p, lab_by[k]))
        if pr: out.append((seq, pr))
    return out


def _cached_clips(name, builder):
    """Read a pre-built clip manifest json (scripts/build_pseudo_manifests.py) to avoid 100+ workers each
    globbing ~10k dirs at startup (lustre metadata storm). Falls back to building (+caching) if absent."""
    cache = f"{FSW}/_clip_manifests/{name}.json"
    if os.path.exists(cache):
        try:
            return [(sid, [tuple(x) for x in pairs]) for sid, pairs in json.load(open(cache))]
        except Exception:
            pass
    clips = builder()
    try:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tmp = f"{cache}.{os.getpid()}.tmp"; json.dump(clips, open(tmp, "w")); os.replace(tmp, cache)
    except Exception:
        pass
    return clips

def _build_dl3dv():   # DL3DV-10K: <group>/<scene>/images_8/*.{png,jpg} -> (scene_id, [(rgb, rgb=nogt placeholder)])
    out = []
    for g in sorted(glob.glob(f"{DL3DV}/*/")):
        for sc in sorted(glob.glob(f"{g}*/")):
            fs = sorted(glob.glob(f"{sc}images_8/*.png") + glob.glob(f"{sc}images_8/*.jpg"))
            if len(fs) >= 6:
                sid = f"{os.path.basename(g.rstrip('/'))}_{os.path.basename(sc.rstrip('/'))[:16]}"
                out.append((sid, [(f, f) for f in fs]))   # mask path = rgb path (dec_nogt ignores content)
    return out

def _build_matterport():   # Matterport3D 2D: <scene>/color/*.jpg -> (scene_id, [(rgb, rgb)])
    out = []
    for sc in sorted(glob.glob(f"{MATTERPORT2D}/*/")):
        fs = sorted(glob.glob(f"{sc}color/*.jpg"))
        if len(fs) >= 6:
            out.append((os.path.basename(sc.rstrip('/')), [(f, f) for f in fs]))
    return out

def _clips_dl3dv():      return _cached_clips("dl3dv", _build_dl3dv)
def _clips_matterport(): return _cached_clips("matterport", _build_matterport)


# subset -> (lister, kind, exhaustive, sparse)
VIDEO_SUBSETS = {
    "kitti":      (_clips_kitti_step,             "step",    True,  True),   # KITTI-STEP panoptic (19-class, EXHAUSTIVE) -- replaces sparse KITTI-MOTS (car/ped only)
    "kitti_mots": (_clips_kitti,                  "kitti",   False, True),   # OLD KITTI-MOTS (sparse car/ped), kept for reference
    "dynrep":     (lambda: _clips_dynrep(os.environ.get("DYNREP_DIR", f"{FSW}/DynamicReplica/_extract")), "rgbid", False,  False),
    "sail":       (lambda: _clips_sail(os.environ.get("SAIL_DIR", f"{FSW}/SAIL-VOS-original/_extract")),   "npy",   False,  True),
    "ytvos":      (_clips_ytvos,                  "palette", False, False),
    "davis":      (_clips_davis,                  "palette", False, False),
    "lvos":       (_clips_lvos,                   "palette", False, False),
    "sav":        (_clips_sav,                    "rgbid",   False, False),  # SA-V masklets are NOT exhaustive -> partial (no ∅ penalty on unmatched)
    "cityscapes": (lambda: _clips_seqdir(CVPS_EXH_SEQS), "rgbid", True,  False),  # EXHAUSTIVE panoptic (thing inst + stuff classes), 100% labelled -> provides ∅ signal; wide 2048x1024 -> 3-tile
    # ---- official re-downloads (replace DynamicVerse repacks; panoptic = EXHAUSTIVE -> provides ∅ signal) ----
    "vipseg":     (_clips_vipseg,                  "pano16",  True,  False),  # official VIPSeg CVPR22: full panoptic (thing+stuff, tracked), 100% non-void -> EXHAUSTIVE
    "jrdb":       (lambda: _clips_seqdir(JRDB_SEQS), "rgbid", True,  False),  # JRDB-PanoTrack perspective (752x480, 5 cam/seq): full panoptic 72-class (thing tracking_id + stuff) -> EXHAUSTIVE
    # ---- DynamicVerse unified corpus (DEPRECATED repacks: sparse/curated, NOT exhaustive; being replaced by official) ----
    "uvo":        (lambda: _clips_seqdir(UVO_SEQS),         "rgbid", False,  False),  # official UVO-Dense: OBJECT-exhaustive (all objects labelled) but NOT pixel-exhaustive (background/stuff unlabelled) -> exhaustive=False so stuff-queries are NOT forced to ∅ (avoids conflict with VIPSeg where stuff IS an instance). ∅ signal comes from VIPSeg/Cityscapes (pixel-exhaustive).
    "uvo_dverse": (lambda: _clips_dverse("uvo"),           "rgbid", False,  False),  # DEPRECATED DynamicVerse repack (sparse 1-2/frame), kept for reference only
    "mose":       (_clips_mose,                    "palette", False, False),  # official MOSE v1 (complex VOS, partial); DAVIS-style palette PNG
    "mevis":      (lambda: _clips_seqdir(MEVIS_SEQS), "rgbid", False, False),  # MeViS motion-referring VOS (partial; anno_id = temporally-consistent instance)
    "mose_dverse":(lambda: _clips_dverse("MOSE"),          "rgbid", False, False),  # DEPRECATED DynamicVerse repack
    "vost":       (lambda: _clips_dverse("VOST"),          "rgbid", False, False),  # transformations (partial)
    "podyssey":   (lambda: _clips_dverse("point_odyssey"), "rgbid", False,  False),  # synthetic dense
    "mvssynth":   (lambda: _clips_dverse("MVS-Synth"),     "rgbid", False,  False),  # synthetic multi-view dense
    "spring":     (lambda: _clips_dverse("spring"),        "rgbid", False,  False),  # synthetic dense
    "savx":       (lambda: _clips_seqdir(SAVX) + _clips_sav(), "rgbid", False, False),  # our SA-V 5-shard + 13 DynamicVerse SAV merged (same rate)  # our SA-V shards (manual+auto masklets); partial
    # ---- PURE-PSEUDO 3D-scene multi-view (NO seg GT at all; EMA teacher supplies all labels). the frozen-teacher variant, sparse window for multi-view coverage. ----
    "dl3dv":      (_clips_dl3dv,      "nogt", False, True),   # DL3DV-10K ~9.7k real captured scenes, no seg GT
    "matterport": (_clips_matterport, "nogt", False, True),   # Matterport3D 90 indoor scenes, 2D seg GT absent
}
TILE_SUBSETS = {"kitti", "cityscapes"}     # aspect outside (0.5,2.0) -> overlapped tiling


def balanced_hw(h, w, res=512, patch=16):
    return _balanced_target_shape(h / max(w, 1), res, patch)   # (target_h, target_w), aspect-preserving ~1024 tokens

def _tile_offsets(w, h, n=3):
    """n FIXED horizontal square tiles (side=min(h,w)) spanning the full width: left/center/right, overlapping.
    Wide frame (KITTI/Cityscapes) -> 3 tiles cover the whole FOV (vs one random crop)."""
    side = min(h, w)
    if w <= side: return [0]
    return sorted(set(int(round(x)) for x in np.linspace(0, w - side, n)))


class Seg4dVideoMixin:
    """Mix into InsSceneDataset; adds video subsets + aspect-aware resize + ignore/exhaustive + large-window sampling."""
    def _seg4d_init(self, video_subsets, tile_overlap=0.2, sparse_window_mult=8, video_weights=None, video_split="all",
                    pseudo_prompt_root=None, pseudo_prompt_subsets=(),
                    gt_pseudo_root=None, gt_pseudo_subsets=()):
        self.tile_overlap = tile_overlap; self.sparse_window_mult = sparse_window_mult
        self._vclips = {}; vw = video_weights or {}
        # GT-OVER-PSEUDO: non-exh subsets WITH real GT keep their (val-isolated) real-GT clips, and at load
        # time merge in the v3 pseudo (real covers matched instances, pseudo fills the rest). rgb->pseudo-png map, cached.
        self._gt_fill = {}
        for s in (gt_pseudo_subsets or ()):
            if gt_pseudo_root is None or s not in video_subsets:
                continue
            import glob as _glob, json as _json
            cache = os.path.join(gt_pseudo_root, s, "_rgb2png.json")
            mp = None
            if os.path.exists(cache):
                try: mp = _json.load(open(cache))
                except Exception: mp = None
            if mp is None:
                mp = {}
                for m in sorted(_glob.glob(os.path.join(gt_pseudo_root, s, "*", "meta.json"))):
                    try: meta = _json.load(open(m))
                    except Exception: continue
                    if meta.get("skip") or not meta.get("frames"): continue
                    d = os.path.dirname(m)
                    for f in meta["frames"]:
                        mp[f["rgb"]] = os.path.join(d, f["png"])
                try:
                    tmp = cache + f".tmp{os.getpid()}"; _json.dump(mp, open(tmp, "w")); os.replace(tmp, cache)
                except Exception: pass
            self._gt_fill[s] = mp
            print(f"[seg4d] gt-over-pseudo {s}: {len(mp)} rgb->pseudo frames from {gt_pseudo_root}", flush=True)
        # (gated, default off): nogt subsets (dl3dv/matterport) swap in offline pseudo id-map PNGs.
        # Clips are rebuilt from the dump's meta.json (labeled frames only); unlabeled/skipped scenes are dropped.
        self._pp_active = set()
        for s in video_subsets:
            lister, kind, exh, sparse = VIDEO_SUBSETS[s]
            clips = [c for c in lister() if c[1]]
            if video_split and video_split != "all":     # keep only clips of this official split (val held out of train)
                clips = [c for c in clips if _clip_split(s, c[0]) in (video_split, "all")]
            if pseudo_prompt_root and s in (pseudo_prompt_subsets or ()):
                # GLOB-based rebuild + SINGLE-FILE INDEX CACHE: without it every rank of every link
                # re-reads 54k meta.json (64 ranks x 54k = lustre metadata storm; v2 links dropped 3.4k->1-1.9k
                # steps/link from init alone). Cache is built once (atomic rename), then each rank reads 1 file.
                import glob as _glob
                import json as _json
                # LITE index (unit name -> meta path only) + LAZY pair resolution in _load_video: the eager
                # 494MB pairs index was copied into every rank AND every persistent worker (COW broken by
                # refcounts) -> hundreds of GB/node, page-cache pressure, ~40% per-step slowdown. Lazy = tiny
                # RAM + one small meta.json read per sample access.
                cache = os.path.join(pseudo_prompt_root, s, "_index_lite.json")
                pc = None
                if os.path.exists(cache):
                    try:
                        pc = [(u, mp) for u, mp in _json.load(open(cache))]
                    except Exception:
                        pc = None
                if pc is None:
                    pc = []
                    for mp in sorted(_glob.glob(os.path.join(pseudo_prompt_root, s, "*", "meta.json"))):
                        try: meta = _json.load(open(mp))
                        except Exception: continue
                        if meta.get("skip") or not meta.get("frames"): continue
                        pc.append((os.path.basename(os.path.dirname(mp)), mp))
                    try:                                        # atomic publish; racing ranks all write same content
                        tmp = cache + f".tmp{os.getpid()}"
                        _json.dump(pc, open(tmp, "w")); os.replace(tmp, cache)
                    except Exception:
                        pass
                clips = pc; self._pp_active.add(s)
                print(f"[seg4d] pseudo-prompt {s}: {len(clips)} labeled clips from {pseudo_prompt_root}", flush=True)
            self._vclips[s] = clips
            # mixing: each clip replicated round(weight) times in items (DistributedSampler then shards).
            # weight=target #samples-per-epoch / #clips, so tiny sets (kitti=21) aren't drowned by ytvos(3471).
            w = vw.get(s, 1.0); reps = max(1, int(round(w)))
            for ci in range(len(clips)):
                self.items.extend([(f"v:{s}", ci)] * reps)

    def _aspect_resize_pair(self, img, idmap, ign, tile=False, tile_off=None):
        """img PIL, idmap/ign np (H,W). Returns (rgb tensor 3,th,tw), (idmap th,tw int), (ign th,tw bool)."""
        W, H = img.size
        if tile:                                            # crop a square-ish overlapped tile, then balanced
            side = min(H, W); x0 = tile_off if tile_off is not None else 0
            img = img.crop((x0, 0, x0 + side, H)); idmap = idmap[:, x0:x0 + side]; ign = ign[:, x0:x0 + side]
            W, H = img.size
        th, tw = balanced_hw(H, W)
        rgb = torch.from_numpy(np.asarray(img.convert("RGB").resize((tw, th), Image.BILINEAR), np.float32) / 255.).permute(2, 0, 1)
        idm = np.asarray(Image.fromarray(idmap.astype(np.int32)).resize((tw, th), Image.NEAREST))
        ig = np.asarray(Image.fromarray(ign.astype(np.uint8)).resize((tw, th), Image.NEAREST)).astype(bool)
        return rgb, idm, ig

    def _load_video(self, subset, ci):
        clips = self._vclips[subset]; sq, pairs = clips[ci]
        lister, kind, exh, sparse = VIDEO_SUBSETS[subset]; dec = DECODE[kind]
        if subset in getattr(self, "_pp_active", ()):        # offline pseudo id-map PNGs replace nogt
            dec = DECODE["pseudopng"]
            if isinstance(pairs, str):                       # LAZY unit: pairs == meta.json path -> resolve now
                import json as _json
                _m = _json.load(open(pairs)); _d = os.path.dirname(pairs)
                pairs = [(f["rgb"], os.path.join(_d, f["png"])) for f in _m["frames"]]
        n = self._n_frames()
        tile = subset in TILE_SUBSETS
        # temporal window: sparse -> large strided span; else whole clip
        if sparse and len(pairs) > n:
            span = min(len(pairs), n * self.sparse_window_mult)
            start = 0 if self.frame_sampling == "consecutive" else int(np.random.randint(0, len(pairs) - span + 1))
            win = list(range(start, start + span))
        else:
            win = list(range(len(pairs)))
        if tile:
            # split each wide frame into 3 FIXED horizontal tiles; pick n views JOINTLY over the (tile x frame)
            # pool within the window -> the sample mixes spatial tiles and time together (per user 2026-06-28).
            im0 = Image.open(pairs[win[0]][0]); W0, H0 = im0.size
            offs = _tile_offsets(W0, H0, 3)
            units = [(fi, off) for fi in win for off in offs]
            if self.frame_sampling == "consecutive":                       # eval: deterministic even stride
                sel = [units[i] for i in np.linspace(0, len(units) - 1, n).round().astype(int)]
            else:
                sel = [units[i] for i in np.random.choice(len(units), size=n, replace=len(units) < n)]
            picks = [(pairs[fi], off) for fi, off in sel]
        elif sparse and len(pairs) > n:
            idx = sorted(np.linspace(win[0], win[-1], n).round().astype(int).tolist())
            picks = [(pairs[i], None) for i in idx]
        else:
            ids = self._pick(win, n); picks = [(pairs[i], None) for i in ids]
        rgbs, insts, igns, psds = [], [], [], []
        _gf = getattr(self, "_gt_fill", {}).get(subset)      # rgb->pseudo-png map (None if not a gt-fill subset)
        for (rgb_p, m_p), off in picks:
            idmap, ign = dec(m_p)
            rgb, idm, ig = self._aspect_resize_pair(Image.open(rgb_p), idmap, ign, tile=(off is not None), tile_off=off)
            rgbs.append(rgb); insts.append(idm); igns.append(ig)
            if _gf is not None:                               # decode + resize the v3 pseudo for this frame (for GT-over-pseudo)
                pp = _gf.get(rgb_p)
                if pp and os.path.exists(pp):
                    pidm, _ = DECODE["pseudopng"](pp)
                    psds.append(np.asarray(Image.fromarray(pidm.astype(np.int32)).resize((idm.shape[1], idm.shape[0]), Image.NEAREST)))
                else:
                    psds.append(np.zeros_like(idm))
        if _gf is not None and any(int(p.max()) > 0 for p in psds):   # merge: real GT covers matched pseudo, pseudo fills rest (clip-level = track-consistent)
            merged = _gt_over_pseudo(np.stack(insts, 0), np.stack(psds, 0))
            insts = [merged[i] for i in range(len(insts))]
        H, W = insts[0].shape
        sample = self._finalize(rgbs, [np.zeros((H, W), np.float32)] * len(rgbs), insts, [None], [None])
        sample["ignore"] = torch.from_numpy(np.stack(igns, 0))   # (S,H,W) bool
        sample["exhaustive"] = bool(exh)
        sample["subset"] = subset
        return sample


class Seg4dDataset(InsSceneDataset, Seg4dVideoMixin):
    """InsScene (static 3D) + video subsets, unified. video_subsets e.g. ['kitti','ytvos','davis',...]."""
    def __init__(self, *a, video_subsets=None, tile_overlap=0.2, sparse_window_mult=8, video_weights=None, video_split="all",
                 pseudo_prompt_root=None, pseudo_prompt_subsets=(), gt_pseudo_root=None, gt_pseudo_subsets=(), **kw):
        self._defer_empty = True                          # allow video-only (subsets=[]) for per-benchmark eval loaders
        kw.setdefault("aspect_resize", True)              # 3D subsets use official VGGT balanced resize like video (no square distortion)
        super().__init__(*a, **kw)
        self._seg4d_init(video_subsets or [], tile_overlap, sparse_window_mult, video_weights, video_split,
                         pseudo_prompt_root=pseudo_prompt_root, pseudo_prompt_subsets=pseudo_prompt_subsets,
                         gt_pseudo_root=gt_pseudo_root, gt_pseudo_subsets=gt_pseudo_subsets)
        if not self.items:
            raise RuntimeError(f"Seg4dDataset empty: no InsScene subsets and no video clips found ({video_subsets})")

    def __getitem__(self, i):
        subset, payload = self.items[i]
        if isinstance(subset, str) and subset.startswith("v:"):
            for _ in range(8):
                try:
                    return self._load_video(subset[2:], payload)
                except Exception as e:
                    self._last = repr(e); i = int(np.random.randint(0, len(self))); subset, payload = self.items[i]
                    if not (isinstance(subset, str) and subset.startswith("v:")): return super().__getitem__(i)
            raise RuntimeError(f"Seg4d video load failed 8x: {self._last}")
        return super().__getitem__(i)
