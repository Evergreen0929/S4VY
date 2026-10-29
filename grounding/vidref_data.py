"""Data access for video referring-segmentation benchmarks (Ref-DAVIS / LVOS / MeViS).

Provides expression lists, ground-truth masklets and frame paths. Scoring functions are
not part of this release.
"""
import os, sys, json, glob
import _env
os.environ.setdefault("MFRAME", "1"); os.environ.setdefault("BOX_OUT", "1")
os.environ.setdefault("NO_INST_PROJ", "1"); os.environ.setdefault("COT", "1")
os.environ.setdefault("MF_COT", "1"); os.environ.setdefault("MF_COT_GEN", "1")
os.environ.setdefault("FIXEDQ", "1"); os.environ.setdefault("SEG_TAG", "s4vy")
AG = os.environ.get("GROUNDING_REPO", os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, AG)
D = _env.data_root()
os.environ.setdefault("MODEL_PATH", f"{D}/../hf_models/Qwen3-VL-2B-Instruct")
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation
import grounder_infer as EA_mod
from grounder_infer import (_agent_mframe, _gen_mf_cot, _gen_mf_box, _gen_mf_box_batch, mf_tree_windows,
                        MF_NF, MF_OVERLAP, MF_COT, MF_COT_GEN, MF_TREE, MF_TREE_DEPTH)
from agent_model import AgentGrounder

DS = os.environ.get("DS", "refdavis"); SPLIT = os.environ.get("SPLIT", "valid")
RV = f"{D}/RefVOS"; VID2D = f"{D}/vid_scenes_2d"
CACHE = os.environ.get("CACHE", f"{D}/vidref_cache_{DS}_{SPLIT}")
_BAN_MAP = {}
if os.environ.get("BAN_MAP"):
    _BAN_MAP = json.load(open(os.environ["BAN_MAP"]))
BOX_DUMP = os.environ.get("BOX_DUMP", "")
BOTH_PATHS = os.environ.get("BOTH_PATHS") == "1"
_PM = [None]
if BOTH_PATHS:
    from prompt_infer import IN_W, IN_H
OUT = os.environ.get("OUT", f"{D}/vidref_eval_{DS}_{SPLIT}"); os.makedirs(OUT, exist_ok=True)
LIMIT = int(os.environ.get("LIMIT", "0"))
SHARD = int(os.environ.get("SHARD", "0")); NSHARD = int(os.environ.get("NSHARD", "1"))

CKPT = os.environ["CKPT"]
ag = AgentGrounder(os.environ["MODEL_PATH"], lora=False, loss="matching").cuda().eval()
ag.load_state_dict(torch.load(f"{CKPT}/full_model.pt", map_location="cpu"), strict=False)
proc = ag.processor


# ---------------------------------------------------------------- GT at source resolution
def gt_paths(vid):
    """Annotation PNGs in the SAME order cache_video_ref indexed them (before its MAXF sub-sample)."""
    if DS == "refdavis":
        return sorted(glob.glob(f"{RV}/refdavis/DAVIS/Annotations_unsupervised/480p/{vid}/*.png"))
    return sorted(glob.glob(f"{RV}/lvos/{SPLIT}/Annotations/{vid}/*.png"))   # LVOS: annotated frames only


def exprs(vid):
    """oid -> [expressions].

    Read from the annotation SOURCE, never from the cache's `descs`. cache_video_ref deliberately leaves that
    field empty (the GPU bridge is text-independent and must not need rebuilding when LVOS's GPT pseudo-labels
    are regenerated), so trusting it silently yields zero queries per video.
    Both sources share one line format:  <vid> <oid> "<expression>"
    """
    if DS == "mevis":
        # MeViS keeps expressions in meta_expressions.json, not the `<vid> <oid> "<expr>"` line format. Single
        # target only, the same filter cache_video_ref.py applies, so eval rows match the cached bridge.
        meta = json.load(open(f"{RV}/mevis/{SPLIT}/meta_expressions.json"))["videos"][vid]
        out = {}
        for e in meta["expressions"].values():
            if len(e.get("obj_id", [])) != 1: continue
            out.setdefault(int(e["obj_id"][0]), []).append(e["exp"])
        return out
    if DS == "refdavis":
        fs = [f"{RV}/refdavis/davis_text_annotations/Davis17_annot{i}.txt" for i in (1, 2)]
    else:
        fs = [f"{RV}/lvos/labels_gpt_{SPLIT}/{vid}.txt"]
    out = {}
    for p in fs:
        if not os.path.exists(p): continue
        for l in open(p):
            q = l.strip().split(None, 2)
            if len(q) == 3 and q[0] == vid:
                out.setdefault(int(q[1]), []).append(q[2].strip().strip('"'))
    return out


_MEVIS_MD = [None]


def load_gt(vid, frame_idx, oid):
    """MeViS ships GT as RLE in mask_dict.json keyed by anno_id, not as palette PNGs, so the PNG path below
    would silently find nothing for it. Decoded the same way cache_video_ref.py decodes it, so the eval GT and
    the cached bridge agree."""
    if DS == "mevis":
        meta = json.load(open(f"{RV}/mevis/{SPLIT}/meta_expressions.json"))["videos"][vid]
        if _MEVIS_MD[0] is None:
            _MEVIS_MD[0] = json.load(open(f"{RV}/mevis/{SPLIT}/mask_dict.json"))     # 392MB, once per worker
        md = _MEVIS_MD[0]
        aid = next((str(e["anno_id"][0]) for e in meta["expressions"].values()
                    if len(e.get("obj_id", [])) == 1 and int(e["obj_id"][0]) == int(oid)), None)
        if aid is None or aid not in md: return None
        from pycocotools import mask as mu
        rles = md[aid]
        hw = None
        for rl in rles:
            if rl: hw = mu.decode(rl).shape; break
        if hw is None: return None
        out = []
        for i in frame_idx:
            i = int(i)
            out.append(mu.decode(rles[i]).astype(bool) if i < len(rles) and rles[i] else np.zeros(hw, bool))
        return np.stack(out) if out else None
    ps = gt_paths(vid)
    out = []
    for i in frame_idx:
        if i >= len(ps): break
        out.append(np.array(Image.open(ps[int(i)])) == oid)
    return np.stack(out) if out else None


# ---------------------------------------------------------------- DAVIS J & F




def up(mask_small, hw):
    """Nearest-upsample the cache-resolution mask stack to the GT PNG size."""
    t = torch.from_numpy(mask_small).float()[:, None]
    return (F.interpolate(t, size=hw, mode="nearest")[:, 0] > 0.5).numpy()


def video_frames(vid, S):
    out = []
    for k in range(S):
        out.append(np.asarray(Image.open(f"{VID2D}/{vid}/color/{k}.jpg").convert("RGB")))
    return np.stack(out)
