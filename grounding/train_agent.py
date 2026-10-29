#!/usr/bin/env python
"""Distributed trainer for the agentic-loop grounder (4 configs via GROUP), R1/R2 MIXED.
GROUP g1|g2|g3|g4 -> LOSS(matching/dense), mask_embed trainable(g4), head-warmup(g3).
Per sample randomly pick a regime (MIX=mix default, or force r1/r2):
  R1  (frame-selection round): ALL-frame register (cap REG_MAX_FRAMES, uniform if more), text-only
       (segmenter register already encodes per-frame vision). Supervise TEMPORAL(BCE) + SPATIAL.
  R2  (grounding round): the object's GT-visible frames re-run through segmenter (r2_* cache):
       images + re-run register. Supervise SPATIAL only (NEVER temporal -- all frames are GT-visible,
       would learn a degenerate input->all-positive shortcut and corrupt the R1 selector).
Caches: register_cache(R1 all-frame reg), segmenter_train_cache(R1 qemb), field_cache(R1 uniform-8 field),
        gt_idmap_cache(visibility + dense GT), r2_match(R2 reg/qemb/target), r2_field(R2 field/gt_mask).
segmenter backbone frozen (all cached)."""
import os, sys, json, glob, random
import _env
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs, InitProcessGroupKwargs
from transformers import get_cosine_schedule_with_warmup
WORK = os.environ.get("GROUNDING_REPO", os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, WORK)
from agent_model import AgentGrounder, REG, TEMP, GROUND, THINK_S, THINK_E, INST, POINT

MODEL_PATH = _env.require("MODEL_PATH", "Qwen3-VL-4B-Instruct checkout for the grounder")
D = os.environ.get("GROUNDING_DATA_ROOT", "/PATH/TO/data")
SEG = os.environ.get("SEG_TAG", "segmenter")                # segmenter backbone tag for MODEL-DERIVED caches (segmenter default; teacher = new backbone). GT/CoT caches are model-agnostic -> stay literal.
REGc, R9c, IDM = f"{D}/{SEG}_register_cache", f"{D}/{SEG}_train_cache", f"{D}/gt_idmap_cache"
FIELDc = f"{D}/{SEG}_field_cache"                      # R1 dense: uniform-8 field
PATCHc = f"{D}/{SEG}_patch_cache"                      # Arm B: per-frame pooled patch grid (S,GH,GW,2048) for <POINT> visual pool
PATCH_VP = os.environ.get("PATCH_VP", "0") == "1"     # Arm B: point head cross-attends patch grid (else fallback zero grid = no visual pool)
PATCH_GH, PATCH_GW = int(os.environ.get("PATCH_GH", "12")), int(os.environ.get("PATCH_GW", "16"))   # patch grid (must match cache_patch.py + PointHead P)
R2M, R2F = f"{D}/{SEG}_r2_match", f"{D}/{SEG}_r2_field"  # R2 reg/qemb/target ; R2 field/gt_mask
S2D = os.environ.get("SCANNET200_2D_ROOT", "/PATH/TO/codebase/SegVGGT/data_processing/scannet200/scenes_2d")
INFIN2D = f"{D}/IGGT_Benchmark/refseg_infinigen_2d"     # synth infinigen frames (key contains "__")
SPPP_2D = f"{D}/scannetpp_scenes_2d"                    # ScanNet++ joint: frames extracted color/{k}.jpg (k matches register order); scene = 10-hex, no "scene" prefix / no "__"
SPPC = os.environ.get("SPPC", f"{D}/scannetpp_cache_v2")   # v2 = InsScene NATIVE per-frame GT + 2D-IoU targets.
VID2D = f"{D}/vid_scenes_2d"                            # joint v3: video frames normalised to color/{k}.jpg
VIDC = {}                                               # joint v3: video id -> its vidref cache file (register+idmap+qemb,
for _p in sorted(glob.glob(f"{D}/vidref_cache_*/*.npz")):   #   same single-file shape ScanNet++ uses)
    VIDC.setdefault(os.path.basename(_p)[:-4], _p)
# MUST match whatever prep_joint_manifest read: v1 and v2 have DIFFERENT instance sets (246 vs 247 for the same
# scene), so a manifest built on v2 targets but fed v1 qemb silently points at the wrong instance.
def _img_root(sc):
    if sc in VIDC: return VID2D                          # video ids collide with neither scannet nor scannetpp
    return INFIN2D if "__" in sc else (S2D if sc.startswith("scene") else SPPP_2D)

GROUP = os.environ.get("GROUP", "g1")
MIX = os.environ.get("MIX", "mix")                    # mix | r1 | r2
R2_PROB = float(os.environ.get("R2_PROB", "0.5"))
LOSS = "dense" if GROUP in ("g2", "g4") else "matching"
IS_G5 = GROUP == "g5"     # G5: history multi-turn (R1+R2 merged in ONE forward, no R1/R2 sampling). matching-only, params == G1.
ME_TRAIN = GROUP == "g4"
EPOCHS = int(os.environ.get("EPOCHS", "2")); BS = int(os.environ.get("BS", "1")); GA = int(os.environ.get("GRAD_ACC", "16"))
LR = float(os.environ.get("LR", "2e-4")); WARMUP = int(os.environ.get("WARMUP", "150"))
REG_MAX = int(os.environ.get("REG_MAX_FRAMES", "128"))  # R1 all-frame register cap (128*17=2176 tok; grad-ckpt fits)
NF_IMG = int(os.environ.get("NF_VLM", "8"))             # R1 uniform field/img frames (dense)
VIS_T = int(os.environ.get("VIS_T", "20"))
WARMUP_HEADS = int(os.environ.get("WARMUP_HEADS", "0"))  # g3
SEL_HEAD = os.environ.get("SEL_HEAD", "0") == "1"        # E4: learn #frames-to-select (count head, GT-count label) vs hard threshold
SEL_W = float(os.environ.get("SEL_W", "0.1"))            # E4 count-loss weight
COT = os.environ.get("COT", "0") == "1"                  # E5: <think> geometry+appearance CoT before <GROUND>, R2 on R1-pred frames
COT_W = float(os.environ.get("COT_W", "1.0"))            # E5 LM-loss weight on the think span
SCHED_MIX = os.environ.get("SCHED_MIX", "0") == "1"      # E1: R2 scheduled-sampling -- mix R1-pred frames in with GT frames
SCHED_P = float(os.environ.get("SCHED_P", "0.5"))        # E1: prob a R2 sample uses the R1-pred cache instead of GT
R1_SELECT_ONLY = os.environ.get("R1_SELECT_ONLY", "0") == "1"  # E7/E3-sft: R1 ONLY selects frames (no grounding loss); R2 grounds on R1-pred frames (pure SFT, no RL)
ACTION = os.environ.get("ACTION", "0") == "1"            # E2: learn the {ground, refine} action in R1 (stop policy)
ACT_TAU = float(os.environ.get("ACT_TAU", "0.5"))        # E2: R1-pred best_iou below this -> "refine" (positive class)
ACT_W = float(os.environ.get("ACT_W", "0.5"))            # E2 action-loss weight
ACT_POSW = float(os.environ.get("ACT_POSW", "5.0"))      # E2 pos_weight (refine is the ~17% minority class)
REG17 = os.environ.get("REG17", "0") == "1"              # E6: use all 17 segmenter reg tokens/frame (no pooling) + per-slot identity
FIXEDQ = os.environ.get("FIXEDQ", "0") == "1"            # teacher fixed-Q: single forward -> fixed all-frame register + Q shared by R1/R2.
# R1 (uniform-8 img) selects frames; R2 (top-8 GT-visible img) grounds by matching the SAME all-frame qemb (train_cache) with manifest target.
# No R2 re-forward -> candidate set never drifts -> removes the R1->R2 distribution mismatch. Use with R1_SELECT_ONLY=1.
LOC = os.environ.get("LOC", "")                          # SAM-hint baseline: "point"|"bbox". R2 = pure Qwen3-VL on ONE key frame,
#   generate the coordinate as NATURAL-LANGUAGE TEXT (LM loss). NO register/<GROUND>/segmenter in R2. R1 = temporal presence (unchanged), top-1 at eval.
MW, MH = 128, 96                                         # gt_idmap_cache 2D resolution (W,H); coords normalized to [0,1] by these
INST_APPEND = os.environ.get("INST_APPEND", "0") == "1"  # ExpA: R2 appends the fixed-Q instance queries as <INST> tokens (VLM contextualizes) -> <GROUND> matches their output hiddens (vs blind external cosine)
SFRAME = os.environ.get("SFRAME", "0") == "1"            # single-frame grounder: ONE frame -> vision tower + that frame's teacher register; <GROUND> matches a qemb query OR null (object absent). Eval votes over frames. No R1/temporal.
SF_NOREG = os.environ.get("SF_NOREG", "0") == "1"        # ablation: SFRAME WITHOUT the 17 teacher register tokens -> ONLY the image via VLM vision tower (+desc) -> <GROUND> matches raw qemb. Tests if teacher register helps at all.
SF_PATCH = os.environ.get("SF_PATCH", "0") == "1"        # SFRAME + teacher dense PATCH tokens (192/frame, 2x2-pooled) injected via the register stream, for finer instance discrimination
PATCHc = f"{D}/{SEG}_patch_cache"
MFRAME = os.environ.get("MFRAME", "0") == "1"            # multi-frame: MF_NF frames (sampled from a MF_WIN-frame context window) + MF_NF*17 reg -> MF_NF <GROUND> tokens, each matches ITS frame's query or null. Batched SF w/ temporal context.
POINT_HEAD = os.environ.get("POINT_HEAD", "0") == "1"    # dual-output: per-frame <POINT> token regresses 2D coord (indep of <GROUND> qemb match); L1 to GT max-inscribed point, only where target visible
BOX_OUT = os.environ.get("BOX_OUT", "0") == "1"          # Arm C: after the per-frame <GROUND> blocks, GENERATE a per-frame text bbox list (leverage Qwen3-VL grounding pretrain), 'absent' if not present; LM loss
PHASE2 = os.environ.get("PHASE2", "0") == "1"            # Phase-2 2-pass self-correction: mix phase2_sft.jsonl (Turn-1 masked + [REVIEW] + Turn-2 LM loss) with Arm C GT replay; warm-start via INIT_CKPT
REPLAY_MULT = float(os.environ.get("REPLAY_MULT", "1.0"))  # replay subset size = REPLAY_MULT * len(phase2) (1.0 = 1:1 correction:replay)
P2_SFT = os.environ.get("P2_SFT", "")                    # phase2_sft.jsonl path
P2_GROUND_W = float(os.environ.get("P2_GROUND_W", "1.0"))   # weight of the Turn-2 <GROUND> matching loss
POINT_W = float(os.environ.get("POINT_W", "1.0"))        # weight of the point L1 loss
POINT_NULL_W = float(os.environ.get("POINT_NULL_W", "1.0"))   # Arm B: weight of the point-head null (absent) BCE, on ALL frames
MF_NF = int(os.environ.get("MF_NF", "8"))                # frames per forward
MF_WIN = int(os.environ.get("MF_WIN", "32"))             # context window to sample the MF_NF frames from
MF_IMG = int(os.environ.get("MF_IMG", "0"))              # MF32 variant: #images for the vision tower (uniform subset of the MF_NF frames); 0 = all MF_NF (original MF)
MF_PROMPT = 'Look at these {n} video frames. For each frame, if the object described below appears in it, ground that object; if it does not appear in that frame, report absent.\nDescription: "{d}"'
SF_PROMPT = 'Look at this single image. If the object described below appears in it, ground that object; if it does not appear, report absent.\nDescription: "{d}"'
R1PRED = f"{D}/{SEG}_r2_match_r1pred"; COTc = f"{D}/cot_cache"   # E5: R2 frames+target (R1-pred) + teacher CoT text (CoT is teacher text = model-agnostic, dir name is legacy/literal; the geometry behind it is s4vy now)
SPPCOT = f"{D}/scannetpp_cot_cache"                    # ScanNet++ Turn-1 CoT, keyed {scene}__{oid} (NOT manifest row index)
HEAD_KEYS = ["reg_proj", "reg_ln", "temporal_head", "text_proj", "inst_proj", "gq_proj", "mask_embed"] \
    + (["count_head"] if SEL_HEAD else []) + (["action_head"] if ACTION else []) + (["reg_slot_emb"] if REG17 else []) \
    + (["inst_in_proj", "inst_in_ln", "inst_out_proj"] if INST_APPEND else []) \
    + (["null_head"] if (SFRAME or MFRAME) else []) \
    + (["point_head"] if POINT_HEAD else [])
OUT_DIR = _env.require("OUT_DIR", "checkpoint output directory")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))
PROMPT_R1 = "Round 1: using the scene tokens from all {nf} frames, find which frames contain the described object.\nDescription: \"{d}\""
PROMPT_R2 = "Round 2: locate the described object in these {nf} frames.\nDescription: \"{d}\""
# LOC (SAM-hint) R2: single key frame -> generate the coordinate as text. Prompt/answer must match grounder_infer.py.
PROMPT_LOC = {
    "point": 'Locate the object described below in this image. Give its center as a normalized point (x, y), each coordinate in [0,1].\nDescription: "{d}"',
    "bbox":  'Locate the object described below in this image. Give its bounding box as normalized [x1, y1, x2, y2], each coordinate in [0,1].\nDescription: "{d}"',
}
def _loc_answer(loc, c):   # c = (cx,cy) for point | (x1,y1,x2,y2) for bbox, normalized [0,1]
    if loc == "point": return f"The target object is located at point ({c[0]:.3f}, {c[1]:.3f})."
    return f"The target object is located within the box [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}, {c[3]:.3f}]."
def _interior_point(mask):  # deepest interior pixel (distance-transform peak) -> ALWAYS inside mask (centroid of a non-convex mask often isn't)
    try:
        from scipy.ndimage import distance_transform_edt
        r, c = np.unravel_index(int(distance_transform_edt(mask).argmax()), mask.shape)
        return int(c), int(r)
    except Exception:
        ys, xs = np.where(mask); j = int(((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2).argmin())
        return int(xs[j]), int(ys[j])
def _mask_coords(mask):    # mask (H,W) bool -> point (interior pt, normalized) and (x1,y1,x2,y2) tight box, normalized [0,1] by (MW,MH)
    px, py = _interior_point(mask); ys, xs = np.where(mask)
    return (px / MW, py / MH), (float(xs.min()) / MW, float(ys.min()) / MH, float(xs.max() + 1) / MW, float(ys.max() + 1) / MH)


def _rng(i):  # deterministic-ish per-sample regime without Math.random restrictions
    return random.Random(i).random()


class UniformRegimeBatchSampler:
    """BS>1: yield batches whose samples share ONE regime (r1 or r2), so collate/loss (which read
    batch[0]'s regime) stay valid. Regime is the SAME deterministic rule as AgentData.__getitem__.
    accelerate sharding (BatchSamplerShard) gives each rank every-Nth batch -> regime stays uniform."""
    def __init__(self, ds, batch_size, drop_last=True):
        self.bs, self.drop_last, self._epoch = batch_size, drop_last, 0
        self.r1, self.r2 = [], []
        for i, it in enumerate(ds.items):
            key = f"{it['scene']}__{it['oid']}"
            ok = (it["target"] >= 0) if FIXEDQ else (               # fixed-Q: R2 only if valid target (must match __getitem__)
                 ds._loc_ok(it["scene"], it["oid"]) if LOC else (   # LOC uses idmap-visibility
                 ds._cot_ok(f"{key}__{it['_qid']}") if COT else ds._r2_ok(key)))   # E5 uses _cot_ok
            (self.r2 if (_rng(i) < R2_PROB and ok) else self.r1).append(i)

    def _chunks(self, lst):
        return [lst[i:i + self.bs] for i in range(0, len(lst), self.bs)
                if (not self.drop_last) or len(lst[i:i + self.bs]) == self.bs]

    def __iter__(self):
        rng = random.Random(1234 + self._epoch); self._epoch += 1   # same across ranks -> consistent shard
        r1, r2 = self.r1[:], self.r2[:]; rng.shuffle(r1); rng.shuffle(r2)
        batches = self._chunks(r1) + self._chunks(r2); rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return len(self.r1) // self.bs + len(self.r2) // self.bs


def _max_inscribed(mask):
    """dual-output GT point = pole-of-inaccessibility (argmax distance-transform) of the 2D mask, normalized [0,1]. Guaranteed INSIDE the mask (unlike centroid on concave shapes)."""
    import cv2
    if mask.sum() == 0: return np.zeros(2, np.float32)
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    y, x = np.unravel_index(int(dt.argmax()), dt.shape)
    return np.array([x / mask.shape[1], y / mask.shape[0]], np.float32)   # (x_norm, y_norm)


class AgentData(Dataset):
    def __init__(self, manifest):
        items = [json.loads(l) for l in open(manifest)]
        for qi, it in enumerate(items): it["_qid"] = qi      # original manifest index = R1-pred/cot cache key suffix (E1/E2/E5)
        def has(sc, oid):
            if sc in VIDC: return True                           # joint v3: video single-file cache
            if os.path.exists(f"{SPPC}/{sc}.npz"): return True   # ScanNet++ joint: single-file cache (register+idmap+qemb)
            base = all(os.path.exists(f"{c}/{sc}.npz") for c in (REGc, IDM, R9c))
            if LOSS == "dense": base = base and os.path.exists(f"{FIELDc}/{sc}.npz")
            return base
        self.items = [it for it in items if has(it["scene"], it["oid"])]
        if IS_G5:   # G5 merges R1+R2 every step -> require the R2 cache too
            self.items = [it for it in self.items if self._r2_ok(f"{it['scene']}__{it['oid']}")]
        self._cache = {}
        self._cot_set = set()
        if COT and not MFRAME:  # E5: freeze cot-ready set ONCE (cotgen may still be writing -> sampler & getitem must agree this job). MFRAME loads CoT directly in __getitem__ -> skip.
            self._cot_set = {f"{it['scene']}__{it['oid']}__{it['_qid']}" for it in self.items
                             if self._cot_ok_raw(f"{it['scene']}__{it['oid']}__{it['_qid']}")}
    def __len__(self): return len(self.items)

    def _r1(self, sc):
        if ("r1", sc) not in self._cache:
            spp = VIDC.get(sc) or f"{SPPC}/{sc}.npz"
            if os.path.exists(spp):                                       # ScanNet++ joint: register+idmap+qemb in ONE file (scene names don't collide w/ scannet200)
                z = np.load(spp, allow_pickle=True)
                self._cache[("r1", sc)] = (z["register"].astype(np.float32), z["idmap"], z)
            else:
                self._cache[("r1", sc)] = (np.load(f"{REGc}/{sc}.npz")["register"],
                                           np.load(f"{IDM}/{sc}.npz")["idmap"],
                                           np.load(f"{R9c}/{sc}.npz", allow_pickle=True))
        return self._cache[("r1", sc)]

    def _flatreg(self, reg3d_sel):
        """(m,17,2048) frame-selected segmenter register -> (reg (ntok,2048) f32, regfr (ntok,) frame idx).
        Default pools 17->1/frame (G1); REG17 (E6) keeps all 17, frame-major layout."""
        if REG17:
            return (reg3d_sel.reshape(-1, 2048).astype(np.float32),
                    np.repeat(np.arange(len(reg3d_sel)), 17).astype(np.int64))
        return reg3d_sel.mean(1).astype(np.float32), np.arange(len(reg3d_sel)).astype(np.int64)

    def _r1pred_ok(self, ckey):                               # E1: R1-pred R2 cache present + non-empty + readable
        p = f"{R1PRED}/{ckey}.npz"
        if not os.path.exists(p): return False
        try: return "empty" not in np.load(p).files           # .files reads zip dir -> EOFError on partial-write npz
        except Exception: return False                        # corrupt/truncated -> treat as absent

    def _cot_ok_raw(self, ckey):                              # E5: R1-pred + teacher CoT both present; require GT cache too (degrade fallback)
        key = ckey.rsplit("__", 1)[0]
        return self._r1pred_ok(ckey) and os.path.exists(f"{COTc}/{ckey}.json") and self._r2_ok(key)

    def _cot_ok(self, ckey):                                  # frozen snapshot (cotgen may run concurrently -> sampler/getitem must agree within a job)
        return ckey in self._cot_set

    def _r2_ok(self, key):
        m = f"{R2M}/{key}.npz"
        if not os.path.exists(m): return False
        try:
            if "empty" in np.load(m).files: return False
            if LOSS == "dense":
                fpath = f"{R2F}/{key}.npz"
                if not os.path.exists(fpath) or "empty" in np.load(fpath).files: return False
        except Exception: return False                        # corrupt/partial -> treat as absent
        return True

    def _loc_ok(self, sc, oid):                               # LOC R2: object visible (>VIS_T px) in >=1 frame of the idmap
        try:
            _, idm, _ = self._r1(sc)
            return bool(((idm == oid).reshape(idm.shape[0], -1).sum(1) > VIS_T).any())
        except Exception: return False

    def _loc_best_frame(self, sc, oid):                       # LOC R2: pick the frame with the LARGEST target mask (clearest hint), + coords
        _, idm, _ = self._r1(sc)
        area = (idm == oid).reshape(idm.shape[0], -1).sum(1)
        f = int(area.argmax())
        pt, bx = _mask_coords(idm[f] == oid)
        return f, (pt if LOC == "point" else bx)

    def __getitem__(self, i):
        it = self.items[i]; sc, oid = it["scene"], it["oid"]; key = f"{sc}__{oid}"
        ckey = f"{key}__{it['_qid']}"                          # E5/E1/E2 cache key (original manifest index)
        _root = _img_root(sc)
        def _imgs1(idxs):                                      # (defined early for SFRAME; identical to _imgs below)
            out = []
            for f in idxs:
                im = Image.open(f"{_root}/{sc}/color/{int(f)}.jpg").convert("RGB")
                if im.width > 512: im = im.resize((512, int(im.height*512/im.width)))
                out.append(im)
            return out
        if SFRAME:                                            # single-frame grounder: ONE frame + null. ~1:1 positive(target visible)/null(target absent).
            reg_all, idm, r9 = self._r1(sc); S = reg_all.shape[0]
            area = (idm == oid).reshape(S, -1).sum(1)          # target GT-mask area per frame
            vis_frames = np.where(area > VIS_T)[0]; neg_frames = np.where(area <= VIS_T)[0]
            pos = (_rng(i) < 0.5) and it["target"] >= 0 and len(vis_frames) > 0   # ~1:1 pos/null coin flip
            if pos:
                f = int(vis_frames[random.Random(i).randrange(len(vis_frames))]); tgt = int(it["target"]); isnull = False
            else:
                pool = neg_frames if len(neg_frames) > 0 else np.arange(S)        # degenerate: object in every frame -> still pick some (rare)
                f = int(pool[random.Random(i + 7919).randrange(len(pool))]); tgt = -1; isnull = True
            return {"regime": "sf", "desc": it["desc"], "reg": reg_all[f].astype(np.float32), "nreg_f": 1,
                    "images": _imgs1([f]), "nimg": 1, "qemb": r9["qemb"].astype(np.float32),
                    "target": tgt, "is_null": isnull}
        if MFRAME:                                            # MF_NF frames from a MF_WIN context window -> per-frame query/null
            reg_all, idm, r9 = self._r1(sc); S = reg_all.shape[0]
            area = (idm == oid).reshape(S, -1).sum(1); vis = np.where(area > VIS_T)[0]
            rng = random.Random(i)
            use_pos = len(vis) > 0 and rng.random() < 0.8      # 0.8: balanced window containing the target; 0.2: random window (keeps some all-null for null calibration)
            c = int(vis[rng.randrange(len(vis))]) if use_pos else rng.randrange(S)
            lo = 0 if S <= MF_WIN else min(max(0, c - MF_WIN // 2), S - MF_WIN)
            win = list(range(lo, min(lo + MF_WIN, S)))
            if use_pos:                                          # force ~half positives (mirror SF 1:1) so windows aren't null-heavy
                wv = [f for f in win if area[f] > VIS_T]; wn = [f for f in win if area[f] <= VIS_T]
                kp = min(len(wv), MF_NF // 2)
                sel = rng.sample(wv, kp) + (rng.sample(wn, min(MF_NF - kp, len(wn))) if wn else [])
                if len(sel) < MF_NF:                             # fill remainder from the rest of the window
                    pool = [f for f in win if f not in sel]; sel += rng.sample(pool, min(MF_NF - len(sel), len(pool)))
                sel = sorted(sel)
            else:
                sel = sorted(rng.sample(win, MF_NF)) if len(win) >= MF_NF else sorted(win)
            if len(sel) < MF_NF: sel = sorted((sel * ((MF_NF // max(1, len(sel))) + 1))[:MF_NF])  # tiny scene: repeat to fill MF_NF
            nq = r9["qemb"].shape[0]
            targets = [(int(it["target"]) if (it["target"] >= 0 and area[f] > VIS_T) else nq) for f in sel]  # per-frame: query idx or null(=nq)
            cot = ""
            if COT:                                              # MF+CoT: reuse the (backbone-agnostic) teacher CoT text, keyed scene__oid__qid
                try: cot = json.load(open(f"{COTc}/{sc}__{oid}__{it['_qid']}.json"))["cot"]
                except Exception:
                    try: cot = json.load(open(f"{SPPCOT}/{sc}__{oid}.json"))["cot"]   # ScanNet++ (gen_spp_cot.py): keyed (scene,oid), immune to manifest row-index shifts
                    except Exception: cot = ""
            pts = np.zeros((len(sel), 2), np.float32); pvalid = np.zeros(len(sel), bool)   # dual-output: per-frame GT 2D point (normalized) + visible mask
            if POINT_HEAD:
                for k, f in enumerate(sel):
                    if area[f] > VIS_T: pts[k] = _max_inscribed(idm[f] == oid); pvalid[k] = True
            _desc = it["desc"]
            if it.get("descs") and len(it["descs"]) > 1:
                _desc = random.choice(it["descs"])   # ScanNet++ carries ~5 SpaceFormer paraphrases of the SAME object.
                # Sampling one PER SAMPLE keeps the row count (and the ScanNet++ share of the joint) unchanged while
                # varying the wording. Deliberately NOT random.Random(i): that is seeded by the fixed sample index,
                # so every epoch would draw the identical caption and the augmentation would do nothing. The plain
                # `random` module is re-seeded per worker per epoch by the DataLoader.
            out = {"regime": "mf", "desc": _desc, "reg": reg_all[sel].astype(np.float32), "nframe": len(sel),
                    "images": _imgs1([sel[i] for i in (np.linspace(0, len(sel)-1, MF_IMG).round().astype(int) if (MF_IMG and MF_IMG < len(sel)) else range(len(sel)))]),
                    "qemb": r9["qemb"].astype(np.float32), "targets": targets, "cot": cot, "points": pts, "point_valid": pvalid}
            if BOX_OUT:                                                    # Arm C: per-frame GT bbox text (<box>x1,y1,x2,y2</box> in 0-1000, or 'absent'); teacher-forced LM target
                parts = []
                for k, f in enumerate(sel):
                    if area[f] > VIS_T:
                        _, bx = _mask_coords(idm[f] == oid)                # (x1,y1,x2,y2) normalized [0,1] by (MW,MH)
                        parts.append(f"F{k+1}: <box>{int(bx[0]*1000)},{int(bx[1]*1000)},{int(bx[2]*1000)},{int(bx[3]*1000)}</box>")
                    else:
                        parts.append(f"F{k+1}: absent")
                out["boxstr"] = " ".join(parts)
            if POINT_HEAD and PATCH_VP:                                    # Arm B: load patch grid for sel frames (1-scene worker cache to amortize npz decompress)
                if getattr(self, "_pc_sc", None) != sc:
                    try: self._pc_arr = np.load(f"{PATCHc}/{sc}.npz")["patch"]   # (S,GH,GW,2048) fp16
                    except Exception: self._pc_arr = None
                    self._pc_sc = sc
                out["patch"] = (self._pc_arr[sel].astype(np.float16) if self._pc_arr is not None
                                else np.zeros((len(sel), PATCH_GH, PATCH_GW, 2048), np.float16))   # missing scene -> zero grid (head sees no visual pool)
            return out
        # choose regime
        if MIX == "r1": regime = "r1"
        elif FIXEDQ: regime = "r2" if (_rng(i) < R2_PROB and it["target"] >= 0) else "r1"  # fixed-Q: R2 only if teacher has a valid instance (target>=0); target=-1 -> R1-only (frame sel still trainable)
        elif LOC: regime = "r2" if (_rng(i) < R2_PROB and self._loc_ok(sc, oid)) else "r1"   # LOC R2 needs only idmap (object visible in >=1 frame)
        elif COT: regime = "r2" if (_rng(i) < R2_PROB and self._cot_ok(ckey)) else "r1"
        elif MIX == "r2": regime = "r2" if self._r2_ok(key) else "r1"
        else: regime = "r2" if (_rng(i) < R2_PROB and self._r2_ok(key)) else "r1"

        _root = _img_root(sc)
        def _imgs(idxs):                                                  # load + resize frames as VLM vision input
            out = []
            for f in idxs:
                im = Image.open(f"{_root}/{sc}/color/{int(f)}.jpg").convert("RGB")
                if im.width > 512: im = im.resize((512, int(im.height*512/im.width)))
                out.append(im)
            return out
        if IS_G5:                                                         # G5: ONE sample = R1 segment + R2 segment (merged history)
            reg_all, idm, r9 = self._r1(sc); S = reg_all.shape[0]
            fidx = np.arange(S) if S <= REG_MAX else np.linspace(0, S-1, REG_MAX).round().astype(int)
            vis = (idm[fidx] == oid).reshape(len(fidx), -1).sum(1)
            iimg = np.linspace(0, S-1, min(S, NF_IMG)).round().astype(int)  # R1 uniform-8 images (same as G1)
            zm = np.load(f"{R2M}/{key}.npz"); gtfr = zm["gtfr"]
            return {"regime": "g5", "desc": it["desc"],
                    "reg_r1": reg_all[fidx].mean(1).astype(np.float32), "nreg_r1": len(fidx),
                    "vis": (vis > VIS_T).astype(np.float32),
                    "images_r1": _imgs(iimg), "nimg_r1": len(iimg),
                    "qemb_r1": r9["qemb"].astype(np.float32), "target_r1": int(it["target"]),
                    "reg_r2": zm["register"].mean(1).astype(np.float32), "nreg_r2": len(gtfr),
                    "images_r2": _imgs(gtfr), "nimg_r2": len(gtfr),
                    "qemb_r2": zm["qemb"].astype(np.float32), "target_r2": int(zm["target"])}
        if regime == "r1":
            reg_all, idm, r9 = self._r1(sc); S = reg_all.shape[0]
            fidx = np.arange(S) if S <= REG_MAX else np.linspace(0, S-1, REG_MAX).round().astype(int)
            reg_sel, reg_fr = self._flatreg(reg_all[fidx])                # pooled 1/frame (G1) or all-17/frame (E6)
            vis = (idm[fidx] == oid).reshape(len(fidx), -1).sum(1)
            vis_lab = (vis > VIS_T).astype(np.float32)                    # temporal BCE target
            iimg = np.linspace(0, S-1, min(S, NF_IMG)).round().astype(int)  # R1 visual context = uniform-8 images
            out = {"regime": "r1", "desc": it["desc"], "reg": reg_sel, "regfr": reg_fr, "nreg_f": len(fidx),
                   "vis": vis_lab, "images": _imgs(iimg), "nimg": len(iimg)}
            if ACTION:                                                    # E2: stop label = will grounding on the R1-pred selection fail?
                out["act_label"] = -1.0                                   # -1 = no/corrupt r1-pred entry -> ignored in loss
                if self._r1pred_ok(ckey):
                    try: out["act_label"] = float(float(np.load(f"{R1PRED}/{ckey}.npz")["best_iou"]) < ACT_TAU)
                    except Exception: out["act_label"] = -1.0             # partial-write npz -> skip label, don't crash worker
            if LOC:                                                       # LOC: R1 is temporal-only (top-1 frame selector), no grounding target
                return out
            if LOSS == "matching":
                out["qemb"] = r9["qemb"].astype(np.float32); out["target"] = int(it["target"])
            else:                                                         # R1 dense: uniform-8 field + gt_mask
                fz = np.load(f"{FIELDc}/{sc}.npz"); fsel = fz["sel"]
                out["field"] = fz["field"].astype(np.float32)             # (n8,256,96,128)
                out["gt_mask"] = (idm[fsel] == oid).astype(np.float32)    # (n8,96,128)
            return out
        elif FIXEDQ:                                                      # R2 fixed-Q: SAME all-frame register as R1 + top-8 GT-visible images -> match fixed all-frame qemb (no re-forward)
            reg_all, idm, r9 = self._r1(sc); S = reg_all.shape[0]
            fidx = np.arange(S) if S <= REG_MAX else np.linspace(0, S-1, REG_MAX).round().astype(int)
            reg_sel, reg_fr = self._flatreg(reg_all[fidx])
            vis = (idm[fidx] == oid).reshape(len(fidx), -1).sum(1)
            top = np.sort(fidx[np.argsort(-vis)[:NF_IMG]])                # top-NF_IMG GT-visible frames (by mask area) = R2 visual context (default)
            if SCHED_MIX and _rng(2*i + 1) < SCHED_P:                     # ExpB: SCHED_P of the time use R1-PREDICTED top-8 frames (exposure-bias fix)
                try:
                    j = json.load(open(f"{D}/r1pred_frames/{sc}.json")).get(f"{oid}__{it['_qid']}")
                    if j: top = np.array(sorted(int(x) for x in j))
                except Exception: pass
            return {"regime": "r2", "desc": it["desc"], "reg": reg_sel, "regfr": reg_fr, "nreg_f": len(fidx),
                    "images": _imgs(top), "nimg": len(top), "vis": None,
                    "qemb": r9["qemb"].astype(np.float32), "target": int(it["target"])}
        elif LOC:                                                         # LOC R2: ONE key frame -> Qwen3-VL vision tower -> generate coord as text (LM loss)
            f, coord = self._loc_best_frame(sc, oid)
            return {"regime": "r2", "desc": it["desc"], "images": _imgs([f]), "nimg": 1, "coord": np.asarray(coord, np.float32)}
        elif COT:                                                         # E5 R2: R1-pred frames + teacher CoT + LM loss
            try:
                zm = np.load(f"{R1PRED}/{ckey}.npz"); gtfr = zm["gtfr"]; reg_flat, reg_fr = self._flatreg(zm["register"])
                qemb = zm["qemb"].astype(np.float32); tgt = int(zm["target"])
                cot = json.load(open(f"{COTc}/{ckey}.json"))["cot"]
            except Exception:                                             # corrupt R1-pred/cot -> GT cache, empty CoT (stays r2, no crash)
                zm = np.load(f"{R2M}/{key}.npz"); gtfr = zm["gtfr"]; reg_flat, reg_fr = self._flatreg(zm["register"])
                qemb = zm["qemb"].astype(np.float32); tgt = int(zm["target"]); cot = ""
            out = {"regime": "r2", "desc": it["desc"], "reg": reg_flat, "regfr": reg_fr,
                   "nreg_f": len(gtfr), "vis": None, "images": _imgs(gtfr), "nimg": len(gtfr),
                   "cot": cot, "qemb": qemb, "target": tgt}
            return out
        else:                                                             # R2: re-run segmenter on selected frames
            src, skey = R2M, key                                          # default: GT-visible frames
            if SCHED_MIX and _rng(2*i + 1) < SCHED_P and self._r1pred_ok(ckey):
                src, skey = R1PRED, ckey                                   # E1: sometimes draw the R1-PREDICTED frames instead
            try:
                zm = np.load(f"{src}/{skey}.npz"); gtfr = zm["gtfr"]; reg_flat, reg_fr = self._flatreg(zm["register"])
            except Exception:                                             # corrupt/partial R1-pred npz -> fall back to GT cache (same regime)
                zm = np.load(f"{R2M}/{key}.npz"); gtfr = zm["gtfr"]; reg_flat, reg_fr = self._flatreg(zm["register"])
            out = {"regime": "r2", "desc": it["desc"], "reg": reg_flat, "regfr": reg_fr,
                   "nreg_f": len(gtfr), "vis": None, "images": _imgs(gtfr), "nimg": len(gtfr)}  # ALL selected frames as images
            if LOSS == "matching":                                        # R2: match the SELECTED-frame re-run instances (design)
                out["qemb"] = zm["qemb"].astype(np.float32); out["target"] = int(zm["target"])
            else:
                zf = np.load(f"{R2F}/{key}.npz")
                out["field"] = zf["field"].astype(np.float32)             # (len,256,96,128)
                gm = np.unpackbits(zf["gt_mask"], axis=1)[:, :96*128].reshape(len(gtfr), 96, 128)
                out["gt_mask"] = gm.astype(np.float32)
            return out


def make_collate(proc):
    THINK_S_ID, THINK_E_ID = (proc.tokenizer.convert_tokens_to_ids(t) for t in (THINK_S, THINK_E))
    def collate(batch):                                                   # BS=1 (one regime per step)
        if SFRAME:                                                        # single-frame grounder: image + that frame's 17 teacher reg tokens -> <GROUND> matches qemb query OR null
            texts, images = [], []
            for b in batch:
                content = [{"type": "image"}, {"type": "text", "text": SF_PROMPT.format(d=b["desc"])}]
                t = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
                t = t + ("" if SF_NOREG else REG * 17) + f' Target: "{b["desc"]}". ' + GROUND   # SF_NOREG: image-only (no teacher reg); else 17 teacher reg tokens of this frame
                texts.append(t); images.append(b["images"])
            enc = proc(text=texts, images=images, return_tensors="pt", padding=True)
            if SF_NOREG:                                                  # no register tokens -> dummy 1-token (regfr=-1 => not scattered), mirrors LOC branch
                reg_pad = torch.zeros(len(batch), 1, 2048); regfr_pad = torch.full((len(batch), 1), -1, dtype=torch.long)
            else:
                reg_pad = torch.zeros(len(batch), 17, 2048); regfr_pad = torch.zeros(len(batch), 17, dtype=torch.long)  # all tokens = frame 0
                for j, b in enumerate(batch): reg_pad[j] = torch.from_numpy(b["reg"]).reshape(17, 2048)
            M = max(b["qemb"].shape[0] for b in batch); P = batch[0]["qemb"].shape[1]
            ie = torch.zeros(len(batch), M, P); im = torch.zeros(len(batch), M, dtype=torch.bool); tgt = torch.zeros(len(batch), dtype=torch.long)
            for j, b in enumerate(batch):
                n = b["qemb"].shape[0]; ie[j, :n] = torch.from_numpy(b["qemb"]); im[j, :n] = True
                tgt[j] = M if b["is_null"] else b["target"]               # null label = column M (appended after the padded queries)
            return dict(enc=dict(enc), register=reg_pad, reg_frame=regfr_pad, regime=["sf"] * len(batch),
                        nreg_f=[1] * len(batch), inst_emb=ie, inst_mask=im, target=tgt)
        if MFRAME:                                                        # multi-frame: NF images + NF*17 reg + NF <GROUND> (one per frame)
            NF = batch[0]["nframe"]
            texts, images, box_pfx = [], [], []
            is_cot_mf = COT
            if BOX_OUT: proc.tokenizer.padding_side = "right"                 # Arm C: box answer at [plen:] -> right-pad so plen indexing is valid
            for b in batch:
                content = [{"type": "image"} for _ in range(len(b["images"]))] + [{"type": "text", "text": MF_PROMPT.format(n=NF, d=b["desc"])}]  # MF32: tower may see a uniform SUBSET of the NF frames
                t = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
                if is_cot_mf and b.get("cot"): t += THINK_S + b["cot"] + THINK_E   # MF+CoT: reason once about the object BEFORE the per-frame grounds
                for j in range(NF): t += f' Frame {j+1}: ' + ("" if SF_NOREG else REG*17) + GROUND + (POINT if POINT_HEAD else "")   # per-frame block: 17 reg + <GROUND> (+ <POINT> for dual-output)
                if BOX_OUT and "boxstr" in b:                                # Arm C: append per-frame bbox list AFTER the ground blocks (generated at eval)
                    pfx = t + " Boxes:"; t = pfx + " " + b["boxstr"]; box_pfx.append(pfx)
                else: box_pfx.append(None)
                texts.append(t); images.append(b["images"])
            enc = proc(text=texts, images=images, return_tensors="pt", padding=True)
            if is_cot_mf or BOX_OUT:                                          # LM labels: supervise the <think> CoT span and/or the Arm C box-list span
                ids = enc["input_ids"]; lm_mf = torch.full_like(ids, -100)
                for j in range(ids.shape[0]):
                    if is_cot_mf:
                        ps = (ids[j] == THINK_S_ID).nonzero(); pe = (ids[j] == THINK_E_ID).nonzero()
                        if len(ps) and len(pe): lm_mf[j, int(ps[0]) + 1: int(pe[0]) + 1] = ids[j, int(ps[0]) + 1: int(pe[0]) + 1]
                    if BOX_OUT and box_pfx[j] is not None:                    # supervise the box answer (tokens after "... Boxes:"); plen via re-tokenizing prefix WITH images
                        plen = int(proc(text=[box_pfx[j]], images=[images[j]], return_tensors="pt")["attention_mask"][0].sum())
                        lm_mf[j, plen:] = ids[j, plen:]; lm_mf[j, enc["attention_mask"][j] == 0] = -100
            if SF_NOREG:
                reg_pad = torch.zeros(len(batch), 1, 2048); regfr_pad = torch.full((len(batch), 1), -1, dtype=torch.long)
            else:
                reg_pad = torch.zeros(len(batch), NF*17, 2048); regfr_pad = torch.zeros(len(batch), NF*17, dtype=torch.long)
                for j, b in enumerate(batch):
                    reg_pad[j] = torch.from_numpy(b["reg"]).reshape(NF*17, 2048); regfr_pad[j] = torch.arange(NF).repeat_interleave(17)
            M = max(b["qemb"].shape[0] for b in batch); P = batch[0]["qemb"].shape[1]
            ie = torch.zeros(len(batch), M, P); im = torch.zeros(len(batch), M, dtype=torch.bool); tgt = torch.zeros(len(batch), NF, dtype=torch.long)
            for j, b in enumerate(batch):
                n = b["qemb"].shape[0]; ie[j, :n] = torch.from_numpy(b["qemb"]); im[j, :n] = True
                tgt[j] = torch.tensor([(M if t >= n else t) for t in b["targets"]])   # per-frame label; null(=n) -> column M
            d_mf = dict(enc=dict(enc), register=reg_pad, reg_frame=regfr_pad, regime=["mf"] * len(batch),
                        nreg_f=[NF] * len(batch), inst_emb=ie, inst_mask=im, mf_targets=tgt)
            if is_cot_mf or BOX_OUT: d_mf["lm_labels"] = lm_mf
            if POINT_HEAD:                                                    # dual-output: per-frame GT point + visible mask
                pts = torch.zeros(len(batch), NF, 2); pval = torch.zeros(len(batch), NF, dtype=torch.bool)
                for j, b in enumerate(batch): pts[j] = torch.from_numpy(b["points"]); pval[j] = torch.from_numpy(b["point_valid"])
                d_mf["mf_points"] = pts; d_mf["mf_point_valid"] = pval
                if PATCH_VP and "patch" in batch[0]:                          # Arm B: patch grid (B,NF,GH*GW,2048) for point-head visual pool
                    pg = torch.zeros(len(batch), NF, PATCH_GH * PATCH_GW, 2048, dtype=torch.float16)
                    for j, b in enumerate(batch): pg[j] = torch.from_numpy(b["patch"]).reshape(NF, PATCH_GH * PATCH_GW, 2048)
                    d_mf["mf_patch"] = pg
            return d_mf
        if LOC and batch[0]["regime"] == "r2":                            # LOC R2: pure Qwen3-VL SFT -- ONE frame image + prompt -> generate coord text
            proc.tokenizer.padding_side = "right"                         # BS>1: prefix stays at [0:plen] so label masking below is correct (default may be left)
            texts, images, prefixes = [], [], []
            for b in batch:
                pr = PROMPT_LOC[LOC].format(d=b["desc"])
                content = [{"type": "image"}, {"type": "text", "text": pr}]
                prefix = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
                texts.append(prefix + _loc_answer(LOC, b["coord"])); images.append(b["images"]); prefixes.append(prefix)
            enc = proc(text=texts, images=images, return_tensors="pt", padding=True)
            ids = enc["input_ids"]; lm = torch.full_like(ids, -100)       # supervise ONLY the answer span (mask prefix + padding)
            for j, pfx in enumerate(prefixes):
                plen = int(proc(text=[pfx], images=[images[j]], return_tensors="pt")["attention_mask"][0].sum())
                am = enc["attention_mask"][j]; lm[j] = ids[j]; lm[j, :plen] = -100; lm[j, am == 0] = -100
            reg_pad = torch.zeros(len(batch), 1, 2048); regfr_pad = torch.full((len(batch), 1), -1, dtype=torch.long)
            return dict(enc=dict(enc), register=reg_pad, reg_frame=regfr_pad,
                        regime=["r2"] * len(batch), nreg_f=[1] * len(batch), lm_labels=lm, loc_r2=True)
        texts, images = [], []
        is_cot_r2 = COT and batch[0]["regime"] == "r2"
        for b in batch:
            nreg = b["reg"].reshape(-1, 2048).shape[0]                    # #<REG> tokens = 1/frame (G1) or 17/frame (E6)
            content = [{"type": "image"} for _ in range(b["nimg"])]
            pr = (PROMPT_R1 if b["regime"] == "r1" else PROMPT_R2).format(nf=b["nreg_f"], d=b["desc"])
            content.append({"type": "text", "text": pr})
            t = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
            t = t + REG * nreg + f' Target: "{b["desc"]}". ' + TEMP       # repeat desc adjacent to readouts (register wall else buries it)
            inst_tok = INST * b["qemb"].shape[0] if (INST_APPEND and b["regime"] == "r2" and "qemb" in b) else ""  # ExpA: append instance-query tokens before <GROUND>
            t = t + (THINK_S + b["cot"] + THINK_E if is_cot_r2 else "") + inst_tok + GROUND   # E5: CoT before <GROUND>
            texts.append(t); images.append(b["images"])
        has_img = any(len(im) > 0 for im in images)
        enc = proc(text=texts, images=(images if has_img else None), return_tensors="pt", padding=True)
        if is_cot_r2:                                                     # LM labels: supervise only the cot span (...<think> COT </think>)
            ids = enc["input_ids"]; lm = torch.full_like(ids, -100)
            for j in range(ids.shape[0]):
                ps = (ids[j] == THINK_S_ID).nonzero(); pe = (ids[j] == THINK_E_ID).nonzero()
                if len(ps) and len(pe): lm[j, int(ps[0]) + 1: int(pe[0]) + 1] = ids[j, int(ps[0]) + 1: int(pe[0]) + 1]
        regs, regfr = [], []
        for b in batch:
            regs.append(torch.from_numpy(b["reg"]).reshape(-1, 2048))     # (ntok, 2048): 1/frame (G1) or 17/frame (E6)
            regfr.append(torch.from_numpy(b["regfr"]) if "regfr" in b else torch.arange(b["nreg_f"]))  # per-token frame index
        Rm = max(r.shape[0] for r in regs)
        reg_pad = torch.zeros(len(batch), Rm, 2048); regfr_pad = torch.full((len(batch), Rm), -1, dtype=torch.long)
        for j, (r, fr) in enumerate(zip(regs, regfr)): reg_pad[j, :r.shape[0]] = r; regfr_pad[j, :len(fr)] = fr
        d = dict(enc=dict(enc), register=reg_pad, reg_frame=regfr_pad,
                 regime=[b["regime"] for b in batch], nreg_f=[b["nreg_f"] for b in batch])
        if is_cot_r2: d["lm_labels"] = lm
        if batch[0]["regime"] == "r1":
            Sm = max(b["nreg_f"] for b in batch); vis_pad = torch.full((len(batch), Sm), -1.0)
            for j, b in enumerate(batch): vis_pad[j, :b["nreg_f"]] = torch.from_numpy(b["vis"])
            d["vis"] = vis_pad
            if ACTION: d["act_label"] = torch.tensor([b.get("act_label", -1.0) for b in batch], dtype=torch.float32)
        if LOC:                                                           # LOC R1: temporal-only (register+vis already set), no spatial targets
            pass
        elif LOSS == "matching":
            M = max(b["qemb"].shape[0] for b in batch); P = batch[0]["qemb"].shape[1]
            ie = torch.zeros(len(batch), M, P); im = torch.zeros(len(batch), M, dtype=torch.bool)
            for j, b in enumerate(batch):
                n = b["qemb"].shape[0]; ie[j, :n] = torch.from_numpy(b["qemb"]); im[j, :n] = True
            d["inst_emb"] = ie; d["inst_mask"] = im; d["target"] = torch.tensor([b["target"] for b in batch])
        else:
            d["gt_mask"] = [torch.from_numpy(b["gt_mask"]) for b in batch]
            d["mask_feat"] = torch.stack([torch.from_numpy(b["field"]) for b in batch])  # (B,n,256,96,128)
        return d

    def collate_g5(batch):                                                # BS=1: R1 turn + R2 turn in ONE multi-turn context
        b = batch[0]; n1, n2, d = b["nreg_r1"], b["nreg_r2"], b["desc"]
        c1 = [{"type": "image"} for _ in range(b["nimg_r1"])] + [{"type": "text", "text": PROMPT_R1.format(nf=n1, d=d)}]
        asst1 = [{"type": "text", "text": REG * n1 + f' Target: "{d}". ' + TEMP + GROUND}]   # R1 readouts (REG wall then desc then tokens, as in G1)
        c2 = [{"type": "image"} for _ in range(b["nimg_r2"])] + [{"type": "text", "text": PROMPT_R2.format(nf=n2, d=d)}]
        t = proc.apply_chat_template([{"role": "user", "content": c1}, {"role": "assistant", "content": asst1},
                                      {"role": "user", "content": c2}], tokenize=False, add_generation_prompt=True)
        t = t + REG * n2 + f' Target: "{d}". ' + GROUND                   # R2 readout (attends to R1 turn via causal history)
        enc = proc(text=[t], images=[b["images_r1"] + b["images_r2"]], return_tensors="pt", padding=True)
        reg = torch.from_numpy(np.concatenate([b["reg_r1"], b["reg_r2"]], 0)).reshape(1, -1, 2048)  # R1 regs then R2 regs (occurrence order)
        regfr = torch.cat([torch.arange(n1), torch.arange(n2)])[None]     # temporal uses reg_h[:, :n1] only
        out = dict(enc=dict(enc), register=reg, reg_frame=regfr, regime=["g5"], nreg_r1=[n1],
                   vis=torch.from_numpy(b["vis"])[None])
        for tag, qk, tk in (("r1", "qemb_r1", "target_r1"), ("r2", "qemb_r2", "target_r2")):
            q = torch.from_numpy(b[qk])
            out[f"inst_emb_{tag}"] = q[None]; out[f"inst_mask_{tag}"] = torch.ones(1, q.shape[0], dtype=torch.bool)
            out[f"target_{tag}"] = torch.tensor([b[tk]])
        return out
    return collate_g5 if IS_G5 else collate


def dice_bce(logits, gt):
    bce = F.binary_cross_entropy_with_logits(logits, gt)
    p = logits.sigmoid(); num = 2*(p*gt).sum() + 1; den = p.sum()+gt.sum()+1
    return bce + (1 - num/den)


FULL_SFT = os.environ.get("FULL_SFT", "0") == "1"   # full fine-tune VLM (DeepSpeed ZeRO-3) instead of LoRA
def main():
    set_seed(0)
    import datetime as _dt
    _pg = InitProcessGroupKwargs(timeout=_dt.timedelta(seconds=int(os.environ.get("NCCL_TIMEOUT_S", "7200"))))
    acc = Accelerator(gradient_accumulation_steps=GA,
                      kwargs_handlers=([_pg] if FULL_SFT else
                                       [_pg, DistributedDataParallelKwargs(find_unused_parameters=True)]))
    if FULL_SFT and getattr(acc.state, "deepspeed_plugin", None) is not None:
        # BS>1 uses a batch_sampler -> DataLoader.batch_size is None -> DeepSpeed can't infer "auto" micro-batch. Set it explicitly.
        acc.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = BS
    LRANK = int(os.environ.get("LORA_RANK", "32")); LALPHA = int(os.environ.get("LORA_ALPHA", "64"))
    model = AgentGrounder(MODEL_PATH, lora=not FULL_SFT, loss=LOSS, mask_embed_trainable=ME_TRAIN,
                          lora_rank=LRANK, lora_alpha=LALPHA)
    me_ckpt = os.environ.get("MASK_EMBED_CKPT")
    if me_ckpt and os.path.exists(me_ckpt):
        model.mask_embed.load_state_dict(torch.load(me_ckpt, map_location="cpu"))
        if acc.is_main_process: print("[agent] loaded segmenter mask_embed", flush=True)
    ds = AgentData(os.environ["MANIFEST"])
    n_train_ds = len(ds)                          # #samples the dl iterates per epoch -> drives total/LR schedule (PHASE2 overrides to the MIXED length, not full replay)
    NW = int(os.environ.get("NUM_WORKERS", "4"))   # workers are PER-PROCESS: 8 GPU procs x 4 = 32 = cpus-per-task (optimal). More -> CPU oversubscription -> SLOWER. persistent+prefetch to hide latency.
    _dl_kw = dict(num_workers=NW, persistent_workers=(NW > 0), prefetch_factor=(4 if NW > 0 else None), pin_memory=True)
    if PHASE2:                                                 # Phase-2: mix phase2 correction (regime 'p2', LM loss on Turn-2) + Arm C GT replay (regime 'mf') ~1:1, dispatch collate on regime
        from selfcorrect_data import SelfCorrectDataset, make_collate as make_collate_p2
        from torch.utils.data import ConcatDataset, Subset
        import random as _rnd
        p2ds = SelfCorrectDataset(P2_SFT); _c_p2 = make_collate_p2(model.processor); _c_mf = make_collate(model.processor)
        valid = [i for i, it in enumerate(ds.items) if it.get("target", -1) >= 0]   # replay from valid-target Arm C queries only
        _rnd.seed(0); ridx = _rnd.sample(valid, min(len(valid), max(1, int(REPLAY_MULT * len(p2ds)))))
        mixed = ConcatDataset([p2ds, Subset(ds, ridx)]); n_train_ds = len(mixed)   # total/LR schedule must use the MIXED length (2190+replay), NOT full replay
        def _mixed_collate(batch):                            # BS=1 single-regime batch -> dispatch on regime
            return _c_p2(batch) if batch[0]["regime"] == "p2" else _c_mf(batch)
        dl = DataLoader(mixed, batch_size=BS, shuffle=True, drop_last=True, collate_fn=_mixed_collate, **_dl_kw)
        if acc.is_main_process: print(f"[agent] PHASE2 mixed: {len(p2ds)} correction + {len(ridx)} replay (1:{REPLAY_MULT}) = {len(mixed)}/epoch", flush=True)
    elif BS > 1 and not IS_G5 and not SFRAME and not MFRAME:   # BS>1: uniform-regime batches (collate/loss read batch[0]'s regime); SFRAME/MFRAME single-regime -> plain loader
        dl = DataLoader(ds, batch_sampler=UniformRegimeBatchSampler(ds, BS), collate_fn=make_collate(model.processor), **_dl_kw)
    else:
        dl = DataLoader(ds, batch_size=BS, shuffle=True, drop_last=True, collate_fn=make_collate(model.processor), **_dl_kw)
    # ---- resume (chained 4h jobs): load lora+heads+step from a prior job's OUT_DIR ----
    RESUME_STEP = 0
    if not FULL_SFT and os.path.exists(f"{OUT_DIR}/meta.json") and os.path.exists(f"{OUT_DIR}/lora"):
        RESUME_STEP = int(json.load(open(f"{OUT_DIR}/meta.json")).get("step", 0))
        if RESUME_STEP > 0:
            import peft.utils.save_and_load as _sl               # peft's TP-shard helper imports EmbeddingParallel (absent in transformers 4.57.1); we use no TP -> no-op it
            _sl._maybe_shard_state_dict_for_tp = lambda m, sd, *a, **k: sd
            from peft import set_peft_model_state_dict, load_peft_weights
            set_peft_model_state_dict(model.vlm, load_peft_weights(f"{OUT_DIR}/lora"))
            h = torch.load(f"{OUT_DIR}/heads.pt", map_location="cpu")
            for k in HEAD_KEYS:
                if k in h: getattr(model, k).load_state_dict(h[k])
            model.logit_scale.data = h["logit_scale"]
            if acc.is_main_process: print(f"[agent] RESUME from step {RESUME_STEP}", flush=True)
    elif FULL_SFT and os.path.exists(f"{OUT_DIR}/meta.json") and os.path.exists(f"{OUT_DIR}/full_model.pt"):
        RESUME_STEP = int(json.load(open(f"{OUT_DIR}/meta.json")).get("step", 0))
        if RESUME_STEP > 0:                                  # full-SFT resume: load entire AgentGrounder state
            model.load_state_dict(torch.load(f"{OUT_DIR}/full_model.pt", map_location="cpu"), strict=False)
            if acc.is_main_process: print(f"[agent] FULL-SFT RESUME from step {RESUME_STEP}", flush=True)
    INIT_CKPT = os.environ.get("INIT_CKPT", "")
    if FULL_SFT and RESUME_STEP == 0 and INIT_CKPT and os.path.exists(INIT_CKPT):   # warm-start weights from a prior model, FRESH LR schedule + step-0
        sd = torch.load(INIT_CKPT, map_location="cpu"); msd = model.state_dict(); loaded = {}
        for k, v in sd.items():                          # adding <POINT> grows embed_tokens/lm_head by 1 row -> load old rows, keep new <POINT> row at init; new point_head absent -> stays init
            if k not in msd: continue
            if msd[k].shape == v.shape: loaded[k] = v
            elif v.dim() == 2 and msd[k].shape[1] == v.shape[1] and msd[k].shape[0] >= v.shape[0]:
                nw = msd[k].clone(); nw[:v.shape[0]] = v; loaded[k] = nw
        model.load_state_dict(loaded, strict=False)
        if acc.is_main_process: print(f"[agent] warm-start INIT from {INIT_CKPT} ({len(loaded)} keys, embed partial-loaded, fresh schedule)", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    HEAD_LR = float(os.environ.get("HEAD_LR", "0") or 0)
    if HEAD_LR > 0:
        _hp = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("vlm.")]
        _bp = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("vlm.")]
        opt = torch.optim.AdamW([{"params": _bp, "lr": LR}, {"params": _hp, "lr": HEAD_LR}], lr=LR, weight_decay=0.01)
        if acc.is_main_process: print(f"[agent] param groups: backbone {len(_bp)} tensors @lr{LR} | head {len(_hp)} tensors @lr{HEAD_LR}", flush=True)
    else:
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    if RESUME_STEP:                                           # cosine sched with last_epoch>0 needs initial_lr preset
        for g in opt.param_groups: g["initial_lr"] = g["lr"]
    if WARMUP_HEADS > 0 and RESUME_STEP < WARMUP_HEADS:        # only still-warming if resumed before warmup end
        for p in lora_params: p.requires_grad_(False)
        if acc.is_main_process: print(f"[agent] G3 head-warmup: LoRA frozen until step {WARMUP_HEADS}", flush=True)
    npx = acc.num_processes
    total = max(1, (n_train_ds//(BS*npx*GA))*EPOCHS)   # n_train_ds = mixed length under PHASE2 (else full ds)
    sched = get_cosine_schedule_with_warmup(opt, WARMUP*npx, total*npx,
                                            last_epoch=(RESUME_STEP*npx - 1) if RESUME_STEP else -1)
    model, opt, dl, sched = acc.prepare(model, opt, dl, sched)
    if acc.is_main_process:
        print(f"[agent] GROUP={GROUP} MIX={MIX} LOSS={LOSS} ME_train={ME_TRAIN} REG_MAX={REG_MAX} "
              f"eff_batch={BS*npx*GA} steps={total} samples={n_train_ds}", flush=True)
    SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "400"))     # periodic save (partition caps at 4h < full run)
    def save_ckpt(tag=""):
        acc.wait_for_everyone()
        if FULL_SFT:
            sd = acc.get_state_dict(model)                        # collective: gathers ZeRO-3 shards (all ranks call)
            if acc.is_main_process:
                os.makedirs(OUT_DIR, exist_ok=True)
                torch.save(sd, f"{OUT_DIR}/full_model.pt")        # entire AgentGrounder (vlm+heads), 16GB
                json.dump({"step": cur_step[0], "full": True}, open(f"{OUT_DIR}/meta.json", "w"))
                print(f"[agent] saved FULL{tag} step={cur_step[0]} -> {OUT_DIR}", flush=True)
            return
        if not acc.is_main_process: return
        os.makedirs(OUT_DIR, exist_ok=True); u = acc.unwrap_model(model)
        u.vlm.save_pretrained(f"{OUT_DIR}/lora")
        torch.save({k: getattr(u, k).state_dict() for k in HEAD_KEYS}
                   | {"logit_scale": u.logit_scale.data.cpu()}, f"{OUT_DIR}/heads.pt")
        json.dump({"step": cur_step[0]}, open(f"{OUT_DIR}/meta.json", "w"))   # for chained resume
        print(f"[agent] saved{tag} step={cur_step[0]} -> {OUT_DIR}", flush=True)
    cur_step = [RESUME_STEP]
    def mark_done():   # chain sentinel (slurm guide §10.1 #1): trainer writes DONE at total_steps, else the
        if acc.is_main_process:   # self-chain keeps resubmitting links that instantly exit ("already at step") = infinite chain
            os.makedirs(OUT_DIR, exist_ok=True); open(f"{OUT_DIR}/DONE", "w").write(str(cur_step[0]))
            print(f"[agent] wrote DONE @step{cur_step[0]} -> chain stops", flush=True)
    um = acc.unwrap_model(model); step = RESUME_STEP
    # FIX (all groups + full-SFT): gradient checkpointing only activates in train mode, but the trainer
    # never set it -> grad-ckpt was a silent no-op (G1-4 fit <80GB by luck; G5's ~5k-tok seq OOMs).
    # Enable train mode AND force dropout=0 so this is NUMERICALLY IDENTICAL to the original eval-mode
    # regime (dropout was already off in eval) -- grad-ckpt is exact, so results are unchanged; only memory drops.
    model.train()
    nd = 0
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout) and mod.p > 0: mod.p = 0.0; nd += 1
    if acc.is_main_process:
        print(f"[agent] train mode (grad-ckpt ACTIVE) + {nd} dropout->0 (numerically == original eval-mode regime)", flush=True)
    if step >= total:
        if acc.is_main_process: print(f"[agent] already at step {step}>={total}; nothing to do", flush=True)
        mark_done(); return
    for ep in range(EPOCHS):
        for b in dl:
            with acc.accumulate(model):
                enc = b["enc"]; mm = {k: enc[k] for k in enc if k not in ("input_ids", "attention_mask")}
                if IS_G5:                                             # history multi-turn: temporal(R1) + spatial(R1) + spatial(R2), one forward
                    temp_h, gh1, gh2, reg_h, regfr = um.encode(enc["input_ids"], enc["attention_mask"],
                                                               b["register"], b["reg_frame"], multi_readout=True, **mm)
                    n1 = b["nreg_r1"][0]                              # temporal uses only the R1-segment register tokens
                    tl = um.temporal_logits(temp_h, reg_h[:, :n1], regfr[:, :n1], n1)
                    vmask = b["vis"] >= 0
                    loss = F.binary_cross_entropy_with_logits(tl[vmask], b["vis"][vmask].to(tl.dtype))
                    loss = loss + F.cross_entropy(um.match_logits(gh1, b["inst_emb_r1"], b["inst_mask_r1"]), b["target_r1"])  # R1 all-frame
                    loss = loss + F.cross_entropy(um.match_logits(gh2, b["inst_emb_r2"], b["inst_mask_r2"]), b["target_r2"])  # R2 selected (sees R1 history)
                else:
                    want_logits = ((COT or LOC) and b["regime"][0] == "r2") or ((COT or BOX_OUT) and b["regime"][0] == "mf") or (b["regime"][0] == "p2")  # E5/LOC r2, MF+CoT/Arm C box-list, or PHASE2 2-pass: need LM logits
                    _inst_kw = {"inst_emb": b["inst_emb"], "inst_valid": b["inst_mask"]} if (INST_APPEND and b["regime"][0] == "r2") else {}
                    _mg = {"multi_ground": True} if (MFRAME or b["regime"][0] == "p2") else {}
                    eo = um.encode(enc["input_ids"], enc["attention_mask"], b["register"], b["reg_frame"],
                                   return_logits=want_logits, **_mg, **_inst_kw, **mm)
                    temp_h, ground_h, reg_h, regfr = eo[:4]; lm_logits = eo[4] if want_logits else None
                    loss = 0.0
                    if b["regime"][0] == "r1":                            # temporal BCE only in R1
                        S = b["vis"].shape[1]
                        tl = um.temporal_logits(temp_h, reg_h, regfr, S)
                        vmask = b["vis"] >= 0
                        loss = loss + F.binary_cross_entropy_with_logits(tl[vmask], b["vis"][vmask].to(tl.dtype))
                        if SEL_HEAD:                                      # E4: learn #frames-to-select (count) vs hard 0.5/top16/cap
                            cnt_pred = um.count_pred(temp_h)              # (B,)
                            cnt_gt = (b["vis"] > 0).sum(1).to(cnt_pred.dtype)   # #GT-visible frames per sample
                            loss = loss + SEL_W * F.smooth_l1_loss(cnt_pred, cnt_gt)
                        if ACTION:                                        # E2: stop policy -- P(refine) BCE, ignore samples w/o r1-pred label
                            al = b["act_label"]; m = al >= 0
                            if m.any():
                                ap = um.action_logit(temp_h)[m]
                                loss = loss + ACT_W * F.binary_cross_entropy_with_logits(
                                    ap, al[m].to(ap.dtype), pos_weight=torch.tensor(ACT_POSW, device=ap.device, dtype=ap.dtype))
                    if b["regime"][0] == "p2":                            # PHASE2 2-pass self-correction
                        lab = b["lm_labels"].to(lm_logits.device); V = lm_logits.size(-1)
                        loss = loss + F.cross_entropy(lm_logits[:, :-1].reshape(-1, V), lab[:, 1:].reshape(-1), ignore_index=-100)
                        if "p2_qtargets" in b:                            # Turn-2 re-emits Turn-1's structure, so its <GROUND>
                            # readouts carry the CORRECTED instance choice and need the same matching loss Turn-1 gets.
                            # ground_h holds every <GROUND> in occurrence order: the first NF are Turn-1's (context,
                            # no loss), the LAST NF are Turn-2's.
                            NF2 = b["p2_qtargets"].shape[1]
                            gh2 = ground_h[:, -NF2:]
                            gl = 0.0
                            for k in range(NF2):
                                ml = um.match_logits(gh2[:, k], b["inst_emb"], b["inst_mask"])
                                gl = gl + F.cross_entropy(torch.cat([ml, um.null_logit(gh2[:, k])], 1), b["p2_qtargets"][:, k])
                            loss = loss + P2_GROUND_W * (gl / NF2)
                    elif MFRAME:                                          # multi-frame: NF <GROUND>, each matches ITS frame's [queries ; null] -> mean CE
                        G = ground_h.shape[1]
                        for k in range(G):
                            ml = um.match_logits(ground_h[:, k], b["inst_emb"], b["inst_mask"])
                            loss = loss + F.cross_entropy(torch.cat([ml, um.null_logit(ground_h[:, k])], 1), b["mf_targets"][:, k])
                        loss = loss / G
                        if POINT_HEAD and getattr(um, "_point_h", None) is not None and "mf_points" in b:   # Arm B: <POINT> cross-attends patch grid -> (x,y,null), indep of matching branch
                            B_, G_, H_ = um._point_h.shape; hd = next(um.point_head.parameters()).dtype
                            q = um._point_h.reshape(B_ * G_, H_).to(hd)
                            if "mf_patch" in b: pg = b["mf_patch"].to(q.device).reshape(B_ * G_, PATCH_GH * PATCH_GW, 2048)
                            else: pg = torch.zeros(B_ * G_, PATCH_GH * PATCH_GW, 2048, device=q.device, dtype=hd)   # no visual pool -> zero grid
                            o = um.point_head(q, pg).float()                                # (M,3): xy(pre-sigmoid), null logit
                            pred = torch.sigmoid(o[:, :2]); null_logit = o[:, 2]
                            pv = b["mf_point_valid"].to(q.device).reshape(B_ * G_)
                            pts = b["mf_points"].to(q.device).float().reshape(B_ * G_, 2)
                            if pv.any(): loss = loss + POINT_W * F.l1_loss(pred[pv], pts[pv])    # coords: L1 only on present frames
                            loss = loss + POINT_W * POINT_NULL_W * F.binary_cross_entropy_with_logits(null_logit, (~pv).float())   # null: 1=absent, all frames
                    elif SFRAME:                                          # single-frame: <GROUND> matches [qemb queries ; null] -> CE (null = column M)
                        ml = um.match_logits(ground_h, b["inst_emb"], b["inst_mask"])
                        loss = loss + F.cross_entropy(torch.cat([ml, um.null_logit(ground_h)], 1), b["target"])
                    elif (R1_SELECT_ONLY or LOC) and b["regime"][0] == "r1":  # E7/LOC: R1 = pure frame selector, NO grounding loss
                        pass
                    elif LOC:                                             # LOC R2: pure coord-text LM (loss added by the want_logits block below)
                        pass
                    elif LOSS == "matching":                              # spatial: R2 (and R1 unless R1_SELECT_ONLY)
                        if INST_APPEND and b["regime"][0] == "r2":        # ExpA: match <GROUND> vs contextualized <INST> output hiddens
                            ml = um.inst_match_logits(ground_h, um._inst_h, b["inst_mask"])
                        else:
                            ml = um.match_logits(ground_h, b["inst_emb"], b["inst_mask"])
                        loss = loss + F.cross_entropy(ml, b["target"])
                    else:
                        mf = b["mask_feat"].to(ground_h.device)
                        pm = um.decode_mask(ground_h, mf)
                        dl_ = sum(dice_bce(pm[j], b["gt_mask"][j].to(pm.device, pm.dtype)) for j in range(len(b["gt_mask"]))) / len(b["gt_mask"])
                        loss = loss + dl_
                    if want_logits and "lm_labels" in b and b["regime"][0] != "p2":   # E5/MF+CoT: LM CE on <think> CoT / box span (p2 already did its Turn-2 LM loss above)
                        lab = b["lm_labels"].to(lm_logits.device); V = lm_logits.size(-1)
                        if (lab[:, 1:] != -100).any():                    # guard: skip when a batch has no CoT text (all-ignored -> CE would be NaN)
                            loss = loss + COT_W * F.cross_entropy(lm_logits[:, :-1].reshape(-1, V), lab[:, 1:].reshape(-1), ignore_index=-100)
                    if FULL_SFT:   # ZeRO: ensure ALL heads get a grad every step (regime/loss-conditional heads else desync grad reduction)
                        loss = loss + 0.0 * sum(p.sum() for k in HEAD_KEYS
                                                for p in getattr(um, k).parameters()) + 0.0 * um.logit_scale.sum()
                acc.backward(loss)
                if acc.sync_gradients: acc.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            if acc.sync_gradients:
                step += 1; cur_step[0] = step
                if WARMUP_HEADS and step == WARMUP_HEADS:
                    for p in lora_params: p.requires_grad_(True)
                    if acc.is_main_process: print(f"[agent] G3 warmup done @ step{step}: LoRA unfrozen", flush=True)
                if acc.is_main_process and step % 10 == 0:
                    _mem = torch.cuda.max_memory_allocated() / 1e9
                    print(f"ep{ep} step{step}/{total} [{b['regime'][0]}] loss {float(loss):.3f} lr {sched.get_last_lr()[0]:.2e} peakmem {_mem:.1f}G", flush=True)
                if step % SAVE_EVERY == 0: save_ckpt(f"@step{step}")   # survive 4h timeout / chained resume
                if step >= total or (MAX_STEPS and step >= MAX_STEPS): break
        if step >= total or (MAX_STEPS and step >= MAX_STEPS): break
    save_ckpt(" (final)")
    if step >= total or (MAX_STEPS and step >= MAX_STEPS): mark_done()   # reached the target -> stop the chain


if __name__ == "__main__": main()
