#!/usr/bin/env python
"""N-round agentic inference on ScanRefer val.
Round 1: segmenter on ALL frames -> register; VLM(uniform frames + register + <TEMP><GROUND>)
         -> per-frame temporal scores -> select top-k frames F.
Round 2..N: segmenter re-inference on F (full forward, segmenter sweet spot) -> register + instances;
         VLM(F frames + register) -> TEMP (refine F) + GROUND (match instance / decode mask).
Output: final GROUND result -> 3D mask -> box IoU. Sweep N via env.
NOTE: needs the segmenter (align3d.build_omega) AND the trained grounder in one environment."""
import os, sys, json, re
import _env
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
WORK = os.environ.get("GROUNDING_REPO", os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, WORK)
import align3d as AE
from align3d import build_omega, load_single_scene_data, map_pred_inst_to_gt_pointcloud, SCORE_THR
from agent_model import AgentGrounder, REG, TEMP, GROUND, THINK_S, THINK_E, INST, POINT
from clip_embed import gt_masks, aabb_iou, SR, PTS, VAL

CKPT = _env.require("CKPT", "directory holding the trained grounder full_model.pt"); GROUP = os.environ.get("GROUP", "g1"); NROUND = int(os.environ.get("NROUND", "2"))
NF_VLM = int(os.environ.get("NF_VLM", "8")); TOPK = int(os.environ.get("TOPK", "16"))
REG_MAX = int(os.environ.get("REG_MAX_FRAMES", "128"))   # R1 all-frame register cap (match training)
ORACLE = os.environ.get("EVAL_ORACLE", "0") == "1"       # DIAG: use GT-visible frames for R2 (bypass temporal selection)
DIAG = os.environ.get("EVAL_DIAG", "0") == "1"            # log head-pick vs oracle-best instance IoU
ALLFRAME = os.environ.get("ALLFRAME_INST", "0") == "1"   # OFF by design: R2 answer = SELECTED-frame re-run instances (not all-frame)
SEL_HEAD = os.environ.get("SEL_HEAD", "0") == "1"        # E4: learned count head -> top-K selection (vs hard >0.5/top16)
REG17 = os.environ.get("REG17", "0") == "1"              # E6: feed all 17 segmenter reg tokens/frame (no pooling) + slot identity
ACTION = os.environ.get("ACTION", "0") == "1"            # E2: action head -> ground-vs-refine stop policy
COT = os.environ.get("COT", "0") == "1"                  # E5: generate <think> CoT then read <GROUND>
TEMPORAL = os.environ.get("TEMPORAL", "0") == "1"        # TEMPORAL-only eval: R1 frame-selection quality vs GT-visible (no R2). Records per-frame scores + GT.
FIXEDQ = os.environ.get("FIXEDQ", "0") == "1"            # teacher fixed-Q: single all-frame forward -> fixed register + Q; R1 selects frames, R2 matches the SAME fixed qemb (matches training).
CEILING = os.environ.get("CEILING", "0") == "1"          # query-perfect oracle: skip VLM, take the best-IoU instance in the fixed Q (= segmenter instance ceiling, no R1/no-match error).
FQ_CAP = int(os.environ.get("FQ_CAP", "256"))            # frames for the single fixed-Q forward (cap; OOM -> auto-retry at 128)
LOCATE = os.environ.get("LOCATE", "0") == "1"            # replace VLM <GROUND> matching with LocateAnything-3B: per frame -> box -> back-map to teacher instance (2D-IoU vote) -> inst3d
LA_DIR = os.environ.get("LOCATE_ANYTHING_ROOT", "/PATH/TO/hf_models/LocateAnything-3B")
LA_K = int(os.environ.get("LA_K", "8"))                  # LOCATE: ground the top-K R1 frames (dump per-frame -> offline sweep of vote/filter/K)
INST_APPEND = os.environ.get("INST_APPEND", "0") == "1"  # ExpA eval: R2 appends <INST> tokens (qemb injected), <GROUND> matches contextualized <INST> hiddens
SFRAME = os.environ.get("SFRAME", "0") == "1"            # single-frame grounder: ground each frame -> match a qemb query OR null; vote across frames -> target instance. No R1.
SF_CAP = int(os.environ.get("SF_CAP", "32"))             # frames to ground per query (uniform-downsample if scene has more); all-null vote -> "not found"
SF_NOREG = os.environ.get("SF_NOREG", "0") == "1"        # ablation: SFRAME without teacher register tokens (image-only via vision tower)
MFRAME = os.environ.get("MFRAME", "0") == "1"            # multi-frame: NF-frame sliding window (stride=NF-overlap) over all frames; NF <GROUND>/forward, overlapping frames vote
MF_NF = int(os.environ.get("MF_NF", "8"))
MF_OVERLAP = int(os.environ.get("MF_OVERLAP", "4"))
MF_BATCH = int(os.environ.get("MF_BATCH", "1"))
RESUME_Q = os.environ.get("RESUME_Q") == "1"
MF_TREE = os.environ.get("MF_TREE") == "1"          # 8-ary coarse-to-fine frame search
MF_TREE_DEPTH = int(os.environ.get("MF_TREE_DEPTH", "4"))   # windows per box-generate call; 1 = the original per-window path      # sliding-window overlap; stride = MF_NF - MF_OVERLAP
MF_COT = os.environ.get("MF_COT", "0") == "1"            # MF+CoT eval: reason (<think>) before the per-frame grounds
MF_ONESHOT = os.environ.get("MF_ONESHOT", "0") == "1"    # MF32: ONE forward per query -- register for MF_NF uniform frames, tower sees MF_IMG uniform images, MF_NF grounds -> vote
MF_IMG = int(os.environ.get("MF_IMG", "0"))              # tower image count for MF32 (0 = all frames, original MF)
MF_COT_GEN = os.environ.get("MF_COT_GEN", "1") == "1"
P2_CORRECT = os.environ.get("P2_CORRECT", "0") == "1"    # Phase-2 second pass INSIDE the sliding window: after Pass-1
#   boxes+qemb for a window, inject [REVIEW] and generate Turn-2; the CORRECTED boxes (not Pass-1's) feed the
#   per-frame vote and therefore MF_DUMP_BOX -> s4vy_prompt -> box@.5. Without this the dump is Pass-1 only.
P2_REVIEW_CONF = os.environ.get("P2_REVIEW_CONF", "0") == "1"   # must match the SFT the ckpt was trained on
P2_ROUNDS = int(os.environ.get("P2_ROUNDS", "1"))        # 1 = single correction; 2 = feed Turn-2's boxes back as Pass-1 and correct again    # 1=GENERATE the CoT (cot1, matches training); 0=skip the <think> span (cot0, off-distribution, fast sanity)
P2_DONE_STOP = os.environ.get("P2_DONE_STOP") == "1"     # <done/> ends the loop for that window AFTER this round's
                                                         # output is taken. P2_ROUNDS then acts as a HARD CAP, not a
                                                         # fixed count -- pair with P2_ROUNDS>1 or it is a no-op.
VIDI_SEL = os.environ.get("VIDI_SEL", "")                # dir of Vidi TR jsons (with "ranges"): MF windows slide only over Vidi-selected frames (dilated to >=MF_NF); no/empty ranges -> full-S fallback
MF_PROMPT = 'Look at these {n} video frames. For each frame, if the object described below appears in it, ground that object; if it does not appear in that frame, report absent.\nDescription: "{d}"'
SF_PROMPT = 'Look at this single image. If the object described below appears in it, ground that object; if it does not appear, report absent.\nDescription: "{d}"'


def _iou_box(a, b):   # xyxy IoU, both normalized [0,1]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
    ua = max(0.0,(a[2]-a[0])*(a[3]-a[1])) + max(0.0,(b[2]-b[0])*(b[3]-b[1])) - inter
    return inter/ua if ua > 0 else 0.0


@torch.no_grad()
def _la_box(la_model, la_proc, frame_np, desc):
    """LocateAnything: one frame + description -> normalized [0,1] xyxy box (or None)."""
    im = Image.fromarray(frame_np.astype(np.uint8))
    if im.width > 1024: im = im.resize((1024, int(im.height*1024/im.width)))
    prompt = f"Locate a single instance that matches the following description: {desc}."
    msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": prompt}]}]
    text = la_proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = la_proc(text=[text], images=[im], return_tensors="pt").to("cuda")
    out = la_model.generate(**inputs, tokenizer=la_proc.tokenizer, max_new_tokens=64, do_sample=False, use_cache=True, generation_mode="hybrid")
    s = out[0] if isinstance(out, (list, tuple)) else str(out)
    m = re.search(r"<box>(.*?)</box>", s)
    if not m: return None
    v = re.findall(r"\d+", m.group(1))
    return [int(v[i])/1000.0 for i in range(4)] if len(v) >= 4 else None


def _la_match(la_boxes, masks_frames):
    """la_boxes: list of box|None per selected frame; masks_frames: (n, k, h, w) instance 2D masks on those k frames.
    Vote: each frame's LA box -> += IoU to each instance's 2D-mask box in that frame; return argmax instance (or -1)."""
    n, k, h, w = masks_frames.shape
    votes = np.zeros(n)
    for j, box in enumerate(la_boxes):
        if box is None: continue
        for i in range(n):
            mk = masks_frames[i, j]
            if mk.sum() == 0: continue
            ys, xs = np.where(mk)
            ib = [xs.min()/w, ys.min()/h, (xs.max()+1)/w, (ys.max()+1)/h]
            votes[i] += _iou_box(box, ib)
    return int(votes.argmax()) if votes.max() > 0 else -1
LOC = os.environ.get("LOC", "")                          # SAM-hint baseline: "point"|"bbox". R1 -> top-1 key frame; R2 = Qwen3-VL generate coord text.
MW, MH = 128, 96                                          # gt_idmap 2D res (W,H); pred coords are normalized [0,1]


def _frame_ap(y_true, y_score):
    """average precision for per-frame binary visibility given the R1 sigmoid scores (rank-based AP)."""
    import numpy as _np
    y_true = _np.asarray(y_true); y_score = _np.asarray(y_score)
    npos = float(y_true.sum())
    if npos == 0:
        return float("nan")
    o = _np.argsort(-y_score); yt = y_true[o]
    tp = _np.cumsum(yt); prec = tp / _np.arange(1, len(yt) + 1)
    return float((prec * yt).sum() / npos)
_D = _env.data_root()
_SEG = os.environ.get("SEG_TAG", "segmenter")   # model-derived cache tag (segmenter default; set teacher to match the swapped INFER_CKPT)
VALc = f"{_D}/{_SEG}_val_cache"
IDMc = f"{_D}/gt_idmap_cache"  # GT, model-agnostic
PATCHc = f"{_D}/s4vy_patch_cache"  # Arm B: patch grid for <POINT> visual pool
PATCH_VP = os.environ.get("PATCH_VP", "0") == "1"
PATCH_GH, PATCH_GW = int(os.environ.get("PATCH_GH", "12")), int(os.environ.get("PATCH_GW", "16"))
BOX_OUT = os.environ.get("BOX_OUT", "0") == "1"          # Arm C: generate per-frame text bbox list after the ground blocks, parse + box@.5 vs GT
LOSS = "dense" if GROUP in ("g2", "g4") else "matching"
OUT = os.environ.get("SR_OUT", f"{_D}/scanrefer_{GROUP}_N{NROUND}")
os.makedirs(OUT, exist_ok=True); _tl = {}


def _r9_hw(images_np, R=int(os.environ.get("R9_RES", "512")), p=16):
    """VGGT-Omega official preprocessing target (mode=balanced): keep token count ~= (R/p)^2 while
    preserving aspect ratio, both dims divisible by patch p. Aspect clamped to [0.5,2.0] (= official crop).
    Set R9_RES=0 to fall back to the legacy fixed 384x512 (max_size variant)."""
    if R <= 0: return (384, 512)
    H, W = int(images_np.shape[1]), int(images_np.shape[2])
    ar = min(2.0, max(0.5, H / max(W, 1)))
    tok = (R // p) ** 2
    wp = max(1, round((tok / ar) ** 0.5)); hp = max(1, round(tok / wp))
    return (hp * p, wp * p)


@torch.no_grad()
def register_only(model, images_np, chunk=120):
    """R1: only need per-frame register (to select frames). Backbone-only + chunk -> no OOM on big scenes."""
    import torch.nn as nn
    t = torch.from_numpy(images_np).float().div(255.).permute(0,3,1,2)
    t = F.interpolate(t, size=_r9_hw(images_np), mode="bilinear", align_corners=False).cuda()
    S = t.shape[0]; regs = []
    for s0 in range(0, S, chunk):
        bb = model.backbone(t[s0:s0+chunk], run_depth=False, run_camera=False)
        tl = [x for x in bb["tokens_list"] if x is not None][-1]
        r = tl[0, :, :17].float()                                     # (chunk,17,2048)
        regs.append((r if (REG17 or SFRAME or MFRAME) else r.mean(1)).cpu().numpy())  # E6/SFRAME/MFRAME: keep 17/frame; else pool to 1/frame
    return np.concatenate(regs, 0)                                    # (S,17,2048) E6/SFRAME | (S,2048) default


@torch.no_grad()
def segmenter_forward(model, decode, images_np, want_field=False, return_soft=False):
    """Full segmenter forward on the given frames -> register(S,17,2048), instances(masks,scores,qemb), [mask_feat].

    return_soft=False (default, eval path): masks are BOOLEAN at native H/4 (96x128).
    return_soft=True (viz path): masks are the SOFT sigmoid probabilities for the kept instances,
        so the caller can bilinear-upsample the soft map and threshold at display resolution
        (avoids the blocky nearest-upsample-of-a-binarized-low-res-mask artifact)."""
    S = images_np.shape[0]
    t = torch.from_numpy(images_np).float().div(255.).permute(0,3,1,2)
    t = F.interpolate(t, size=_r9_hw(images_np), mode="bilinear", align_corners=False).cuda()
    fp = model.query_decoder.frame_pos
    if S > fp.num_embeddings:
        import torch.nn as nn
        w = fp.weight.data; wn = F.interpolate(w.t()[None], size=S, mode="linear", align_corners=False)[0].t().contiguous()
        ne = nn.Embedding(S, w.shape[1]).to(w.device); ne.weight.data = wn; model.query_decoder.frame_pos = ne
    out = model(t)
    _r = [x for x in _tl["tl"] if x is not None][-1][0, :, :17].float()                          # (S,17,2048)
    reg = (_r if REG17 else _r.mean(1)).cpu().numpy()                                            # E6: 17/frame | default pooled
    scores_all = out["objectness"][0].softmax(-1)[..., 0]; keep = scores_all > SCORE_THR
    ml = out["mask_logits"][0]; prob = ml.sigmoid(); masks = (prob > 0.5) & keep.view(-1,1,1,1)
    keep = keep & (masks.flatten(1).sum(-1) > 10)
    qemb = _tl["q"][0][keep].float().cpu().numpy()
    _tl["keep_idx"] = torch.where(keep)[0].cpu().numpy()          # kept-row -> original query slot
    masks_out = prob[keep] if return_soft else masks[keep].float()
    return reg, masks_out.float().cpu().numpy(), scores_all[keep].cpu().numpy(), qemb


def main():
    m9, decode = build_omega()
    m9.backbone.register_forward_hook(lambda mod,i,o: _tl.__setitem__("tl", o["tokens_list"]))
    m9.query_decoder.decoder_norm.register_forward_hook(lambda mod,i,o: _tl.__setitem__("q", o.detach()))
    m9.mask_feature_head.register_forward_hook(lambda mod,i,o: _tl.__setitem__("field", o.detach()))  # G2/G4 dense lift
    ag = AgentGrounder(_env.require("MODEL_PATH", "base Qwen3-VL checkout"), lora=False, loss=LOSS).cuda().eval()
    if os.path.exists(f"{CKPT}/full_model.pt"):                # full-SFT: load entire AgentGrounder (vlm+heads)
        ag.load_state_dict(torch.load(f"{CKPT}/full_model.pt", map_location="cpu"), strict=False)
    else:
        from peft import PeftModel
        ag.vlm = PeftModel.from_pretrained(ag.vlm, f"{CKPT}/lora").cuda()
        h = torch.load(f"{CKPT}/heads.pt", map_location="cpu")
        for k in ["reg_proj","reg_ln","temporal_head","text_proj","inst_proj","gq_proj","mask_embed"] \
                 + (["count_head"] if SEL_HEAD else []) + (["action_head"] if ACTION else []) + (["reg_slot_emb"] if REG17 else []):
            if k in h: getattr(ag, k).load_state_dict(h[k])
        ag.logit_scale.data = h["logit_scale"].cuda()
    proc = ag.processor
    la_model = la_proc = None
    if LOCATE:                                                # LocateAnything-3B: box grounder that replaces the VLM <GROUND> matcher
        from transformers import AutoModel, AutoProcessor
        la_proc = AutoProcessor.from_pretrained(LA_DIR, trust_remote_code=True)
        la_model = AutoModel.from_pretrained(LA_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
        print("[locate] LocateAnything-3B loaded", flush=True)

    if os.environ.get("R1PRED_DUMP") == "1":                  # ExpB prep: dump R1 top-8 frames per TRAIN manifest query -> {scene}.json {oid__qid: [top8 abs frames]}
        import json as _json
        man = [_json.loads(l) for l in open(os.environ["MANIFEST_DUMP"])]
        for qi, it in enumerate(man): it["_qid"] = qi
        S2Dr = os.environ.get("SCANNET200_2D_ROOT", "/PATH/TO/codebase/SegVGGT/data_processing/scannet200/scenes_2d")
        byc = {}
        for it in man: byc.setdefault(it["scene"], []).append(it)
        keep = set(l.strip() for l in open(os.environ["SCENE_LIST"])) if os.environ.get("SCENE_LIST") else None
        for sc in sorted(byc):
            if keep and sc not in keep: continue
            of = f"{OUT}/{sc}.json"
            if os.path.exists(of): continue
            try:
                sample = load_single_scene_data("data_processing/scannet200", sc, "scannet200", 1)
                if sample is None: _json.dump({}, open(of, "w")); continue
                imgs_all = sample["images"]; S = imgs_all.shape[0]
                reg_all = register_only(m9, imgs_all)
                fidx = np.arange(S) if S <= REG_MAX else np.linspace(0, S-1, REG_MAX).round().astype(int)
                reg_r1 = reg_all[fidx]; r1_imgs = imgs_all[np.linspace(0, S-1, min(S, NF_VLM)).round().astype(int)]
                out = {}
                for it in byc[sc]:
                    scores = _agent_round_r1(ag, proc, reg_r1, it["desc"], len(fidx), r1_imgs)
                    out[f"{it['oid']}__{it['_qid']}"] = [int(x) for x in np.sort(fidx[np.argsort(-scores)[:NF_VLM]])]
                _json.dump(out, open(of, "w")); print(f"{sc}: {len(out)} queries", flush=True)
            except Exception as e:
                print(f"{sc}: FAIL {type(e).__name__}: {e}", flush=True)
        return
    valset = set(re.findall(r"scene[0-9]{4}_[0-9]{2}", open(VAL).read()))
    # SPLIT=train: run the SAME prediction pipeline on the training scenes. The critic's input has to be g2's
    # real predictions or it sees a different distribution at inference than it was trained on; generating them
    # needs the train half of ScanRefer, which this filter otherwise excludes.
    if os.environ.get("SPLIT", "val") == "train":
        sr = [x for x in json.load(open(SR)) if x["scene_id"] not in valset]
    else:
        sr = [x for x in json.load(open(SR)) if x["scene_id"] in valset]
    by = {}
    for x in sr: by.setdefault(x["scene_id"], []).append(x)
    scenes = sorted(by)
    sl = os.environ.get("SCENE_LIST")
    if sl: keep = set(l.strip() for l in open(sl)); scenes = [s for s in scenes if s in keep]
    print(f"[agent-eval] GROUP={GROUP} N={NROUND} {len(scenes)} scenes -> {OUT}", flush=True)

    for sc in scenes:
        of = f"{OUT}/{sc}.json"
        if os.path.exists(of): continue
        try:
            sample = load_single_scene_data("data_processing/scannet200", sc, "scannet200", 1)
            if sample is None: json.dump([], open(of,"w")); continue
            imgs_all = sample["images"]; S = imgs_all.shape[0]
            coord = np.fromfile(f"{PTS}/{sc}.bin", np.float32).reshape(-1,6)[:,:3]
            gtm, Nseg = gt_masks(sc)
            reg_all = register_only(m9, imgs_all)                      # (S,17,2048) once per scene
            fidx = np.arange(S) if S <= REG_MAX else np.linspace(0, S-1, REG_MAX).round().astype(int)
            reg_r1 = reg_all[fidx]
            r1_imgidx = np.linspace(0, S-1, min(S, NF_VLM)).round().astype(int)   # R1 visual context: uniform-8 images
            r1_imgs = imgs_all[r1_imgidx]
            idm = np.load(f"{IDMc}/{sc}.npz")["idmap"] if (ORACLE or TEMPORAL or LOC or FIXEDQ) else None  # GT-visible frames (oracle R2 / temporal / LOC / fixed-Q oracle)
            patch_all = None
            if getattr(ag, "point_out", False) and PATCH_VP:               # Arm B: patch grid for <POINT> visual pool (per scene)
                try: patch_all = np.load(f"{PATCHc}/{sc}.npz")["patch"]     # (S,GH,GW,2048) fp16
                except Exception: patch_all = None
            vmap = None
            if VIDI_SEL and MFRAME and os.path.exists(f"{VIDI_SEL}/{sc}.json"):
                vmap = {(str(r["oid"]), str(r.get("ann_id"))): r.get("ranges", []) for r in json.load(open(f"{VIDI_SEL}/{sc}.json"))}
            qemb_all = inst3d_all = None
            masks_all = None
            if FIXEDQ:                                                     # single forward -> fixed Q (qemb) + inst3d (answers). LOCATE: forward on fidx (align R1 frames) + keep 2D masks.
                caps = [len(fidx)] if LOCATE else [FQ_CAP, 128, 96]        # LOCATE: forward exactly the R1 register frames so top-8 index masks_all directly
                for _cap in caps:
                    try:
                        fcap = fidx if LOCATE else (np.arange(S) if S <= _cap else np.linspace(0, S-1, _cap).round().astype(int))
                        _mf_fcap = fcap
                        _, masks_fq, _, qemb_all = segmenter_forward(m9, decode, imgs_all[fcap])
                        if qemb_all.shape[0] > 0:
                            inst3d_all = map_pred_inst_to_gt_pointcloud(sample["points"], sample["super_points"],
                                sample["intrinsics"][fcap], sample["extrinsics"][fcap], sample["depth_maps"][fcap], masks_fq, depth_tolerance=0.1)
                            masks_all = (masks_fq > 0.5) if masks_fq.dtype != bool else masks_fq   # (n, len(fcap), h, w) per-frame 2D masks for LOCATE
                        break
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache(); print(f"{sc}: fixedq OOM at cap={_cap}, retrying smaller", flush=True)
                        if _cap == caps[-1]: raise
            elif ALLFRAME:                                                 # FIX: answer = full-coverage all-frame segmenter instance (from cache)
                zc = np.load(f"{VALc}/{sc}.npz", allow_pickle=True)
                if "empty" not in zc.files:
                    qemb_all = zc["qemb"].astype(np.float32)
                    inst3d_all = np.unpackbits(zc["inst3d"], axis=1)[:, :int(zc["N"])].astype(bool)
            if (SFRAME or MFRAME) and qemb_all is not None and qemb_all.shape[0] > 0:               # SF/MF: pre-normalized match-space qemb (scene-level, reused across queries)
                _qt = torch.from_numpy(qemb_all).float().cuda()
                sf_e = F.normalize(_qt if ag.no_inst_proj else ag.inst_proj(_qt), dim=-1)           # ablation NO_INST_PROJ: match raw qemb (no query-side proj)
            else:
                sf_e = None
            sf_frames = np.arange(S) if S <= SF_CAP else np.linspace(0, S-1, SF_CAP).round().astype(int)  # ALL S frames (not fidx-capped); downsample only if S>SF_CAP
            _part = f"{OUT}/{sc}.partial.jsonl"
            _done_q = {}
            if RESUME_Q and os.path.exists(_part):
                try:
                    for _l in open(_part):
                        try:
                            _r = json.loads(_l); _done_q[(_r.get("oid"), _r.get("ann_id"))] = _r
                        except Exception: pass
                except Exception as _e:
                    print(f"{sc}: partial read failed {_e}, running scene from scratch", flush=True); _done_q = {}
                if _done_q: print(f"{sc}: resuming, skipping completed {len(_done_q)} q", flush=True)

            class _Res(list):
                def append(self, r):
                    super().append(r)
                    if RESUME_Q:
                        try:
                            with open(_part, "a") as _f: _f.write(json.dumps(r) + "\n")
                        except Exception as _e:
                            print(f"{sc}: partial write failed {type(_e).__name__}: {_e} (results unaffected)", flush=True)

            res = _Res(_done_q.values()) if RESUME_Q else []
            _qf = None
            if os.environ.get("QUERY_FILTER"):
                import json as _j
                _qf = set(_j.load(open(os.environ["QUERY_FILTER"])))
            for q in by[sc]:
                if _qf is not None and f"{sc}__{q['object_id']}__{q.get('ann_id')}" not in _qf: continue
                EA_BAN = _BAN_MAP.get(f"{sc}__{q['object_id']}__{q.get('ann_id')}")
                globals()['BAN_INST'] = {int(EA_BAN)} if EA_BAN is not None else set()
                if RESUME_Q and (q["object_id"], q.get("ann_id")) in _done_q: continue
                gt = gtm.get(q["object_id"])
                if gt is None or gt.sum()==0: continue
                if FIXEDQ:                                                 # match training: R1 selects top-8 frames, R2 matches the fixed all-frame qemb -> inst3d answer
                    if inst3d_all is None or qemb_all.shape[0] == 0: continue
                    oid = int(q["object_id"]); nf = len(fidx)
                    # query-perfect oracle: assume the CORRECT object query is picked -> best-IoU instance in the fixed Q (= teacher instance ceiling; no R1/no-match error)
                    ious_qp = (inst3d_all & gt).sum(1) / np.maximum((inst3d_all | gt).sum(1), 1)
                    qp = int(ious_qp.argmax()); pmq = inst3d_all[qp]
                    row = {"oid": q["object_id"], "name": q["object_name"], "ann_id": q.get("ann_id"),
                           "miou_qp": round(float(ious_qp[qp]),3), "biou_qp": round(aabb_iou(coord[pmq], coord[gt]),3)}
                    if not CEILING and MFRAME and MF_ONESHOT:                                     # MF32: ONE forward -- 32 uniform frames' registers, 8 uniform tower images, 32 grounds -> vote
                        if sf_e is None: continue
                        n = inst3d_all.shape[0]
                        f32 = np.linspace(0, S - 1, min(S, MF_NF)).round().astype(int)
                        tower = f32[np.linspace(0, len(f32) - 1, min(len(f32), MF_IMG or len(f32))).round().astype(int)]
                        cot_text = ""
                        if MF_COT and MF_COT_GEN:                                                   # MF+CoT single-forward: reason ONCE over the frame spread, then ground
                            cot_text = _gen_mf_cot(ag, proc, imgs_all[tower], q["description"])
                        votes = np.zeros(n)
                        for idx, cf in _agent_mframe(ag, proc, reg_all[f32], imgs_all[f32], q["description"], sf_e,
                                                     cot_text=cot_text, imgs_tower=imgs_all[tower]):
                            if 0 <= idx < n: votes[idx] += cf
                        if votes.max() > 0:
                            sel = int(votes.argmax()); pm = inst3d_all[sel]
                            row["miou"] = round(float((pm & gt).sum()/max((pm | gt).sum(),1)),3); row["biou"] = round(aabb_iou(coord[pm], coord[gt]),3)
                            if os.environ.get("MF_DUMP_ANSWER"):                                   # closed-loop Tool B (single-forward variant): dump this run's own answer
                                _m2 = masks_all[sel].astype(bool) if masks_all is not None else None
                                _fc = fidx if LOCATE else _mf_fcap
                                np.savez_compressed(f"{OUT}/{sc}__{q['object_id']}__{q.get('ann_id')}__mf.npz",
                                    inst3d=np.packbits(pm.astype(np.uint8)), N=np.int64(pm.shape[0]),
                                    m2=np.packbits(_m2.astype(np.uint8)) if _m2 is not None else np.zeros(0, np.uint8),
                                    m2shape=np.array(_m2.shape) if _m2 is not None else np.zeros(3, int),
                                    fcap=np.asarray(_fc, np.int32))
                        else:
                            row["miou"] = 0.0; row["biou"] = 0.0
                        res.append(row); continue
                    if not CEILING and MFRAME:                                                    # multi-frame sliding window (stride=NF-overlap); overlapping frames vote
                        if sf_e is None: continue
                        n = inst3d_all.shape[0]; stride = max(1, MF_NF - MF_OVERLAP)
                        FR = np.arange(S)                                                           # VIDI_SEL: slide only over Vidi-selected frames (dilate to a full window), else all S
                        if vmap is not None:
                            _rr = vmap.get((str(q["object_id"]), str(q.get("ann_id"))))
                            if _rr:
                                _m = np.zeros(S, bool)
                                for _a, _b in _rr:
                                    _i0, _i1 = int(np.floor(_a * S)), int(np.ceil(_b * S))
                                    _m[max(0, _i0):min(S, max(_i1, _i0 + 1))] = True
                                while 0 < _m.sum() < min(MF_NF, S):
                                    _iv = np.where(_m)[0]
                                    _m[np.maximum(_iv - 1, 0)] = True; _m[np.minimum(_iv + 1, S - 1)] = True
                                if _m.any(): FR = np.where(_m)[0]
                        L = len(FR)
                        starts = list(range(0, max(1, L - MF_NF + 1), stride))
                        if L > MF_NF and starts[-1] != L - MF_NF: starts.append(L - MF_NF)          # cover tail
                        if os.environ.get("PROBE_RL", "0") == "1":                                   # RL exploration probe: K CoT rollouts vs greedy, reward=inst IoU vs GT (NOTE: "0" is truthy in Python -> must compare =="1")
                            K = int(os.environ.get("PROBE_K", "8")); _gw = FR[np.linspace(0, L - 1, min(L, MF_NF)).round().astype(int)]
                            inst_iou = (inst3d_all & gt).sum(1) / np.maximum((inst3d_all | gt).sum(1), 1)   # per-instance reward table

                            def _ground(ct):                                                     # PROBE: single representative window (the CoT-gen spread) -> ~58x faster
                                v = np.zeros(n)
                                for idx, cf in _agent_mframe(ag, proc, reg_all[_gw], imgs_all[_gw], q["description"], sf_e, cot_text=ct):
                                    if 0 <= idx < n: v[idx] += cf
                                return int(v.argmax()) if v.max() > 0 else -1
                            gsel = _ground(_gen_mf_cot(ag, proc, imgs_all[_gw], q["description"]))    # greedy CoT
                            ksels = [_ground(c) for c in _gen_mf_cot(ag, proc, imgs_all[_gw], q["description"], nsamp=K, temp=1.0)]
                            row.update({"probe": True, "greedy_r": float(inst_iou[gsel]) if gsel >= 0 else 0.0,
                                        "k_rs": [float(inst_iou[s]) if s >= 0 else 0.0 for s in ksels], "k_sels": ksels,
                                        "greedy_sel": gsel, "oracle_r": float(inst_iou.max()) if n else 0.0, "n_inst": int(n)})
                            res.append(row); continue
                        votes = np.zeros(n)
                        pt_hit = pt_tot = 0; oid_i = int(q["object_id"])   # point-in-GT-2D-mask accuracy over all GT-visible frames
                        ptg_hit = ptg_tot = 0                              # GATED by POINT-HEAD's OWN null: point scored only where it predicts PRESENT (null<0.5)
                        pres_tp = pres_fn = pres_fp = pres_tn = 0          # point-head null quality vs GT visibility (Arm B: does the point head know when to abstain?)
                        box_hit = box_tot = 0                              # Arm C: per-frame generated box@.5 vs GT box (on GT-visible frames)
                        abx = {}                                           # Arm C: collect generated per-frame boxes {frame: (x0,y0,x1,y1)} for s4vy_prompt box dump (final box@.5)
                        apt = {}                                           # point heads: collect predicted per-frame points {frame: (x,y)} for s4vy_prompt point dump (final box@.5)
                        qpf = {}                                           # per-frame qemb <GROUND> matched instance idx (-1=null) -> for qemb-present gating of box/point (deployable, no GT leak)
                        bxwin = {}                                         # per-frame ALL windows' box (or None=absent) -> sliding-window VOTE (median box + majority presence), like qemb voting
                        p2wins = []                                        # Phase-2: (win, pass1_boxes, qemb_fres) per window, corrected after the vote settles
                        ptwin = {}                                         # per-frame ALL windows' (x, y, P(present)=1-sigmoid(null)) -> vote (median point + avg confidence)
                        cot_text = ""
                        if MF_COT and MF_COT_GEN:                                                   # generate CoT ONCE per query (reused across windows), from a spread of frames
                            _gw = FR[np.linspace(0, L - 1, min(L, MF_NF)).round().astype(int)]
                            cot_text = _gen_mf_cot(ag, proc, imgs_all[_gw], q["description"])
                        if MF_TREE and os.environ.get("PROBE_TREE") == "1":
                            globals()["TREE_TRACE"] = []
                        if MF_TREE:                                                             # coarse-to-fine instead of uniform sliding
                            _wins, _bxall = mf_tree_windows(FR, L, lambda _ck: (
                                _gen_mf_box_batch(ag, proc, [reg_all[w] for w in _ck],
                                                  [imgs_all[w] for w in _ck], q["description"], cot_text=cot_text)
                                if len(_ck) > 1 else
                                [_gen_mf_box(ag, proc, reg_all[_ck[0]], imgs_all[_ck[0]], q["description"], cot_text=cot_text)]))
                        else:
                            _wins = [FR[s0:s0 + MF_NF] for s0 in starts]
                        if not MF_TREE: _bxall = None
                        if (not MF_TREE) and MF_BATCH > 1 and BOX_OUT and idm is not None:                            # batch the box generation across windows
                            _bxall = []
                            for _i in range(0, len(_wins), MF_BATCH):
                                _ck = _wins[_i:_i + MF_BATCH]
                                _bxall += _gen_mf_box_batch(ag, proc, [reg_all[w] for w in _ck],
                                                            [imgs_all[w] for w in _ck], q["description"], cot_text=cot_text)
                        for _wi, win in enumerate(_wins):
                            ag._cur_patch = patch_all[win] if patch_all is not None else None       # Arm B: patch grid for this window's <POINT> visual pool
                            fres = list(_agent_mframe(ag, proc, reg_all[win], imgs_all[win], q["description"], sf_e, cot_text=cot_text))   # per-frame (idx, conf)
                            for idx, conf in fres:
                                if 0 <= idx < n: votes[idx] += conf
                            for j, f in enumerate(win):                                            # per-frame qemb <GROUND> matched instance (for qemb-present gating)
                                if j < len(fres): qpf.setdefault(int(f), int(fres[j][0]))
                            lp = getattr(ag, "_last_points", None); pn = getattr(ag, "_last_point_null", None)
                            if lp is not None and idm is not None:                                 # point metric: is the predicted point inside the target's GT 2D mask?
                                for j, f in enumerate(win):
                                    gm = (idm[int(f)] == oid_i); vis = gm.sum() > 10
                                    p_present = (pn is None or (j < len(pn) and pn[j] < 0.5))      # point head predicts PRESENT (its own null gate)
                                    if vis and p_present: pres_tp += 1
                                    elif vis and not p_present: pres_fn += 1
                                    elif (not vis) and p_present: pres_fp += 1
                                    else: pres_tn += 1
                                    _pp = float(1.0 - pn[j]) if (pn is not None and j < len(pn)) else 1.0        # P(present) = 1 - sigmoid(null_logit), the point head's direct confidence
                                    ptwin.setdefault(int(f), []).append((float(lp[j, 0]), float(lp[j, 1]), _pp))  # sliding-window vote: point + confidence
                                    if p_present: apt.setdefault(int(f), (float(lp[j, 0]), float(lp[j, 1])))   # (legacy first-window present-point, kept for compat)
                                    if vis:
                                        px = min(max(int(lp[j, 0] * gm.shape[1]), 0), gm.shape[1] - 1)
                                        py = min(max(int(lp[j, 1] * gm.shape[0]), 0), gm.shape[0] - 1)
                                        hit = int(gm[py, px]); pt_tot += 1; pt_hit += hit
                                        if p_present: ptg_tot += 1; ptg_hit += hit                # GATED: point scored only where point-head says present
                            if BOX_OUT and idm is not None:                                        # Arm C: generate per-frame box list + box@.5 vs GT (presence via 'absent')
                                bxs = (_bxall[_wi] if _bxall is not None else
                                       _gen_mf_box(ag, proc, reg_all[win], imgs_all[win], q["description"], cot_text=cot_text))
                                if P2_CORRECT: p2wins.append((np.asarray(win), list(bxs), list(fres)))   # keep Pass-1 for the second pass
                                for j, f in enumerate(win):
                                    gm = (idm[int(f)] == oid_i); vis = gm.sum() > 10; b_present = bxs[j] is not None
                                    if vis and b_present: pres_tp += 1
                                    elif vis and not b_present: pres_fn += 1
                                    elif (not vis) and b_present: pres_fp += 1
                                    else: pres_tn += 1
                                    bxwin.setdefault(int(f), []).append(bxs[j])                   # sliding-window vote: record this window's box (or None) for the frame
                                    if bxs[j] is not None: abx.setdefault(int(f), bxs[j])          # (legacy first-window box, kept for box_acc metric)
                                    if vis and bxs[j] is not None:
                                        ys, xs = np.where(gm)
                                        gb = (xs.min()/gm.shape[1], ys.min()/gm.shape[0], (xs.max()+1)/gm.shape[1], (ys.max()+1)/gm.shape[0])
                                        box_tot += 1; box_hit += int(_iou_box(bxs[j], gb) >= 0.5)
                        if P2_CORRECT and p2wins:                                                  # ---- Phase-2 second pass ----
                            # the "target" for [REVIEW] is the query-level qemb-VOTED instance: exactly what the
                            # model predicts anyway, so no GT enters inference. Rounds>1 feed the corrected boxes
                            # back in as the new Pass-1.
                            _tgt = int(votes.argmax()) if votes.max() > 0 else -1
                            P2GEN = p2wins
                            if P2_RESAMPLE:
                                P2GEN, _nc, _nt = _p2_resample(p2wins, _tgt, len(imgs_all), MF_NF, P2_MIX, P2_TRIG)
                                row["p2_contested"] = int(_nc); row["p2_frames"] = int(_nt)
                                row["p2_genwins"] = len(P2GEN); row["p2_p1wins"] = len(p2wins)
                            _cur = [list(b) for _, b, _ in P2GEN]
                            _p1s0 = [_p2_boxstr(_cur[w], MF_NF) for w in range(len(P2GEN))]            # Pass-1 boxes: FIXED, this is Turn-1
                            _revs0 = [_p2_review(_cur[w], [x[0] for x in P2GEN[w][2]],
                                                 [x[1] for x in P2GEN[w][2]], _tgt, MF_NF) for w in range(len(P2GEN))]
                            _curq = [list(x[2]) for x in P2GEN]                                    # per-window qemb (idx, conf), updated by Turn-2's own readout
                            _hist = [""] * len(P2GEN); _kinds = []
                            p2_rounds_state = []
                            def _snap():                                                               # per-frame box vote + qemb set for the CURRENT state
                                bw = {}; qp_ = {}; vt = np.zeros(n); _fix = set()
                                if P2_ACC:
                                    for w, (win_w, bxs_w, fres_w) in enumerate(p2wins):
                                        for j, f in enumerate(win_w):
                                            f = int(f)
                                            bw.setdefault(f, []).append(bxs_w[j] if j < len(bxs_w) else None)
                                            if j < len(fres_w):
                                                _ix, _cf = fres_w[j]
                                                if 0 <= _ix < n: vt[_ix] += _cf
                                                qp_.setdefault(f, int(_ix))
                                for w, (win_w, _, _) in enumerate(P2GEN):
                                    for j, f in enumerate(win_w):
                                        f = int(f)
                                        bw.setdefault(f, []).append(_cur[w][j] if j < len(_cur[w]) else None)
                                        if j < len(_curq[w]):
                                            _ix, _cf = _curq[w][j]
                                            if 0 <= _ix < n: vt[_ix] += _cf
                                            (qp_.__setitem__(f, int(_ix)) if P2_ACC else qp_.setdefault(f, int(_ix))); _fix.add(f)
                                if P2_RESAMPLE and not P2_ACC:
                                    for w, (win_w, bxs_w, fres_w) in enumerate(p2wins):
                                        for j, f in enumerate(win_w):
                                            f = int(f)
                                            if f in _fix: continue
                                            bw.setdefault(f, []).append(bxs_w[j] if j < len(bxs_w) else None)
                                            if j < len(fres_w):
                                                _ix, _cf = fres_w[j]
                                                if 0 <= _ix < n: vt[_ix] += _cf
                                                qp_.setdefault(f, int(_ix))
                                return bw, qp_, vt
                            def _snap0():                                                              # T1 = state BEFORE any correction
                                bw = {}; qp_ = {}; vt = np.zeros(n)
                                for w, (win_w, _, fres_w) in enumerate(p2wins):
                                    for j, f in enumerate(win_w):
                                        bw.setdefault(int(f), []).append(p2wins[w][1][j] if j < len(p2wins[w][1]) else None)
                                        if j < len(fres_w):
                                            _ix, _cf = fres_w[j]
                                            if 0 <= _ix < n: vt[_ix] += _cf
                                            qp_.setdefault(int(f), int(_ix))
                                return bw, qp_, vt
                            p2_rounds_state.append(_snap0())
                            _W = len(P2GEN)
                            _alive = [True] * _W          # P2_DONE_STOP: <done/> means "THIS round stands, run no
                            _used = [0] * _W              # further round for this window" -- the gate ends the loop,
                            _kinds = ["done"] * _W        # it does NOT suppress the round that emitted it.
                            for _rd in range(max(1, P2_ROUNDS)):
                                if P2_FREE and P2_RESAMPLE and _rd > 0:
                                    _fx = {}
                                    for _w2, (_wf2, _, _) in enumerate(P2GEN):
                                        for _j2, _f2 in enumerate(_wf2):
                                            _fx[int(_f2)] = (_cur[_w2][_j2] if _j2 < len(_cur[_w2]) else None,
                                                             _curq[_w2][_j2] if _j2 < len(_curq[_w2]) else (-1, 0.0))
                                    _p1upd = []
                                    for (_wf2, _bx2, _fr2) in p2wins:
                                        _b = [ _fx[int(f)][0] if int(f) in _fx else (_bx2[j] if j < len(_bx2) else None)
                                               for j, f in enumerate(_wf2) ]
                                        _r = [ _fx[int(f)][1] if int(f) in _fx else (_fr2[j] if j < len(_fr2) else (-1, 0.0))
                                               for j, f in enumerate(_wf2) ]
                                        _p1upd.append((_wf2, _b, _r))
                                    P2GEN, _nc2, _nt2 = _p2_resample(_p1upd, _tgt, len(imgs_all), MF_NF, P2_MIX, P2_TRIG)
                                    row[f"p2_contested_r{_rd}"] = int(_nc2); row[f"p2_genwins_r{_rd}"] = len(P2GEN)
                                    _W = len(P2GEN)
                                    if _W == 0: break
                                    _cur = [list(b) for _, b, _ in P2GEN]
                                    _curq = [list(x[2]) for x in P2GEN]
                                    _p1s0 = [_p2_boxstr(_cur[w], MF_NF) for w in range(_W)]
                                    _revs0 = [_p2_review(_cur[w], [x[0] for x in P2GEN[w][2]],
                                                         [x[1] for x in P2GEN[w][2]], _tgt, MF_NF) for w in range(_W)]
                                    _hist = [""] * _W; _alive = [True] * _W
                                    _used = _used + [0] * max(0, _W - len(_used)); _kinds = ["done"] * _W
                                _ids = [w for w in range(_W) if _alive[w]] if P2_DONE_STOP else list(range(_W))
                                if not _ids:
                                    if not p2_rounds_state[1:]: p2_rounds_state.append(p2_rounds_state[0])
                                    break
                                _bylen2 = {}
                                for _w in _ids: _bylen2.setdefault(len(P2GEN[_w][0]), []).append(_w)
                                _outmap = {}
                                for _grp in _bylen2.values():
                                    _og = _gen_mf_turn2(ag, proc, [reg_all[P2GEN[w][0]] for w in _grp],
                                                        [imgs_all[P2GEN[w][0]] for w in _grp],
                                                        q["description"], cot_text,
                                                        [_p1s0[w] for w in _grp], [_revs0[w] for w in _grp],
                                                        hists=[_hist[w] for w in _grp], e_norm=sf_e)
                                    for _w, _r in zip(_grp, _og): _outmap[_w] = _r
                                _out = [_outmap[w] for w in _ids]
                                for _j, w in enumerate(_ids):
                                    _bb, _k, _txt, _qm = _out[_j]
                                    _used[w] += 1; _kinds[w] = _k
                                    if _bb is not None: _cur[w] = _bb                                  # corrected boxes
                                    if _qm is not None: _curq[w] = _qm                                 # corrected qemb readout (Turn-2 re-emits it)
                                    if P2_DONE_STOP and _k == "done": _alive[w] = False
                                    if _rd + 1 < max(1, P2_ROUNDS):                                    # carry this round's OWN output + the next [REVIEW] forward
                                        _hist[w] = (_hist[w] + " " + _txt.strip() + " " +
                                                    _p2_review(_cur[w], [x[0] for x in _curq[w]],
                                                               [x[1] for x in _curq[w]], _tgt, MF_NF)).strip()
                                p2_rounds_state.append(_snap())                                        # snapshot AFTER EACH round -> T2, T3, ...
                            row["p2_rounds_ran"] = int(_rd + 1) if _used else 0
                            row["p2_used_mean"] = round(float(np.mean(_used)), 3) if _used else 0.0
                            row["p2_used_max"] = int(max(_used)) if _used else 0; row["p2_used_hist"] = _used
                            bxwin, qpf, votes = p2_rounds_state[-1]                                 # live state = last round
                            row["p2_done"] = int(sum(1 for k in _kinds if k == "done"))
                            row["p2_reflect"] = int(sum(1 for k in _kinds if k == "reflect"))
                            row["p2_rounds"] = int(max(1, P2_ROUNDS))
                            # qemb-path result PER ROUND. `row["biou"]/["miou"]` below are written once, from the
                            # FINAL votes, so a ROUNDS=2 run only ever recorded T3 and T1's qemb result existed
                            # nowhere -- the box dumps were fixed to snapshot every round but this second record
                            # path was missed.
                            for _ri, _st in enumerate(p2_rounds_state):
                                _vt = _st[2]
                                if _vt is None or _vt.max() <= 0:
                                    row[f"biou_T{_ri+1}"] = 0.0; row[f"miou_T{_ri+1}"] = 0.0; continue
                                _sl = int(_vt.argmax()); _pm = inst3d_all[_sl]
                                row[f"sel_T{_ri+1}"] = _sl
                                row[f"miou_T{_ri+1}"] = round(float((_pm & gt).sum() / max((_pm | gt).sum(), 1)), 3)
                                row[f"biou_T{_ri+1}"] = round(aabb_iou(coord[_pm], coord[gt]), 3)
                        if os.environ.get("PROBE_SPACE") == "1":
                          try:
                              _iq = ious_qp
                              _bq = np.array([aabb_iou(coord[inst3d_all[i]], coord[gt]) for i in range(len(_iq))])
                              _best = int(np.argmax(_iq))
                              _pr = {"scene": sc, "oid": q["object_id"], "ann": q.get("ann_id"), "n_inst": int(len(_iq)),
                                     "best_inst": _best, "best_miou": round(float(_iq[_best]), 4),
                                     "best_biou": round(float(_bq[_best]), 4)}
                              _ord = np.argsort(-votes)
                              _pr["qemb_rank_of_best"] = int(np.where(_ord == _best)[0][0]) if votes.max() > 0 else -1
                              _pr["qemb_topk_miou"] = [round(float(_iq[i]), 4) for i in _ord[:20]]
                              _pr["qemb_topk_biou"] = [round(float(_bq[i]), 4) for i in _ord[:20]]
                              _pr["qemb_topk_vote"] = [round(float(votes[i]), 4) for i in _ord[:20]]
                              if masks_all is not None and bxwin:
                                  _bv = np.zeros(len(_iq))
                                  _fr = sorted(bxwin)
                                  _loc2 = {int(f): j for j, f in enumerate(_mf_fcap)} if _mf_fcap is not None else {}
                                  for _f in _fr:
                                      _bs = [b for b in bxwin[_f] if b is not None]
                                      if not _bs or int(_f) not in _loc2: continue
                                      _med = [float(np.median([b[k] for b in _bs])) for k in range(4)]
                                      _j = _loc2[int(_f)]
                                      for _i in range(len(_iq)):
                                          _mk = masks_all[_i, _j]
                                          if _mk.sum() == 0: continue
                                          _ys, _xs = np.where(_mk)
                                          _h2, _w2 = _mk.shape
                                          _ib = [_xs.min()/_w2, _ys.min()/_h2, (_xs.max()+1)/_w2, (_ys.max()+1)/_h2]
                                          _bv[_i] += _iou_box(_med, _ib)
                                  _bo = np.argsort(-_bv)
                                  _pr["box_rank_of_best"] = int(np.where(_bo == _best)[0][0]) if _bv.max() > 0 else -1
                                  _pr["box_topk_miou"] = [round(float(_iq[i]), 4) for i in _bo[:20]]
                                  _pr["box_topk_biou"] = [round(float(_bq[i]), 4) for i in _bo[:20]]
                                  _pr["union_top5_biou"] = round(float(max(_bq[list(_ord[:5]) + list(_bo[:5])])), 4)
                                  _pr["union_top5_miou"] = round(float(max(_iq[list(_ord[:5]) + list(_bo[:5])])), 4)
                              if os.environ.get("PROBE_TREE") == "1" and isinstance(globals().get("TREE_TRACE"), list):
                                _vis = None
                                if masks_all is not None and _best < masks_all.shape[0]:
                                    _am = masks_all[_best].reshape(masks_all.shape[1], -1).sum(1)
                                    _vis = {int(_mf_fcap[_j]): bool(_am[_j] > 0) for _j in range(len(_am))}
                                _nodes = []
                                for _nd in globals()["TREE_TRACE"]:
                                    _fr2 = _nd["frames"]; _fi = set(_nd["fired"])
                                    _tp = _fp = _fn = _tn = 0
                                    if _vis is not None:
                                        for _j, _f2 in enumerate(_fr2):
                                            _v = _vis.get(int(_f2))
                                            if _v is None: continue
                                            if _j in _fi and _v: _tp += 1
                                            elif _j in _fi and not _v: _fp += 1
                                            elif _j not in _fi and _v: _fn += 1
                                            else: _tn += 1
                                    _cov = set()
                                    for _c0, _c1 in _nd["children"]: _cov |= set(range(_c0, _c1 + 1))
                                    _lost = 0
                                    if _vis is not None:
                                        for _f3 in range(_nd["lo"], _nd["hi"] + 1):
                                            if _f3 not in _cov and _vis.get(int(_f3)): _lost += 1
                                    _nodes.append({**_nd, "tp": _tp, "fp": _fp, "fn": _fn, "tn": _tn, "lost_vis": _lost})
                                _pr["tree_nodes"] = _nodes
                                _pr["tree_n_win"] = len(_wins) if "_wins" in dir() else None
                                globals()["TREE_TRACE"] = None
                              _pd = os.environ.get("PROBE_DIR", f"{OUT}_probe"); os.makedirs(_pd, exist_ok=True)
                              json.dump(_pr, open(f"{_pd}/{sc}__{q['object_id']}__{q.get('ann_id')}.json", "w"))
                          except Exception as _pe:
                            print(f"[probe] {sc} oid{q['object_id']} FAIL {type(_pe).__name__}: {_pe}", flush=True)
                        if votes.max() > 0:
                            sel = int(votes.argmax()); pm = inst3d_all[sel]
                            row["sel"] = sel
                            _ki = _tl.get("keep_idx")
                            row["sel_q"] = int(_ki[sel]) if _ki is not None and sel < len(_ki) else -1
                            row["miou"] = round(float((pm & gt).sum()/max((pm | gt).sum(),1)),3); row["biou"] = round(aabb_iou(coord[pm], coord[gt]),3)
                            if os.environ.get("MF_DUMP_ANSWER"):                                   # closed-loop Tool B: dump THIS run's own answer artifacts
                                _m2 = masks_all[sel].astype(bool) if masks_all is not None else None
                                _fc = fidx if LOCATE else _mf_fcap
                                np.savez_compressed(f"{OUT}/{sc}__{q['object_id']}__{q.get('ann_id')}__mf.npz",
                                    inst3d=np.packbits(pm.astype(np.uint8)), N=np.int64(pm.shape[0]),
                                    m2=np.packbits(_m2.astype(np.uint8)) if _m2 is not None else np.zeros(0, np.uint8),
                                    m2shape=np.array(_m2.shape) if _m2 is not None else np.zeros(3, int),
                                    fcap=np.asarray(_fc, np.int32))
                        else:
                            sel = -1
                            row["sel"] = -1
                            row["miou"] = 0.0; row["biou"] = 0.0
                        if pt_tot: row["pt_acc"] = round(pt_hit / pt_tot, 3); row["pt_n"] = pt_tot   # dual-output: point-in-GT-mask accuracy (ungated, on GT-visible frames)
                        if ptg_tot: row["ptg_acc"] = round(ptg_hit / ptg_tot, 3); row["ptg_n"] = ptg_tot   # GATED by ground-null: point-in-mask only where ground says present (share qemb presence)
                        if (pres_tp + pres_fn + pres_fp + pres_tn):                                        # presence gate quality vs GT visibility (Arm B point-null / Arm C box-absent)
                            row["pres_recall"] = round(pres_tp / max(pres_tp + pres_fn, 1), 3); row["pres_prec"] = round(pres_tp / max(pres_tp + pres_fp, 1), 3)
                        if box_tot: row["box_acc"] = round(box_hit / box_tot, 3); row["box_n"] = box_tot   # Arm C: per-frame generated box@.5 vs GT box
                        _qp = sorted(f for f, mi in qpf.items() if mi == sel)                             # qemb-present frames (matched the voted target instance) -- deployable gate, no GT
                        if BOX_OUT and os.environ.get("MF_DUMP_BOX") and bxwin:                            # sliding-window VOTE: per frame -> median box + majority-presence fraction
                            # ONE run dumps EVERY round: T1 (before any correction), T2 (after 1), T3 (after 2).
                            # The pre-correction boxes are already in memory, so re-running the whole eval just to
                            # get T1 -- as was done before -- was pure waste.
                            _states = locals().get("p2_rounds_state") or [(bxwin, _qp if isinstance(_qp, dict) else {}, None)]
                            for _ri, _st in enumerate(_states):
                                _bw = _st[0]; _qpr = _st[1]
                                vb = []
                                for f in sorted(_bw):
                                    bs = [b for b in _bw[f] if b is not None]
                                    if not bs: continue
                                    pf = len(bs) / len(_bw[f])                                            # fraction of windows voting present
                                    med = [float(np.median([b[k] for b in bs])) for k in range(4)]         # median box across windows
                                    vb.append([int(f)] + [round(c, 4) for c in med] + [round(pf, 3)])      # [frame, x0,y0,x1,y1, presence_frac]
                                _qpl = sorted(f for f, ix in _qpr.items() if ix >= 0) if isinstance(_qpr, dict) else _qp
                                _od = f"{OUT}_T{_ri+1}" if len(_states) > 1 else OUT
                                os.makedirs(_od, exist_ok=True)
                                json.dump({"scene": sc, "oid": q["object_id"], "ann": q.get("ann_id"), "qp": _qpl, "boxes": vb},
                                          open(f"{_od}/{sc}__{q['object_id']}__{q.get('ann_id')}__box.json", "w"))
                                # Dump the qemb path's ANSWER MASK here, where the prediction is made. `sel` is a
                                # row index into the instance list of THIS forward pass and means nothing in another
                                # process -- five attempts to rebuild the mask from it downstream all produced the
                                # wrong object (the frame cap depends on runtime memory, not on code). The box
                                # path never had this problem because coordinates are absolute.
                                if os.environ.get("MF_DUMP_QMASK") == "1" and masks_all is not None:
                                    _sv = _st[2] if len(_st) > 2 else None
                                    _si = int(np.argmax(_sv)) if _sv is not None and np.size(_sv) and np.max(_sv) > 0 else int(row.get("sel", -1))
                                    if 0 <= _si < masks_all.shape[0]:
                                        _mm = masks_all[_si].astype(bool)                                  # (len(fcap),h,w)
                                        np.savez_compressed(f"{_od}/{sc}__{q['object_id']}__{q.get('ann_id')}__qmask.npz",
                                                            mask=np.packbits(_mm.reshape(_mm.shape[0], -1), axis=1),
                                                            shape=np.array(_mm.shape), frames=np.array(_mf_fcap, int), sel=_si)
                        if os.environ.get("MF_DUMP_POINT") and ptwin:                                     # sliding-window VOTE: per frame -> median point + avg P(present) confidence
                            vp = []
                            for f in sorted(ptwin):
                                ps = ptwin[f]
                                x = float(np.median([p[0] for p in ps])); y = float(np.median([p[1] for p in ps]))
                                conf = float(np.mean([p[2] for p in ps]))                                  # avg P(present)=1-sigmoid(null) across windows = confidence
                                vp.append([int(f), round(x, 4), round(y, 4), round(conf, 3)])              # [frame, x, y, confidence]
                            json.dump({"scene": sc, "oid": q["object_id"], "ann": q.get("ann_id"), "qp": _qp, "points": vp},
                                      open(f"{OUT}/{sc}__{q['object_id']}__{q.get('ann_id')}__point.json", "w"))
                        res.append(row); continue
                    if not CEILING and SFRAME:                                                    # single-frame grounder: ground each frame -> vote (conf-weighted) over non-null -> instance
                        if sf_e is None: continue
                        n = inst3d_all.shape[0]
                        votes = np.zeros(n); nfr = 0
                        for f in sf_frames:
                            idx, conf = _agent_sframe(ag, proc, reg_all[f], imgs_all[f], q["description"], sf_e)
                            if 0 <= idx < n: votes[idx] += conf; nfr += 1
                        if votes.max() > 0:                                                        # realistic: >=1 frame grounded a query
                            sel = int(votes.argmax()); pm = inst3d_all[sel]
                            row["miou"] = round(float((pm & gt).sum()/max((pm | gt).sum(),1)),3); row["biou"] = round(aabb_iou(coord[pm], coord[gt]),3)
                        else:
                            row["miou"] = 0.0; row["biou"] = 0.0                                   # all frames -> null -> "not found"
                        vis = (idm[fidx] == oid).reshape(len(fidx), -1).sum(1)                     # oracle: vote only over GT-visible frames (SF-matcher ceiling)
                        _vord = np.argsort(-vis); _vord = _vord[vis[_vord] > 20][:SF_CAP]          # GT-visible frames (area>20), top SF_CAP by area
                        vfr = fidx[_vord]
                        if len(vfr) > 0:
                            vo = np.zeros(n)
                            for f in vfr:
                                idx, conf = _agent_sframe(ag, proc, reg_all[f], imgs_all[f], q["description"], sf_e)
                                if 0 <= idx < n: vo[idx] += conf
                            if vo.max() > 0:
                                pmo = inst3d_all[int(vo.argmax())]
                                row["miou_orc"] = round(float((pmo & gt).sum()/max((pmo | gt).sum(),1)),3); row["biou_orc"] = round(aabb_iou(coord[pmo], coord[gt]),3)
                            else:
                                row["miou_orc"] = 0.0; row["biou_orc"] = 0.0
                        res.append(row); continue
                    if not CEILING:
                        scores = _agent_round_r1(ag, proc, reg_r1, q["description"], nf, r1_imgs)   # R1 temporal (same head as training)
                        if LOCATE:                                                                  # DUMP per-frame: R1 score, LA-best-instance, box-IoU confidence + each instance's 3D IoU -> offline sweep of vote/filter/K
                            n = inst3d_all.shape[0]; h, w = masks_all.shape[2], masks_all.shape[3]
                            row["inst_b"] = [round(float(aabb_iou(coord[inst3d_all[i]], coord[gt])),3) for i in range(n)]   # each instance's box IoU vs GT
                            row["inst_m"] = [round(float((inst3d_all[i]&gt).sum()/max((inst3d_all[i]|gt).sum(),1)),3) for i in range(n)]
                            def _fr(order):
                                recs = []
                                for o in order:
                                    box = _la_box(la_model, la_proc, imgs_all[int(fidx[o])], q["description"])
                                    bi, bc = -1, 0.0
                                    if box is not None:
                                        for i in range(n):
                                            mk = masks_all[i, o]
                                            if not mk.any(): continue
                                            ys, xs = np.where(mk); v = _iou_box(box, [xs.min()/w, ys.min()/h, (xs.max()+1)/w, (ys.max()+1)/h])
                                            if v > bc: bc, bi = v, i
                                    recs.append([round(float(scores[o]),4), int(bi), round(float(bc),3)])   # [R1_score, best_inst, box_IoU_conf]
                                return recs
                            Kd = min(LA_K, nf)
                            row["fr_real"] = _fr(np.argsort(-scores)[:Kd])                           # R1 top-K frames
                            vis = (idm[fidx] == oid).reshape(len(fidx), -1).sum(1)
                            row["fr_orc"] = _fr(np.argsort(-vis)[:Kd]) if (vis > 20).any() else []   # GT-visible top-K frames (oracle)
                        else:                                                                       # VLM <GROUND> match (top-8 vote)
                            sel = (_agent_ground_r2_fixedq_inst if INST_APPEND else _agent_ground_r2_fixedq)(ag, proc, reg_r1, nf, imgs_all[np.sort(fidx[np.argsort(-scores)[:NF_VLM]])], q["description"], qemb_all)
                            pm = inst3d_all[sel]; inter = (pm & gt).sum(); union = (pm | gt).sum()
                            row["miou"] = round(float(inter/max(union,1)),3); row["biou"] = round(aabb_iou(coord[pm], coord[gt]),3)
                            vis = (idm[fidx] == oid).reshape(len(fidx), -1).sum(1)
                            if (vis > 20).any():
                                sel_o = (_agent_ground_r2_fixedq_inst if INST_APPEND else _agent_ground_r2_fixedq)(ag, proc, reg_r1, nf, imgs_all[np.sort(fidx[np.argsort(-vis)[:NF_VLM]])], q["description"], qemb_all)
                                pmo = inst3d_all[sel_o]; io = (pmo & gt).sum(); uo = (pmo | gt).sum()
                                row["miou_orc"] = round(float(io/max(uo,1)),3); row["biou_orc"] = round(aabb_iou(coord[pmo], coord[gt]),3)
                    res.append(row); continue
                if LOC:                                                    # SAM-hint: R1 top-1 key frame (realistic) + oracle best-visible frame; generate coord -> hit test
                    oid = int(q["object_id"])
                    vis = (idm == oid).reshape(idm.shape[0], -1).sum(1)     # per-frame target-mask area (idmap res)
                    if (vis > 20).sum() == 0: continue                     # object not visible in any frame -> undefined
                    scores = _agent_round_r1(ag, proc, reg_r1, q["description"], len(fidx), r1_imgs)
                    f_real = int(fidx[int(np.argmax(scores))])             # R1 top-1 frame (realistic pipeline)
                    f_orc = int(np.argmax(vis))                            # best GT-visible frame (isolates R2 pointing quality)
                    cr = _agent_ground_loc(ag, proc, imgs_all[f_real], q["description"])
                    co = _agent_ground_loc(ag, proc, imgs_all[f_orc], q["description"])
                    res.append({"oid": q["object_id"], "name": q["object_name"],
                                "hit": _loc_hit(cr, idm[f_real] == oid),   # realistic: R1-picked frame hit (0 if that frame lacks the object)
                                "hit_orc": _loc_hit(co, idm[f_orc] == oid),  # oracle-frame upper bound
                                "r1_ok": int(vis[f_real] > 20),            # did R1 even land on a frame containing the target?
                                "coord": cr, "coord_orc": co})
                    continue
                if TEMPORAL:                                               # TEMPORAL-only: R1 frame-selection quality vs GT-visible
                    r1o = _agent_round_r1(ag, proc, reg_r1, q["description"], len(fidx), r1_imgs)
                    sc_t = r1o[0] if (SEL_HEAD or ACTION) else r1o          # per-fidx-frame sigmoid temporal scores
                    vis = (idm == int(q["object_id"])).reshape(idm.shape[0], -1).sum(1)  # object_id is a STR; idmap is int16
                    gtv = (vis[fidx] > 20).astype(np.float32)              # GT-visible over the register frames
                    if gtv.sum() == 0: continue                            # object not in any sampled frame -> undefined
                    sel = sc_t > 0.5; inter = float((sel & (gtv > 0)).sum())
                    res.append({"oid": q["object_id"], "name": q["object_name"],
                                "tap": round(_frame_ap(gtv, sc_t), 4),
                                "tiou": round(inter / max(float((sel | (gtv > 0)).sum()), 1), 4),
                                "tprec": round(inter / max(float(sel.sum()), 1), 4),
                                "trec": round(inter / max(float(gtv.sum()), 1), 4),
                                "npos": int(gtv.sum()), "nframe": int(len(gtv)),
                                "scores": [round(float(s), 3) for s in sc_t], "gtv": [int(x) for x in gtv]})
                    continue
                if ORACLE:                                                 # DIAG: bypass temporal, use GT-visible frames
                    vis = (idm == int(q["object_id"])).reshape(idm.shape[0], -1).sum(1)  # object_id is a STR; idmap is int16
                    fsel = np.where(vis > 20)[0]
                    if len(fsel) == 0: fsel = np.array([int(vis.argmax())])
                elif SEL_HEAD:                                             # E4: learned count -> top-K (vs hard threshold)
                    scores, K = _agent_round_r1(ag, proc, reg_r1, q["description"], len(fidx), r1_imgs)
                    fsel = fidx[np.argsort(-scores)[:K]]
                else:                                                      # R1 temporal: frames containing the target
                    r1out = _agent_round_r1(ag, proc, reg_r1, q["description"], len(fidx), r1_imgs)
                    scores, p_refine = (r1out if ACTION else (r1out, 1.0))  # E2: action head's refine prob
                    fsel = fidx[scores > 0.5]
                    if len(fsel) < 8: fsel = fidx[np.argsort(-scores)[:16]]   # fallback: not enough confident frames
                if len(fsel) > 32: fsel = fsel[np.linspace(0, len(fsel)-1, 32).round().astype(int)]  # cap 32, uniform (per spec)
                fsel = np.sort(fsel)
                pm = None
                for r in range(2, NROUND + 1):                             # R2..N: segmenter re-inference on selected frames
                    reg_s, masks_s, _, qemb_s = segmenter_forward(m9, decode, imgs_all[fsel])
                    if COT:                                                # E5: generate <think> CoT then read <GROUND>
                        gr = _agent_ground_r2_cot(ag, proc, imgs_all[fsel], reg_s, q["description"], qemb_s, masks_s, sample, fsel)
                    elif GROUP == "g5":                                     # G5: ground in a context that INCLUDES the R1 turn (history)
                        gr = _agent_ground_r2_g5(ag, proc, r1_imgs, reg_r1, len(fidx),
                                                 imgs_all[fsel], reg_s, q["description"], qemb_s, masks_s, sample, fsel)
                    else:
                        gr = _agent_ground_r2(ag, proc, imgs_all[fsel], reg_s, q["description"], qemb_s, masks_s, sample, fsel,
                                              qemb_all=qemb_all, inst3d_all=inst3d_all)
                    pm = gr["inst3d"]
                    if ACTION and p_refine <= 0.5:                         # E2: action head says selection is good -> stop, don't refine
                        break
                    if r < NROUND and gr.get("sel") is not None:           # content-based refine: keep frames where matched inst appears
                        cov = masks_s[gr["sel"]].reshape(len(fsel), -1).sum(1)
                        nxt = fsel[cov > 0]
                        if len(nxt) >= 4: fsel = np.sort(nxt[np.linspace(0, len(nxt)-1, min(32, len(nxt))).round().astype(int)])
                if pm is None: continue
                inter=(pm&gt).sum(); union=(pm|gt).sum()
                row = {"oid":q["object_id"],"name":q["object_name"],
                       "miou":round(float(inter/max(union,1)),3),"biou":round(aabb_iou(coord[pm],coord[gt]),3)}
                if DIAG and gr.get("all") is not None:                     # head pick vs oracle-best instance
                    allm = gr["all"]; ious = (allm & gt).sum(1) / np.maximum((allm | gt).sum(1), 1)
                    row["best_iou"] = round(float(ious.max()), 3); row["head_iou"] = round(float(ious[gr["sel"]]), 3); row["ninst"] = int(allm.shape[0])
                res.append(row)
            json.dump(list(res), open(of,"w")); print(f"{sc}: {len(res)} q", flush=True)
            if RESUME_Q and os.path.exists(_part): os.remove(_part)
        except Exception as e:
            print(f"{sc}: FAIL {type(e).__name__}: {e}", flush=True)
            # A caught exception keeps its traceback -> the frames -> every CUDA tensor they hold. Without this the
            # first OOM poisons the shard: scene0208_00 died on a 41 GiB request and all 14 scenes after it then
            # OOMed on ~300 MiB, so ONE bad query cost 15 scenes. Drop the frames, then hand the memory back.
            import gc, sys as _sys
            _sys.exc_info(); e.__traceback__ = None; del e
            gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def _agent_round_r1(ag, proc, reg_np, desc, nf, imgs):
    """R1: uniform-8 images + all-frame register (nf frames) -> per-frame temporal scores."""
    enc, register, regfr = _prep(ag, proc, imgs, reg_np, desc, "r1", nf)
    mm = {k: enc[k] for k in enc if k not in ("input_ids", "attention_mask")}
    temp_h, ground_h, reg_h, rf = ag.encode(enc["input_ids"], enc["attention_mask"], register, regfr, **mm)
    tl = ag.temporal_logits(temp_h, reg_h, rf, nf)
    scores = tl[0].sigmoid().float().cpu().numpy()
    if SEL_HEAD:                                                   # E4: also predicted #frames-to-select
        k = float(ag.count_pred(temp_h)[0]); return scores, int(np.clip(round(k), 4, nf))
    if ACTION:                                                     # E2: also P(selection insufficient -> refine)
        return scores, float(torch.sigmoid(ag.action_logit(temp_h))[0])
    return scores


@torch.no_grad()
def _agent_ground_r2(ag, proc, frames_np, reg_np, desc, qemb, masks_s, sample, fsel, qemb_all=None, inst3d_all=None):
    """R2: images (<=NF_VLM of fsel) + re-run register -> <GROUND> -> matched/decoded 3D inst.
    ALLFRAME: match against all-frame qemb and answer with full-coverage all-frame inst3d."""
    enc, register, regfr = _prep(ag, proc, frames_np, reg_np, desc, "r2", frames_np.shape[0])
    mm = {k: enc[k] for k in enc if k not in ("input_ids", "attention_mask")}
    temp_h, ground_h, reg_h, rf = ag.encode(enc["input_ids"], enc["attention_mask"], register, regfr, **mm)
    # masks_s / field are on the fsel frames -> intrinsics/extrinsics/depth must all be subset to fsel (map_pred indexes per view)
    intr, ext, dep = sample["intrinsics"][fsel], sample["extrinsics"][fsel], sample["depth_maps"][fsel]
    if LOSS == "matching":
        t = F.normalize(ag.text_proj(ground_h.float()), dim=-1)
        if qemb_all is not None:                                          # match all-frame instances -> full-coverage 3D
            e = F.normalize(ag.inst_proj(torch.from_numpy(qemb_all).float().cuda()), dim=-1)
            sel = int((t @ e.t()).argmax(1))
            return {"inst3d": inst3d_all[sel], "all": (inst3d_all if DIAG else None), "sel": sel}
        e = F.normalize(ag.inst_proj(torch.from_numpy(qemb).float().cuda()), dim=-1)   # heads are fp32; ground_h is bf16
        sel = int((t @ e.t()).argmax(1))
        inst3d = map_pred_inst_to_gt_pointcloud(sample["points"], sample["super_points"],
                    intr, ext, dep, masks_s, depth_tolerance=0.1)
        return {"inst3d": inst3d[sel], "all": (inst3d if DIAG else None), "sel": sel}
    else:
        # dense: gq_proj(ground)->mask_embed-> dot segmenter frozen mask_feat field (fsel frames) -> per-frame 96x128 -> lift to 3D
        field = _tl["field"].to(ground_h.dtype)                          # (1,n,256,96,128) from segmenter_forward hook
        pm = ag.decode_mask(ground_h, field)                            # (1,n,96,128)
        m2d = (pm.sigmoid() > 0.5)[0].float().cpu().numpy()[None]       # (1,n,96,128) single instance
        inst3d = map_pred_inst_to_gt_pointcloud(sample["points"], sample["super_points"],
                    intr, ext, dep, m2d, depth_tolerance=0.1)
        return {"inst3d": inst3d[0]}


# prompts MUST match train_agent.py
PROMPT_R1 = "Round 1: using the scene tokens from all {nf} frames, find which frames contain the described object.\nDescription: \"{d}\""
PROMPT_R2 = "Round 2: locate the described object in these {nf} frames.\nDescription: \"{d}\""
PROMPT_LOC = {   # MUST match train_agent.PROMPT_LOC
    "point": 'Locate the object described below in this image. Give its center as a normalized point (x, y), each coordinate in [0,1].\nDescription: "{d}"',
    "bbox":  'Locate the object described below in this image. Give its bounding box as normalized [x1, y1, x2, y2], each coordinate in [0,1].\nDescription: "{d}"',
}


def _parse_coord(txt):
    """extract the (up-to-4) normalized floats the model generated. point->2, bbox->4. None if too few."""
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", txt)
    vals = [float(x) for x in nums]
    need = 2 if LOC == "point" else 4
    if len(vals) < need: return None
    return [min(1.0, max(0.0, v)) for v in vals[:need]]


def _loc_hit(coord, mask):
    """coord normalized [0,1]; mask (MH,MW) bool of the target in this frame. point: center in mask. bbox: IoU>=0.5 vs tight GT box."""
    if coord is None or mask.sum() == 0: return 0
    if LOC == "point":
        c = int(min(MW - 1, max(0, round(coord[0] * MW)))); r = int(min(MH - 1, max(0, round(coord[1] * MH))))
        return int(bool(mask[r, c]))
    ys, xs = np.where(mask)
    gt = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    pb = [coord[0] * MW, coord[1] * MH, coord[2] * MW, coord[3] * MH]
    ix1, iy1 = max(pb[0], gt[0]), max(pb[1], gt[1]); ix2, iy2 = min(pb[2], gt[2]), min(pb[3], gt[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ap = max(0.0, pb[2] - pb[0]) * max(0.0, pb[3] - pb[1]); ag = (gt[2] - gt[0]) * (gt[3] - gt[1])
    return int(inter / max(ap + ag - inter, 1e-6) >= 0.5)


@torch.no_grad()
def _agent_ground_r2_fixedq(ag, proc, reg_np, nreg_f, frames_np, desc, qemb_all):
    """R2 fixed-Q (matches training): all-frame register + top-8 images -> <GROUND> -> match the FIXED all-frame qemb -> instance idx."""
    enc, register, regfr = _prep(ag, proc, frames_np, reg_np, desc, "r2", nreg_f)
    mm = {k: enc[k] for k in enc if k not in ("input_ids", "attention_mask")}
    temp_h, ground_h, reg_h, rf = ag.encode(enc["input_ids"], enc["attention_mask"], register, regfr, **mm)
    t = F.normalize(ag.text_proj(ground_h.float()), dim=-1)
    e = F.normalize(ag.inst_proj(torch.from_numpy(qemb_all).float().cuda()), dim=-1)   # heads fp32
    return int((t @ e.t()).argmax(1))


@torch.no_grad()
def _agent_ground_r2_fixedq_inst(ag, proc, reg_np, nreg_f, frames_np, desc, qemb_all):
    """ExpA eval: append <INST> tokens (qemb injected, distribution-aligned), <GROUND> matches their CONTEXTUALIZED hiddens."""
    frs = []
    for f in frames_np:
        im = Image.fromarray(f.astype(np.uint8))
        if im.width > 512: im = im.resize((512, int(im.height*512/im.width)))
        frs.append(im)
    n = qemb_all.shape[0]
    pr = PROMPT_R2.format(nf=nreg_f, d=desc)
    content = [{"type": "image"} for _ in frs] + [{"type": "text", "text": pr}]
    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True) + REG*nreg_f + f' Target: "{desc}". ' + TEMP + INST*n + GROUND
    enc = proc(text=[text], images=[frs], return_tensors="pt", padding=True).to("cuda")
    mm = {k: enc[k] for k in enc if k not in ("input_ids", "attention_mask")}
    register = torch.from_numpy(reg_np.reshape(-1, 2048).astype(np.float32))[None].cuda()
    regfr = torch.arange(nreg_f)[None].cuda()
    inst_emb = torch.from_numpy(qemb_all.astype(np.float32))[None].cuda(); inst_valid = torch.ones(1, n, dtype=torch.bool, device="cuda")
    temp_h, ground_h, reg_h, rf = ag.encode(enc["input_ids"], enc["attention_mask"], register, regfr, inst_emb=inst_emb, inst_valid=inst_valid, **mm)
    ml = ag.inst_match_logits(ground_h.float(), ag._inst_h, inst_valid)
    return int(ml.argmax(1))


@torch.no_grad()
def _agent_sframe(ag, proc, reg_1, img_1, desc, e_norm):
    """Single-frame grounder: ONE frame image + that frame's 17 teacher reg tokens -> <GROUND>.
    Match vs [qemb queries (scaled cosine) ; null (raw)] EXACTLY as trained. Returns (inst_idx or -1=null, softmax_conf)."""
    im = Image.fromarray(img_1.astype(np.uint8))
    if im.width > 512: im = im.resize((512, int(im.height * 512 / im.width)))
    content = [{"type": "image"}, {"type": "text", "text": SF_PROMPT.format(d=desc)}]
    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True) + ("" if SF_NOREG else REG*17) + f' Target: "{desc}". ' + GROUND
    enc = proc(text=[text], images=[[im]], return_tensors="pt", padding=True).to("cuda")
    mm = {k: enc[k] for k in enc if k not in ("input_ids", "attention_mask")}
    if SF_NOREG:                                                          # image-only: dummy 1-token register (regfr=-1 -> not scattered)
        register = torch.zeros(1, 1, 2048).cuda(); regfr = torch.full((1, 1), -1, dtype=torch.long).cuda()
    else:
        register = torch.from_numpy(reg_1.reshape(-1, 2048).astype(np.float32))[None].cuda()   # (1,17,2048)
        regfr = torch.zeros(1, 17, dtype=torch.long).cuda()
    temp_h, ground_h, reg_h, rf = ag.encode(enc["input_ids"], enc["attention_mask"], register, regfr, **mm)
    t = F.normalize(ag.text_proj(ground_h.float()), dim=-1)             # (1,P)
    ml = (t @ e_norm.t()) * ag.logit_scale.exp().float()                # (1,N) scaled cosine (== match_logits)
    nl = ag.null_logit(ground_h.float())                               # (1,1) raw null logit
    full = torch.cat([ml, nl], 1)                                       # (1,N+1)
    p = full.softmax(1)[0]; idx = int(full.argmax(1)); N = e_norm.shape[0]
    return (-1 if idx >= N else idx), float(p[idx])


@torch.no_grad()
@torch.no_grad()
def _gen_mf_cot(ag, proc, imgs_win, desc, nsamp=1, temp=1.0):
    """MF+CoT (cot1): generate the <think> reasoning ONCE per query (reused across all sliding windows).
    No register in the prefix (register lives in the Frame blocks, AFTER the think) -> plain multimodal generate."""
    ims = []
    for f in imgs_win:
        im = Image.fromarray(f.astype(np.uint8))
        if im.width > 512: im = im.resize((512, int(im.height*512/im.width)))
        ims.append(im)
    content = [{"type": "image"} for _ in ims] + [{"type": "text", "text": MF_PROMPT.format(n=len(ims), d=desc)}]
    pre = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True) + THINK_S
    enc = proc(text=[pre], images=[ims], return_tensors="pt", padding=True).to("cuda")
    try:
        _mt = int(os.environ.get("MF_COT_MAXTOK", "96"))
        if nsamp > 1:                                                          # PROBE/RL: sample K CoTs at temperature
            gen = ag.vlm.generate(**enc, max_new_tokens=_mt, do_sample=True, temperature=temp, top_p=0.95,
                                  num_return_sequences=nsamp, eos_token_id=ag.think_e_id, pad_token_id=proc.tokenizer.pad_token_id)
            return [proc.tokenizer.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for g in gen]
        gen = ag.vlm.generate(**enc, max_new_tokens=_mt, do_sample=False, eos_token_id=ag.think_e_id, pad_token_id=proc.tokenizer.pad_token_id)
        new = gen[0][enc["input_ids"].shape[1]:]
        cot = proc.tokenizer.decode(new, skip_special_tokens=True).strip()
        return cot
    except Exception:
        return [""] * nsamp if nsamp > 1 else ""


@torch.no_grad()
def _agent_mframe(ag, proc, reg_win, imgs_win, desc, e_norm, cot_text="", imgs_tower=None):
    """Multi-frame: tower images + NF*17 reg + NF <GROUND> (one per frame) in ONE forward.
    NF = len(reg_win) frame blocks; tower sees imgs_tower (MF32 subset) or all imgs_win (original MF).
    Returns list of (inst_idx or -1=null, conf) per frame (length NF). Layout matches MFRAME training.
    cot_text: if given (MF+CoT cot1), insert <think> cot_text </think> before the Frame blocks."""
    nf = len(reg_win); ims = []
    for f in (imgs_tower if imgs_tower is not None else imgs_win):
        im = Image.fromarray(f.astype(np.uint8))
        if im.width > 512: im = im.resize((512, int(im.height*512/im.width)))
        ims.append(im)
    content = [{"type": "image"} for _ in range(len(ims))] + [{"type": "text", "text": MF_PROMPT.format(n=nf, d=desc)}]
    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    if MF_COT and cot_text: text += THINK_S + cot_text + THINK_E          # MF+CoT: reason before grounding (teacher-force the generated CoT)
    for j in range(nf): text += f' Frame {j+1}: ' + ("" if SF_NOREG else REG*17) + GROUND + (POINT if getattr(ag, "point_out", False) else "")   # dual-output: <POINT> must be present at eval too (train/eval token layout consistency)
    enc = proc(text=[text], images=[ims], return_tensors="pt", padding=True).to("cuda")
    mm = {k: enc[k] for k in enc if k not in ("input_ids", "attention_mask")}
    if SF_NOREG:
        register = torch.zeros(1, 1, 2048).cuda(); regfr = torch.full((1, 1), -1, dtype=torch.long).cuda()
    else:
        register = torch.from_numpy(reg_win.reshape(-1, 2048).astype(np.float32))[None].cuda()   # (1, nf*17, 2048)
        regfr = torch.arange(nf).repeat_interleave(17)[None].cuda()
    temp_h, ground_all, reg_h, rf = ag.encode(enc["input_ids"], enc["attention_mask"], register, regfr, multi_ground=True, **mm)
    gh = ground_all[0]                                                    # (nf, H)
    if getattr(ag, "point_out", False) and getattr(ag, "_point_h", None) is not None:   # Arm B: <POINT> cross-attends patch grid -> (x,y,null)
        hd = next(ag.point_head.parameters()).dtype; q = ag._point_h[0].reshape(nf, ag.H).to(hd)
        _cp = getattr(ag, "_cur_patch", None)
        if _cp is not None: pg = torch.from_numpy(np.ascontiguousarray(_cp)).reshape(nf, PATCH_GH * PATCH_GW, 2048).to(q.device)
        else: pg = torch.zeros(nf, PATCH_GH * PATCH_GW, 2048, device=q.device, dtype=hd)
        o = ag.point_head(q, pg).float()
        ag._last_points = torch.sigmoid(o[:, :2]).cpu().numpy()             # (nf,2) predicted point
        ag._last_point_null = torch.sigmoid(o[:, 2]).cpu().numpy()          # (nf,) P(absent) = point head's own null gate
    else:
        ag._last_points = None; ag._last_point_null = None
    t = F.normalize(ag.text_proj(gh.float()), dim=-1)                     # (nf, P)
    ml = (t @ e_norm.t()) * ag.logit_scale.exp().float()                 # (nf, N)
    nl = ag.null_logit(gh.float())                                       # (nf, 1)
    if BAN_INST:
        for _b in BAN_INST:
            if 0 <= _b < ml.shape[1]: ml[:, _b] = -1e4
    full = torch.cat([ml, nl], 1); p = full.softmax(1); idx = full.argmax(1); N = e_norm.shape[0]
    return [((-1 if int(idx[j]) >= N else int(idx[j])), float(p[j, idx[j]])) for j in range(nf)]


@torch.no_grad()
def _p2_boxstr(boxes, nf):
    """Pass-1 boxes in the exact training text form (same as selfcorrect_data.pass1_box_str)."""
    return " ".join(f"F{k+1}: <box>{int(round(b[0]*1000))},{int(round(b[1]*1000))},{int(round(b[2]*1000))},{int(round(b[3]*1000))}</box>"
                    if b is not None else f"F{k+1}: absent" for k, b in enumerate(boxes[:nf]))


_BAN_MAP = {}
if os.environ.get("BAN_MAP"):
    import json as _bj; _BAN_MAP = _bj.load(open(os.environ["BAN_MAP"]))
BAN_INST = set()
P2_FREE = os.environ.get("P2_FREE") == "1"
P2_ACC = os.environ.get("P2_ACC") == "1"
P2_RESAMPLE = os.environ.get("P2_RESAMPLE") == "1"
P2_TRIG = float(os.environ.get("P2_TRIG", "0"))
P2_MIX = tuple(int(x) for x in os.environ.get("P2_MIX", "4,2,2").split(","))


def _p2_resample(p2wins0, target, L, nf, mix=None, trig=0.0):
    """" +  + "



    """
    import numpy as _np
    nc, nn, na = mix or (4, 2, 2)
    box_n, q_n, q_cf, seen_n, box_of, q_of = {}, {}, {}, {}, {}, {}
    for (win_w, bxs_w, fres_w) in p2wins0:
        for j, f in enumerate(win_w):
            f = int(f); seen_n[f] = seen_n.get(f, 0) + 1
            bp = j < len(bxs_w) and bxs_w[j] is not None
            box_n[f] = box_n.get(f, 0) + int(bp)
            if bp: box_of.setdefault(f, bxs_w[j])
            if j < len(fres_w):
                ix, cf = fres_w[j]
                q_n[f] = q_n.get(f, 0) + int(ix == target)
                if ix == target: q_cf[f] = q_cf.get(f, 0.0) + float(cf)
                q_of.setdefault(f, (int(ix), float(cf)))
    frames = sorted(seen_n)
    contested = [f for f in frames if (box_n.get(f, 0) > 0) != (q_n.get(f, 0) > 0)]
    if not frames or (trig > 0 and len(contested) / len(frames) < trig): return [], len(contested), len(frames)
    if not contested: return [], 0, len(frames)
    anchors = sorted([f for f in frames if q_n.get(f, 0) > 0 and box_n.get(f, 0) > 0],
                     key=lambda f: -q_cf.get(f, 0.0))[:max(na, 1) * 4]

    home = {}
    for wi, (win_w, _, _) in enumerate(p2wins0):
        for f in win_w: home.setdefault(int(f), wi)
    order = sorted(contested, key=lambda f: (home.get(f, 0), f))
    order = [order[i] for k in range(len(order)) for i in range(k, len(order), max(1, len(order) // max(nc, 1)))][:len(contested)]
    seen_o, ordered = set(), []
    for f in order:
        if f not in seen_o: seen_o.add(f); ordered.append(f)

    out, used_nb = [], set()
    for i in range(0, len(ordered), max(nc, 1)):
        grp = ordered[i:i + nc]
        nbs = []
        for f in grp:
            for d in (1, -1, 2, -2, 3, -3):
                g = f + d
                if 0 <= g < L and g not in used_nb and g not in grp:
                    nbs.append(g); used_nb.add(g); break
            if len(nbs) >= nn: break
        wf = list(grp) + nbs[:nn] + [a for a in anchors[:na]]
        wf = list(dict.fromkeys(wf))[:nf]
        while len(wf) < nf and anchors:
            for a in anchors:
                if a not in wf: wf.append(a); break
            else: break
        if len(wf) < nf: wf += [wf[-1]] * (nf - len(wf))
        out.append((_np.asarray(wf, int),
                    [box_of.get(f) for f in wf],
                    [q_of.get(f, (-1, 0.0)) for f in wf]))
    return out, len(contested), len(frames)


def _p2_review(boxes, qidx, qconf, target, nf):
    """[REVIEW] line. At inference the manifest target is unknown, so `target` is the query-level qemb-VOTED
    instance (what the model already predicts) -- no GT leaks in."""
    parts = []
    for k in range(nf):
        bp = (k < len(boxes)) and (boxes[k] is not None)
        qm = (qidx[k] == target) if (target >= 0 and k < len(qidx)) else False
        bs = (f"<box>{int(round(boxes[k][0]*1000))},{int(round(boxes[k][1]*1000))},{int(round(boxes[k][2]*1000))},{int(round(boxes[k][3]*1000))}</box>"
              if bp else "none")
        qcs = f"({qconf[k]:.2f})" if (P2_REVIEW_CONF and k < len(qconf)) else ""
        parts.append(f"F{k+1}: box={bs} qemb={'target' if qm else 'null'}{qcs} {'agree' if bp == qm else 'CONFLICT'}")
    return "[REVIEW] " + " \u00b7 ".join(parts)


P2_BS = int(os.environ.get("P2_BS", "8"))    # windows per Turn-2 forward (see _gen_mf_turn2 below)
P2_T2_GROUND = os.environ.get("P2_T2_GROUND", "1") == "1"  # 0 -> Turn-2 emits no <GROUND> (same flag as training)
P2_DONE_KEEP = os.environ.get("P2_DONE_KEEP") == "1"       # pair with DONE_NO_BOX=1
P2_DEBUG = os.environ.get("P2_DEBUG") == "1"     # dump the RAW Turn-2 stage-1 text (gate diagnosis)


@torch.no_grad()
def _gen_mf_turn2(ag, proc, reg_wins, imgs_wins, desc, cot_text, p1strs, reviews, hists=None, e_norm=None):
    """CHUNKED wrapper over the batched implementation. One query can carry 40+ windows and each window feeds
    8 images, so a single batch asked for a 41 GiB attention buffer on scene0208_00 -- and because the failure
    left that memory unreclaimed, EVERY later scene in the shard then OOMed on ~300 MiB allocations (15 scenes
    lost). Output is per-window, so splitting the batch is exact, not an approximation. On OOM we retry the
    chunk one window at a time rather than dropping the scene."""
    n = len(reg_wins)
    if n <= P2_BS:
        return _gen_mf_turn2_batch(ag, proc, reg_wins, imgs_wins, desc, cot_text, p1strs, reviews, hists, e_norm)
    out = []
    for i in range(0, n, P2_BS):
        s = slice(i, min(i + P2_BS, n))
        _hs = None if hists is None else hists[s]
        try:
            out += _gen_mf_turn2_batch(ag, proc, reg_wins[s], imgs_wins[s], desc, cot_text, p1strs[s], reviews[s], _hs, e_norm)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            for j in range(i, min(i + P2_BS, n)):                                  # degrade to one window at a time
                out += _gen_mf_turn2_batch(ag, proc, reg_wins[j:j+1], imgs_wins[j:j+1], desc, cot_text,
                                           p1strs[j:j+1], reviews[j:j+1],
                                           None if hists is None else hists[j:j+1], e_norm)
    return out


@torch.no_grad()
def _gen_mf_turn2_batch(ag, proc, reg_wins, imgs_wins, desc, cot_text, p1strs, reviews, hists=None, e_norm=None):
    """BATCHED Turn-2 in TWO STAGES, because Turn-2 now mirrors Turn-1's structure and therefore contains
    <REG>/<GROUND> slots -- those are injected/read by the harness, never generated by the model:

      stage 1: ... [REVIEW]                      -> model generates '<reflect> ... </reflect>'
      stage 2: append 'Frame j: <REG>x17 <GROUND>' x NF + ' Boxes(corrected):', ONE forward that both
               READS the new per-frame <GROUND> hiddens (corrected instance choice) and GENERATES the boxes.

    Returns list of (boxes|None, kind, raw_text, qmatch) where qmatch[j] = (instance idx | -1 null, conf)
    from Turn-2's own readout -- so reflection can revise the qemb path, not just the boxes.
    `hists[w]` carries earlier rounds' full output so round>=2 continues the same rollout."""
    nf = len(reg_wins[0]); pre, images = [], []
    for w in range(len(reg_wins)):
        ims = []
        for f in imgs_wins[w]:
            im = Image.fromarray(f.astype(np.uint8))
            if im.width > 512: im = im.resize((512, int(im.height * 512 / im.width)))
            ims.append(im)
        content = [{"type": "image"} for _ in ims] + [{"type": "text", "text": MF_PROMPT.format(n=nf, d=desc)}]
        t = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        if MF_COT and cot_text: t += THINK_S + cot_text + THINK_E
        for j in range(nf): t += f' Frame {j+1}: ' + ("" if SF_NOREG else REG*17) + GROUND
        t += " Boxes: " + p1strs[w] + " " + reviews[w]
        if hists and hists[w]: t += " " + hists[w]
        pre.append(t); images.append(ims)

    def _reg2(k):        # k runs of the SAME per-frame registers (Turn-1's, plus one per Turn-2 emitted so far)
        one = np.stack([r.reshape(-1, 2048) for r in reg_wins]).astype(np.float32)
        return (torch.from_numpy(np.concatenate([one] * k, 1)).cuda(),
                torch.arange(nf).repeat_interleave(17).repeat(k)[None].repeat(len(reg_wins), 1).cuda())

    def _reg2_var(ks):   # PER-SAMPLE run count: a done window emits no Turn-2 readout, so it has one run fewer
        one = np.stack([r.reshape(-1, 2048) for r in reg_wins]).astype(np.float32)   # (B, nf*17, 2048)
        R = max(ks) * nf * 17
        reg = torch.zeros(len(ks), R, 2048); rf = torch.full((len(ks), R), -1, dtype=torch.long)
        base = torch.arange(nf).repeat_interleave(17)
        for w, k in enumerate(ks):
            reg[w, :k * nf * 17] = torch.from_numpy(np.concatenate([one[w]] * k, 0))
            rf[w, :k * nf * 17] = base.repeat(k)
        return reg.cuda(), rf.cuda()

    n_prev = 1 + (0 if not (hists and any(hists)) else sum(hists[0].count(GROUND) for _ in [0]) // nf)
    # ---- stage 1: the reflection text ----
    proc.tokenizer.padding_side = "left"
    enc = proc(text=pre, images=images, return_tensors="pt", padding=True).to("cuda")
    mm = {k: enc[k] for k in enc if k not in ("input_ids", "attention_mask")}
    reg1, rf1 = _reg2(n_prev)
    g1 = ag.gen_boxes(enc["input_ids"], enc["attention_mask"], reg1, rf1, max_new_tokens=260,
                      eos_id=proc.tokenizer.eos_token_id, pad_id=proc.tokenizer.pad_token_id, **mm)
    refl = []
    for w in range(len(reg_wins)):
        raw = proc.tokenizer.decode(g1[w], skip_special_tokens=True)
        # The gate token sits AFTER the closing tag in training ("<reflect> ... </reflect> <done/>"), so cutting
        # at "</reflect>" would delete it and the gate would read as 'reflect' for every well-formed generation.
        # Keep the trailing <done/> when it is there; still cut whatever runs on after it.
        if "</reflect>" in raw:
            head, _, tail = raw.partition("</reflect>")
            tx = head + "</reflect>" + (" <done/>" if tail.lstrip().startswith("<done/>") else "")
        else:
            tx = raw.strip()
        if P2_DEBUG and w < 3:
            print(f"  [gate-probe w{w}] kind={'done' if '<done/>' in tx else 'reflect'} "
                  f"| raw tail ...{raw[-70:]!r} | kept tail ...{tx[-40:]!r}", flush=True)
        refl.append(tx.strip())
    # ---- stage 2: (readout blocks only when the model did NOT say <done/>) + corrected boxes ----
    # P2-g3: <done/> now gates the qemb head alone. A done window keeps Pass-1's instance choice (no readout
    # emitted, so nothing to read), while its boxes are refined like every other window.
    dn = [("<done/>" in refl[w]) for w in range(len(reg_wins))]
    # P2_T2_GROUND=0: Turn-2 emits no <GROUND>, so it cannot re-decide the instance and corrects the box list
    # alone. The per-frame <REG>x17 are STILL injected for every window -- those are Turn-2's INPUT, and dropping
    # them would silently change what the second turn sees, which is a different experiment.
    # P2_DONE_KEEP: <done/> means what it says -- no box list follows, Pass-1 stands. Must match the training
    # data (DONE_NO_BOX), because appending 'Boxes(corrected):' to a <done/> the model was never trained to
    # continue is off-distribution and the boxes it then invents are noise, not a correction.
    emit = [True] * len(reg_wins) if not P2_T2_GROUND else [not d for d in dn]
    full = []
    for w in range(len(reg_wins)):
        t = pre[w] + " " + refl[w]
        if emit[w]:
            for j in range(nf): t += f' Frame {j+1}: ' + ("" if SF_NOREG else REG*17) + (GROUND if P2_T2_GROUND else "")
        if not (P2_DONE_KEEP and dn[w]): t += " Boxes(corrected):"
        full.append(t)
    enc2 = proc(text=full, images=images, return_tensors="pt", padding=True).to("cuda")
    mm2 = {k: enc2[k] for k in enc2 if k not in ("input_ids", "attention_mask")}
    reg2, rf2 = _reg2_var([n_prev + (1 if e else 0) for e in emit])
    g2 = ag.gen_boxes(enc2["input_ids"], enc2["attention_mask"], reg2, rf2, max_new_tokens=220,
                      eos_id=proc.tokenizer.eos_token_id, pad_id=proc.tokenizer.pad_token_id, **mm2)
    proc.tokenizer.padding_side = "right"
    enc3 = proc(text=full, images=images, return_tensors="pt", padding=True).to("cuda")
    mm3 = {k: enc3[k] for k in enc3 if k not in ("input_ids", "attention_mask")}
    reg3, rf3 = _reg2_var([n_prev + (1 if e else 0) for e in emit])
    gall = None
    if e_norm is not None and any(emit) and P2_T2_GROUND:                 # nothing to read when no window emits
        _, gall, _, _ = ag.encode(enc3["input_ids"], enc3["attention_mask"], reg3, rf3, multi_ground=True, **mm3)
    out = []
    for w in range(len(reg_wins)):
        if P2_DONE_KEEP and dn[w]:                                        # <done/> -> Pass-1 stands, nothing parsed
            out.append((None, "done", refl[w], None)); continue
        txt = proc.tokenizer.decode(g2[w], skip_special_tokens=True)
        bb = [None] * nf
        seg = re.split(r"<done/>|Boxes\(corrected\)|<\|", txt)[0]
        for k in range(nf):
            m = re.search(rf"F{k+1}\s*:\s*<box>([\d,\s]+)</box>", seg)
            if m:
                v = re.findall(r"\d+", m.group(1))
                if len(v) >= 4: bb[k] = [int(v[i]) / 1000.0 for i in range(4)]
        qm = None
        if e_norm is not None and emit[w] and P2_T2_GROUND:           # Turn-2's OWN readout (absent for done windows)
            gh = gall[w, -nf:]
            tp = F.normalize(ag.text_proj(gh.float()), dim=-1)
            ml = (tp @ e_norm.t()) * ag.logit_scale.exp().float(); nl = ag.null_logit(gh.float())
            fu = torch.cat([ml, nl], 1); pr = fu.softmax(1); ix = fu.argmax(1); Nq = e_norm.shape[0]
            qm = [(-1 if int(ix[j]) >= Nq else int(ix[j]), round(float(pr[j, ix[j]]), 3)) for j in range(nf)]
        kind = "done" if "<done/>" in refl[w] else "reflect"
        out.append((bb, kind, refl[w], qm))
    return out



TREE_TRACE = None


def mf_tree_windows(FR, L, genfn, nf=None, depth=None, batch=None, policy=None):
    """,  grounder_infer  ScanNet++/videos bench

    """
    nf = MF_NF if nf is None else nf
    depth = MF_TREE_DEPTH if depth is None else depth
    batch = MF_BATCH if batch is None else batch
    _tw, _lvl, _cache = [], _tree_levels(L, nf), []
    _trace = TREE_TRACE if isinstance(TREE_TRACE, list) else None
    for _d in range(depth):
        if not _lvl: break
        _fr = [_pick(a, b, nf) for a, b in _lvl]
        _wl = [FR[np.array(f, int)] for f in _fr]
        _bl = [None] * len(_wl)
        _bylen = {}
        for _j, _w in enumerate(_wl): _bylen.setdefault(len(_w), []).append(_j)
        for _idxs in _bylen.values():
            for _i in range(0, len(_idxs), max(batch, 1)):
                _cid = _idxs[_i:_i + max(batch, 1)]
                for _j, _r in zip(_cid, genfn([_wl[_j] for _j in _cid])): _bl[_j] = _r
        _nx = []
        for (_a, _b), _f, _w, _bx in zip(_lvl, _fr, _wl, _bl):
            _tw.append((_w, _bx))
            _p = [j for j in range(len(_f)) if j < len(_bx) and _bx[j] is not None]
            _kids = _children(_a, _b, _f, _p, nf, _d)
            if policy is not None and not policy(int(_a), int(_b), [int(x) for x in _f], _p, int(_d), int(L), _bx):
                _kids = []
            if _trace is not None:
                _trace.append({"depth": int(_d), "lo": int(_a), "hi": int(_b),
                               "frames": [int(x) for x in _f],
                               "fired": [int(j) for j in _p],
                               "children": [[int(c0), int(c1)] for c0, c1 in _kids]})
            for (_ca, _cb) in _kids:
                if _cb - _ca + 1 > nf:
                    _nx.append((_ca, _cb))
                else:
                    _cache.extend(int(x) for x in range(_ca, _cb + 1))
        _lvl = _nx
    _seen = set(); _cf = [x for x in _cache if not (x in _seen or _seen.add(x))]
    for _i in range(0, len(_cf), nf):
        _ck = FR[np.array(_cf[_i:_i + nf], int)]
        if len(_ck) == 0: continue
        _tw.append((_ck, genfn([_ck])[0]))
    if os.environ.get("TREE_WINCNT"):
        print(f"[wincnt] tree_windows={len(_tw)} L={L}", flush=True)
    return [w for w, _ in _tw], [b for _, b in _tw]


def _tree_levels(L, nf):
    """Coarse-to-fine frame search, replacing the uniform sliding window.

    Level 0 samples nf frames across the WHOLE video; wherever the object is reported present we recurse into
    the span between the neighbouring samples, until a span is short enough that the nf samples are the frames
    themselves. Cost is ~log_nf(L) levels of a few windows instead of L/stride windows -- and it matches what
    the box path actually consumes, since only PROMPT_K=5 boxes are ever fed to the prompt encoder.

    A miss at level 0 would otherwise end the search with 'absent everywhere', so level 0 expands into all nf
    sub-intervals UNCONDITIONALLY; pruning only starts at level 1.
    """
    return [(0, L - 1)]


def _pick(lo, hi, nf):
    """nf frame indices spanning [lo,hi]; the whole span once it is short enough."""
    if hi - lo + 1 <= nf:
        return list(range(lo, hi + 1))
    return sorted(set(int(round(x)) for x in np.linspace(lo, hi, nf)))


def _children(lo, hi, frames, pos, nf, depth):
    """pos = indices (into `frames`) reported present -> spans to refine."""
    if not pos:
        if depth > 0:
            return []                                                     # prune: nothing here
        e = np.linspace(lo, hi + 1, nf + 1).round().astype(int)            # level 0 miss -> expand everything
        return [(int(e[i]), int(e[i + 1]) - 1) for i in range(nf) if int(e[i + 1]) - 1 > int(e[i])]
    out = []; a = 0
    while a < len(pos):
        b = a
        while b + 1 < len(pos) and pos[b + 1] == pos[b] + 1: b += 1        # maximal run of present samples
        l = frames[pos[a] - 1] if pos[a] > 0 else lo                       # one neighbour out on each side:
        r = frames[pos[b] + 1] if pos[b] + 1 < len(frames) else hi         # the object may extend into the gap
        if r - l + 1 > nf:
            if l <= lo and r >= hi:
                # DEGENERATE CASE: the run spans the whole parent, so (l,r) == (lo,hi) and _pick is a
                # deterministic linspace -- the next level would re-sample the IDENTICAL frames and the
                # remaining depth is spent re-running one window. Measured: an object present in every
                # sampled frame gave 4 windows but only 1 distinct one, covering 8 of 90 frames.
                # Split at the nf sampled frames -- the nf samples define nf-1 gaps, and those gaps ARE the
                # units we have evidence about, so they are the natural children. Each gap must still pass the
                # SAME `> nf` gate as any other child: without it a 2-frame gap keeps recursing and a 90-frame
                # clip blows up to 57 windows (more than sliding's 22), which was my first attempt at this fix.
                for i in range(len(frames) - 1):
                    s0, s1 = int(frames[i]), int(frames[i + 1])
                    if s1 - s0 > 1: out.append((s0, s1))
            else:
                out.append((int(l), int(r)))
        a = b + 1

    return out


def _gen_mf_box_batch(ag, proc, reg_wins, imgs_wins, desc, cot_text=""):
    """BATCHED _gen_mf_box over B windows. The eval generated ONE window per call (batch=1) while label_traj
    has had a batched version all along -- with ~50 windows per query that is the single biggest waste in the
    whole eval. Left-padded so greedy decoding is unaffected; MF_BATCH=1 falls back to the per-window path, so
    the two can be diffed against each other.

    nf comes from the WINDOW, not the global MF_NF: tree traversal takes a span of <=MF_NF frames whole, so a
    window can be shorter. _gen_mf_box has always done this; the batch path predates the tree and did not."""
    nf = len(reg_wins[0]); texts, images = [], []
    assert all(len(r) == nf for r in reg_wins), "batch mixes window lengths -- group by length before calling"
    for imgs_win in imgs_wins:
        ims = []
        for f in imgs_win:
            im = Image.fromarray(f.astype(np.uint8))
            if im.width > 512: im = im.resize((512, int(im.height * 512 / im.width)))
            ims.append(im)
        content = [{"type": "image"} for _ in range(len(ims))] + [{"type": "text", "text": MF_PROMPT.format(n=nf, d=desc)}]
        t = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        if MF_COT and cot_text: t += THINK_S + cot_text + THINK_E
        for j in range(nf): t += f' Frame {j+1}: ' + ("" if SF_NOREG else REG*17) + GROUND
        t += " Boxes:"
        texts.append(t); images.append(ims)
    proc.tokenizer.padding_side = "left"                                  # generate needs left-pad
    enc = proc(text=texts, images=images, return_tensors="pt", padding=True).to("cuda")
    mm = {k: enc[k] for k in enc if k not in ("input_ids", "attention_mask")}
    register = torch.from_numpy(np.stack([r.reshape(-1, 2048) for r in reg_wins]).astype(np.float32)).cuda()
    regfr = torch.arange(nf).repeat_interleave(17)[None].expand(len(reg_wins), -1).contiguous().cuda()
    # VLLM_BOX swaps ONLY the generate. Prompt building above and box parsing below are shared with the torch
    # path verbatim, so any backend difference is attributable to generation and not to a drifted prompt.
    if os.environ.get("VLLM_BOX") == "1":
        from vllm_box_backend import gen_texts
        txts = gen_texts(ag, proc, texts, images, enc["input_ids"], register, regfr, nf)
    else:
        gen = ag.gen_boxes(enc["input_ids"], enc["attention_mask"], register, regfr,
                           eos_id=proc.tokenizer.eos_token_id, pad_id=proc.tokenizer.pad_token_id, **mm)
        txts = [proc.tokenizer.decode(gen[b], skip_special_tokens=True) for b in range(len(reg_wins))]
    out = []
    for b in range(len(reg_wins)):
        txt = txts[b]
        boxes = [None] * nf
        for j in range(nf):
            m = re.search(rf"F{j+1}\s*:\s*<box>([\d,\s]+)</box>", txt)
            if m:
                v = re.findall(r"\d+", m.group(1))
                if len(v) >= 4: boxes[j] = [int(v[k]) / 1000.0 for k in range(4)]
        out.append(boxes)
    return out


def _gen_mf_box(ag, proc, reg_win, imgs_win, desc, cot_text=""):
    """Arm C: build the SAME prefix as training (frames + <GROUND> blocks + ' Boxes:'), register-inject + generate
    the per-frame box list, parse. Returns list[nf] of (x1,y1,x2,y2) normalized [0,1] or None (absent/unparsed)."""
    nf = len(reg_win); ims = []
    for f in imgs_win:
        im = Image.fromarray(f.astype(np.uint8))
        if im.width > 512: im = im.resize((512, int(im.height * 512 / im.width)))
        ims.append(im)
    content = [{"type": "image"} for _ in range(len(ims))] + [{"type": "text", "text": MF_PROMPT.format(n=nf, d=desc)}]
    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    if MF_COT and cot_text: text += THINK_S + cot_text + THINK_E
    for j in range(nf): text += f' Frame {j+1}: ' + ("" if SF_NOREG else REG*17) + GROUND
    text += " Boxes:"
    enc = proc(text=[text], images=[ims], return_tensors="pt", padding=True).to("cuda")
    mm = {k: enc[k] for k in enc if k not in ("input_ids", "attention_mask")}
    register = torch.from_numpy(reg_win.reshape(-1, 2048).astype(np.float32))[None].cuda()
    regfr = torch.arange(nf).repeat_interleave(17)[None].cuda()
    gen = ag.gen_boxes(enc["input_ids"], enc["attention_mask"], register, regfr,
                       eos_id=proc.tokenizer.eos_token_id, pad_id=proc.tokenizer.pad_token_id, **mm)
    txt = proc.tokenizer.decode(gen[0], skip_special_tokens=True)
    boxes = [None] * nf
    for j in range(nf):
        m = re.search(rf"F{j+1}\s*:\s*<box>([\d,\s]+)</box>", txt)
        if m:
            v = re.findall(r"\d+", m.group(1))
            if len(v) >= 4: boxes[j] = [int(v[k]) / 1000.0 for k in range(4)]
    return boxes


@torch.no_grad()
def _agent_ground_loc(ag, proc, frame_np, desc):
    """R2 (LOC): ONE frame image through Qwen3-VL vision tower -> greedy-generate the coord answer text -> parse."""
    im = Image.fromarray(frame_np.astype(np.uint8))
    if im.width > 512: im = im.resize((512, int(im.height * 512 / im.width)))
    content = [{"type": "image"}, {"type": "text", "text": PROMPT_LOC[LOC].format(d=desc)}]
    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    enc = proc(text=[text], images=[[im]], return_tensors="pt", padding=True).to("cuda")
    gen = ag.vlm.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=proc.tokenizer.pad_token_id)
    out = proc.tokenizer.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return _parse_coord(out)


def _prep(ag, proc, frames_np, reg_np, desc, regime, nreg_f):
    """R1: uniform-8 images + all-frame register(1/frame); R2: ALL selected-frame images + register(1/frame).
    register is pooled to 1 token/frame upstream. Matches training collate."""
    ntok = nreg_f * 17 if REG17 else nreg_f                              # E6: 17 <REG>/frame, else 1/frame
    frs = []
    for f in frames_np:                                                  # use ALL provided frames (no subsample)
        im = Image.fromarray(f.astype(np.uint8))
        if im.width > 512: im = im.resize((512, int(im.height*512/im.width)))
        frs.append(im)
    images, nimg = [frs], len(frs)
    pr = (PROMPT_R1 if regime == "r1" else PROMPT_R2).format(nf=nreg_f, d=desc)   # text always #frames
    content = [{"type": "image"} for _ in range(nimg)] + [{"type": "text", "text": pr}]
    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True) + REG*ntok + f' Target: "{desc}". ' + TEMP + GROUND
    enc = proc(text=[text], images=images, return_tensors="pt", padding=True).to("cuda")
    register = torch.from_numpy(reg_np.reshape(-1, 2048).astype(np.float32))[None].cuda()   # (1, ntok, 2048)
    regfr = (torch.arange(nreg_f).repeat_interleave(17) if REG17 else torch.arange(nreg_f))[None].cuda()
    return enc, register, regfr


def _topil(arr):
    out = []
    for f in arr:
        im = Image.fromarray(f.astype(np.uint8))
        if im.width > 512: im = im.resize((512, int(im.height*512/im.width)))
        out.append(im)
    return out


@torch.no_grad()
def _inject_reg(ag, input_ids, register, reg_frame):
    """Replicate AgentGrounder.encode's register injection -> input embeds (for KV-cache generation)."""
    base = ag.vlm.get_base_model() if hasattr(ag.vlm, "get_base_model") else ag.vlm
    emb = base.get_input_embeddings()(input_ids)
    regm = (input_ids == ag.reg_id)
    reg_in = F.normalize(register.float(), dim=-1).to(ag.reg_proj.weight.dtype)
    rp = ag.reg_proj(reg_in)
    if getattr(ag, "reg17", False):
        slot = (torch.arange(register.shape[1], device=rp.device) % 17)
        rp = rp + ag.reg_slot_emb(slot).to(rp.dtype)[None]
    rp = ag.reg_ln(rp)
    tok_norm = emb.detach().norm(dim=-1).mean().clamp(min=1e-6).to(rp.dtype)
    rp = rp / rp.norm(dim=-1, keepdim=True).clamp(min=1e-6) * tok_norm
    rvalid = reg_frame.ge(0)
    emb = emb.clone(); emb[regm] = rp[rvalid].to(emb.dtype)
    return emb


@torch.no_grad()
def _agent_ground_r2_cot(ag, proc, frames_np, reg_np, desc, qemb, masks_s, sample, fsel):
    """E5: R2 prompt ending at <think>; KV-cache generate the CoT; append <GROUND>; read it; match."""
    nreg_f = frames_np.shape[0]; pil = _topil(frames_np)
    content = [{"type": "image"} for _ in pil] + [{"type": "text", "text": PROMPT_R2.format(nf=nreg_f, d=desc)}]
    pre = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    pre = pre + REG * nreg_f + f' Target: "{desc}". ' + TEMP + THINK_S          # generate from here
    enc = proc(text=[pre], images=[pil], return_tensors="pt", padding=True).to("cuda")
    mm = {k: enc[k] for k in enc if k not in ("input_ids", "attention_mask")}
    register = torch.from_numpy(reg_np.reshape(-1, 2048).astype(np.float32))[None].cuda()
    regfr = torch.arange(nreg_f)[None].cuda()
    ids, am = enc["input_ids"], enc["attention_mask"]
    emb = _inject_reg(ag, ids, register, regfr)
    gen = ag.vlm.generate(inputs_embeds=emb, attention_mask=am, max_new_tokens=96, do_sample=False,
                          eos_token_id=ag.think_e_id, pad_token_id=proc.tokenizer.pad_token_id, **mm)  # new tokens only
    g = torch.tensor([[ag.ground_id]], device=ids.device)
    full = torch.cat([ids, gen, g], 1); am2 = torch.ones_like(full)
    temp_h, ground_h, reg_h, rf = ag.encode(full, am2, register, regfr, **mm)    # read the (only) <GROUND>
    t = F.normalize(ag.text_proj(ground_h.float()), dim=-1)
    e = F.normalize(ag.inst_proj(torch.from_numpy(qemb).float().cuda()), dim=-1)
    sel = int((t @ e.t()).argmax(1))
    intr, ext, dep = sample["intrinsics"][fsel], sample["extrinsics"][fsel], sample["depth_maps"][fsel]
    inst3d = map_pred_inst_to_gt_pointcloud(sample["points"], sample["super_points"], intr, ext, dep, masks_s, depth_tolerance=0.1)
    return {"inst3d": inst3d[sel], "sel": sel, "all": (inst3d if DIAG else None)}


@torch.no_grad()
def _agent_ground_r2_g5(ag, proc, r1_imgs, reg_r1, n1, sel_imgs, reg_s, desc, qemb, masks_s, sample, fsel):
    """G5: build the SAME 2-turn context as training (R1 turn + R2 turn) and read the LAST <GROUND>
    (the R2 readout, which attends to the R1 turn via causal history). Mirrors collate_g5 verbatim."""
    pil1, pil2 = _topil(r1_imgs), _topil(sel_imgs); n2 = len(pil2)
    c1 = [{"type": "image"} for _ in pil1] + [{"type": "text", "text": PROMPT_R1.format(nf=n1, d=desc)}]
    asst1 = [{"type": "text", "text": REG*n1 + f' Target: "{desc}". ' + TEMP + GROUND}]
    c2 = [{"type": "image"} for _ in pil2] + [{"type": "text", "text": PROMPT_R2.format(nf=n2, d=desc)}]
    text = proc.apply_chat_template([{"role": "user", "content": c1}, {"role": "assistant", "content": asst1},
                                     {"role": "user", "content": c2}], tokenize=False, add_generation_prompt=True)
    text = text + REG*n2 + f' Target: "{desc}". ' + GROUND
    enc = proc(text=[text], images=[pil1 + pil2], return_tensors="pt", padding=True).to("cuda")
    register = torch.from_numpy(np.concatenate([reg_r1.reshape(-1, 2048), reg_s.reshape(-1, 2048)], 0).astype(np.float32))[None].cuda()
    regfr = torch.cat([torch.arange(n1), torch.arange(n2)])[None].cuda()
    mm = {k: enc[k] for k in enc if k not in ("input_ids", "attention_mask")}
    temp_h, gh1, gh2, reg_h, rf = ag.encode(enc["input_ids"], enc["attention_mask"], register, regfr, multi_readout=True, **mm)
    intr, ext, dep = sample["intrinsics"][fsel], sample["extrinsics"][fsel], sample["depth_maps"][fsel]
    t = F.normalize(ag.text_proj(gh2.float()), dim=-1)                   # R2 <GROUND> reads with R1 history
    e = F.normalize(ag.inst_proj(torch.from_numpy(qemb).float().cuda()), dim=-1)
    sel = int((t @ e.t()).argmax(1))
    inst3d = map_pred_inst_to_gt_pointcloud(sample["points"], sample["super_points"], intr, ext, dep, masks_s, depth_tolerance=0.1)
    return {"inst3d": inst3d[sel], "all": (inst3d if DIAG else None), "sel": sel}


if __name__ == "__main__": main()
