#!/usr/bin/env python
"""End-to-end language grounding on a static ScanNet++ scene, in one process.

Runs segmenter -> grounder -> 2D-to-3D lift -> critic cards -> critic with no intermediate
cache, so the segmenter forward happens once.

Reports mask and box agreement for the query stream (qemb), the box stream, and the
critic's selection.
"""
import os, sys, json, glob
import _env
import numpy as np
import torch
import cv2

os.environ.setdefault("MFRAME", "1"); os.environ.setdefault("BOX_OUT", "1")
os.environ.setdefault("NO_INST_PROJ", "1"); os.environ.setdefault("COT", "1")
os.environ.setdefault("MF_COT", "1"); os.environ.setdefault("MF_COT_GEN", "1")
os.environ.setdefault("FIXEDQ", "1"); os.environ.setdefault("SEG_TAG", "s4vy")
os.environ.setdefault("MF_TREE", "1"); os.environ.setdefault("PERFRAME", "1")

D = os.environ.get("GROUNDING_DATA_ROOT", "/PATH/TO/data")
DL = os.environ.get("GROUNDING_DATA_LOG_ROOT", f"{D}/../data_log")
os.environ.setdefault("MODEL_PATH", os.environ.get("QWEN3_VL_2B", ""))
OUT = _env.require("OUT", "output directory"); os.makedirs(OUT, exist_ok=True)
TMP = os.environ.get("E2E_TMP", "/tmp/e2e_spp"); os.makedirs(f"{TMP}/img", exist_ok=True)
EVAL = os.environ.get("SPP_EVAL", f"{DL}/scannetpp_eval_filt.jsonl")
LIMIT = int(os.environ.get("LIMIT", "0"))
SHARD = int(os.environ.get("SHARD", "0")); NSHARD = int(os.environ.get("NSHARD", "1"))
NO_CRITIC = os.environ.get("NO_CRITIC") == "1"
DUMP_PRED = os.environ.get("DUMP_PRED") == "1"
TOL = float(os.environ.get("TOL", "0.1"))
CRITIC = os.environ.get("CRITIC_CKPT", "/PATH/TO/checkpoints/agent/critic_2b/full_model.pt")
_cn = os.path.basename(os.path.dirname(CRITIC))
CFMT = (os.environ.get("CRITIC_FMT")
        or ("pairframe" if _cn.endswith("pf") else
            "card3" if "c3" in _cn else
            "perframe5" if "v31" in _cn else "card3"))
CK = (5 if CFMT in ("pairframe", "pfab") else
      int(CFMT[len("perframe"):]) if CFMT.startswith("perframe") else 1)

import torch.nn.functional as F                                              # noqa: E402
from cache_scene_masks import seg_forward_keepmasks                         # noqa: E402
from cache_scene import load_scene, _last                                # noqa: E402
from align3d import build_omega, map_pred_inst_to_gt_pointcloud           # noqa: E402
from grounder_infer import (_agent_mframe, _gen_mf_cot, _gen_mf_box, mf_tree_windows, MF_NF)   # noqa: E402
from agent_model import AgentGrounder                                        # noqa: E402
from prompt_infer import load_model as load_prompt, IN_W, IN_H          # noqa: E402
import vidbox as VBS                                                   # noqa: E402
import perframe_cards as PF                                                  # noqa: E402
from PIL import Image                                                        # noqa: E402

SPPP_2D = f"{D}/scannetpp_scenes_2d"
MAG = np.array([255, 0, 255], np.float32)


def scene_frames(sc, idx):
    """cards critic train(cards_scene.scene_frames ): 512  color/
     load_scene  ->  token  -> distribution differs from critic training"""
    out = []
    for k in idx:
        im = Image.open(f"{SPPP_2D}/{sc}/color/{int(k)}.jpg").convert("RGB")
        if im.width > 512: im = im.resize((512, int(im.height * 512 / im.width)))
        out.append(np.asarray(im))
    return np.stack(out)


def render_card3(imgs, fs, mask, path):
    """ cards_scene.render ( prompt_infer.render_card ):
    critic2b_v3 , train"""
    area = mask.reshape(len(fs), -1).sum(1)
    top = np.argsort(-area)[:3]
    if area[top[0]] == 0: top = np.linspace(0, len(fs) - 1, min(3, len(fs))).astype(int)
    rows = []
    for ti in top:
        im = imgs[int(ti)]
        m = cv2.resize(mask[int(ti)].astype(np.float32), (im.shape[1], im.shape[0]),
                       interpolation=cv2.INTER_LINEAR) > 0.5
        o = im.astype(np.float32).copy()
        o[m] = o[m] * 0.6 + MAG * 0.4
        cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(o, cont, -1, tuple(int(c) for c in MAG), 2)
        rows.append(np.concatenate([im, o.astype(np.uint8)], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path, quality=90)
    return [int(fs[int(t)]) for t in top]


def aabb_iou(a, b):
    """ eval_scannetpp.aabb_iou """
    if not len(a) or not len(b): return 0.0
    a0, a1 = a.min(0), a.max(0); b0, b1 = b.min(0), b.max(0)
    lo = np.maximum(a0, b0); hi = np.minimum(a1, b1)
    inter = float(np.prod(np.clip(hi - lo, 0, None)))
    va = float(np.prod(np.clip(a1 - a0, 0, None))); vb = float(np.prod(np.clip(b1 - b0, 0, None)))
    u = va + vb - inter
    return inter / u if u > 0 else 0.0


def main():
    seg, _ = build_omega()
    seg.query_decoder.decoder_norm.register_forward_hook(lambda m, i, o: _last.__setitem__("q", o.detach()))
    seg.backbone.register_forward_hook(lambda m, i, o: _last.__setitem__("reg", o["tokens_list"]))
    pm = load_prompt()
    _sd_dev = os.environ.get("SEG_DEV", "cuda")
    if _sd_dev != "cuda":
        seg.to(_sd_dev); pm.to(_sd_dev)
        print(f"[dev] segmenter/prompt -> {_sd_dev}, grounder/critic -> cuda:0", flush=True)
    ag = AgentGrounder(_env.require("MODEL_PATH", "base Qwen3-VL checkout"), lora=False, loss="matching").cuda().eval()
    _sd = torch.load(f"{os.environ['CKPT']}/full_model.pt", map_location="cpu")
    _r = ag.load_state_dict(_sd, strict=False)
    print(f"[ckpt] {len(_sd)} keys -> missing {len(_r.missing_keys)} unexpected {len(_r.unexpected_keys)}", flush=True)
    proc = ag.processor

    duel = None
    if not NO_CRITIC:
        from train_critic import Critic, PROMPT, PROMPT_PF
        _cb = os.environ.get("CRITIC_BASE") or f"{D}/../hf_models/Qwen3-VL-2B-Instruct"
        cm = Critic(_cb)
        _r = cm.load_state_dict(torch.load(CRITIC, map_location="cpu"), strict=False)
        print(f"[critic] {_cn} fmt={CFMT} K={CK} -> missing {len(_r.missing_keys)} unexpected {len(_r.unexpected_keys)}", flush=True)
        cm = cm.cuda().eval()
        cm.proc.tokenizer.padding_side = "right"

        @torch.no_grad()
        def _fwd(ims, txt):
            c = [{"type": "image"} for _ in ims] + [{"type": "text", "text": txt}]
            t = cm.proc.apply_chat_template([{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True)
            enc = cm.proc(text=[t], images=[ims], return_tensors="pt", padding=True).to("cuda")
            return float(cm(**dict(enc)))

        @torch.no_grad()
        def duel(pa, pb, desc):
            """ —— "", """
            s = 0.0
            for first, second, sgn in ((pa, pb, +1), (pb, pa, -1)):
                ims = [Image.open(x).convert("RGB") for x in list(first) + list(second)]
                txt = (PROMPT_PF.format(k=CK, k1=CK + 1, k2=2 * CK, d=desc) if CK > 1
                       else PROMPT.format(d=desc))
                s += sgn * _fwd(ims, txt)
            return s

    rows = [json.loads(l) for l in open(EVAL)]
    if os.environ.get("SCENE_LIST"):
        keep = set(s.strip() for s in open(os.environ["SCENE_LIST"]) if s.strip())
        rows = [r for r in rows if r["scene"] in keep]
    by = {}
    for r in rows: by.setdefault(r["scene"], []).append(r)
    scenes = sorted(by)
    scenes = [s for i, s in enumerate(scenes) if i % NSHARD == SHARD]
    if LIMIT: scenes = scenes[:LIMIT]
    print(f"[e2e-spp] {sum(len(by[s]) for s in scenes)} query / {len(scenes)} scenes -> {OUT}"
          f"  critic={'off' if NO_CRITIC else 'on'}", flush=True)

    for si, sc in enumerate(scenes):
        fo = f"{OUT}/{sc}.json"
        if os.path.exists(fo): continue
        try:
            geo = load_scene(sc)
        except Exception as e:
            print(f"{sc}: LOAD FAIL {type(e).__name__}: {str(e)[:100]}", flush=True); continue
        imgs = geo["images"]; S = len(imgs)
        pts = np.asarray(geo["points"]); instmask = np.asarray(geo["instmask"])
        try:
            cimgs = scene_frames(sc, range(S))
        except Exception as e:
            print(f"{sc}: card frame source failed {type(e).__name__}: {e} -> falling back to raw frames, distribution differs from critic training", flush=True)
            cimgs = imgs
        with torch.no_grad():
            masks_full, masks2d, scores, qemb, register = seg_forward_keepmasks(seg, imgs)
        n = qemb.shape[0]
        if n == 0: print(f"{sc}: 0 instances, skipping", flush=True); continue
        e_norm = F.normalize(torch.from_numpy(qemb).float().cuda(), dim=-1)
        reg_all = register.astype(np.float32)
        _all = np.arange(S, dtype=int)
        _dummy_gt = np.zeros((S, 1, 1), bool)

        def lift3d(m2d):
            """2D masklet -> 3D mask cards_scene : ,
            IoU ( 0.2229 vs 0.2563)"""
            if m2d is None or not m2d.any(): return np.zeros(len(pts), bool)
            return map_pred_inst_to_gt_pointcloud(
                geo["points"], geo["super_points"], geo["intrinsics"][_all], geo["extrinsics"][_all],
                geo["depth_maps"][_all], m2d[_all][None].astype(np.float32),
                depth_tolerance=TOL).astype(bool)[0]

        out_rows = []
        for _qi, q in enumerate(by[sc]):
            oid = int(q["oid"]); desc = q["desc"]
            gt3 = instmask == oid
            if gt3.sum() == 0: continue
            gw = np.linspace(0, S - 1, min(S, MF_NF)).round().astype(int)
            cot = _gen_mf_cot(ag, proc, imgs[gw], desc)
            votes = np.zeros(n); qhits = []

            def _g(ck):
                out = []
                for w in ck:
                    qhits.append(_agent_mframe(ag, proc, reg_all[w], imgs[w], desc, e_norm, cot_text=cot))
                    out.append(_gen_mf_box(ag, proc, reg_all[w], imgs[w], desc, cot_text=cot))
                return out

            wins, bxall = mf_tree_windows(_all, S, _g)
            for _q in qhits:
                for idx, cf in _q:
                    if 0 <= idx < n: votes[idx] += cf
            sel = int(votes.argmax()) if votes.max() > 0 else -1

            mA = masks2d[sel] if sel >= 0 else np.zeros_like(masks2d[0])
            liftA = lift3d(mA) if sel >= 0 else np.zeros(len(pts), bool)
            items = VBS.boxes_to_items(wins, bxall)
            _, _, nb, mBox = VBS.score_box_path(pm, imgs, items, _dummy_gt, IN_W, IN_H,
                                                up=lambda m, shp: m, jf=lambda a, b: (0.0, 0.0),
                                                tag=f"{sc}/{oid}", return_mask=True)
            mB = mBox.astype(bool) if mBox is not None else np.zeros_like(mA)
            if mB.shape != mA.shape:
                t = np.zeros_like(mA); k = min(len(t), len(mB)); t[:k] = mB[:k]; mB = t
            liftB = lift3d(mB)

            iou = lambda l: (float((l & gt3).sum()) / max(float((l | gt3).sum()), 1.0),
                             aabb_iou(pts[l], pts[gt3]))
            mIA, bA = iou(liftA); mIB, bB = iou(liftB)
            rec = dict(scene=sc, oid=oid, ann=q.get("ann", 0), desc=desc, name=q.get("name"),
                       sel=sel, n_box=nb, best_iou2d=q.get("best_iou2d"),
                       A_miou=round(mIA, 4), A_biou=round(bA, 4),
                       B_miou=round(mIB, 4), B_biou=round(bB, 4))
            key = f"{sc}__{oid}__{q.get('ann', 0)}"
            mA_by = {int(f): mA[int(f)] for f in range(len(mA))}
            mB_by = {int(f): mB[int(f)] for f in range(len(mB))}
            if DUMP_PRED: PF.dump_masks(OUT, key, mA_by, mB_by)
            if duel is not None:
                ps = {}
                if CK > 1:
                    frs = PF.pick_frames(mA_by, mB_by, k=CK)
                    for nm, mm in (("A", mA_by), ("B", mB_by)):
                        PF.render_perframe(lambda f: cimgs[int(f)], frs, mm, TMP, key, nm)
                        ps[nm] = [f"{TMP}/img/{key}__{nm}{j}.jpg" for j in range(len(frs))]
                else:
                    for nm, mm in (("A", mA), ("B", mB)):
                        ar = mm.reshape(len(mm), -1).sum(1)
                        fsx = sorted(int(x) for x in np.argsort(-ar)[:3] if ar[int(x)] > 0) or [0]
                        fnx = np.array(fsx, int)
                        pth = f"{TMP}/img/{key}__{nm}.jpg"
                        render_card3(cimgs[fnx], fsx, mm[fnx], pth)
                        ps[nm] = [pth]
                pick = "A" if duel(ps["A"], ps["B"], desc) > 0 else "B"
                rec["pick"] = pick
                rec["critic_miou"] = rec[f"{pick}_miou"]; rec["critic_biou"] = rec[f"{pick}_biou"]
            out_rows.append(rec)
            if (_qi + 1) % 5 == 0 or _qi == 0:
                print(f"  [{si+1}/{len(scenes)}] {sc} query {_qi+1}/{len(by[sc])}", flush=True)
        json.dump(out_rows, open(fo, "w"))
        print(f"[{si+1}/{len(scenes)}] {sc}: {len(out_rows)} query", flush=True)

    res = []
    for f in glob.glob(f"{OUT}/*.json"):
        try: res += json.load(open(f))
        except Exception: pass
    if not res: print("[e2e-spp] no results"); return
    g = lambda k: float(np.mean([r[k] for r in res if k in r]))
    print(f"\n=== ScanNet++ end-to-end (no cache)  n={len(res)} ===")
    print(f"  qemb  mask {g('A_miou'):.4f}  box {g('A_biou'):.4f}")
    print(f"  box    mask {g('B_miou'):.4f}  box {g('B_biou'):.4f}")
    if not NO_CRITIC:
        print(f"  +critic mask {g('critic_miou'):.4f}  box {g('critic_biou'):.4f}"
              f"   picked A {100*np.mean([r['pick']=='A' for r in res if 'pick' in r]):.0f}%")
    print("E2E_SPP_DONE", flush=True)


if __name__ == "__main__":
    main()
