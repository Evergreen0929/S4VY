#!/usr/bin/env python
"""End-to-end language grounding on a video benchmark, in one process.

Runs segmenter -> grounder -> critic cards -> critic without writing any intermediate
cache. Keeping the segmenter output in memory avoids a second scene forward pass that a
staged cards step would otherwise need.

Grounder entry points: grounder_infer._gen_mf_cot / mf_tree_windows / _agent_mframe /
_gen_mf_box. Candidates come from the query-matching stream (A) and the box stream (B);
the critic picks between them.
"""
import os, sys, json, glob
import _env
import cv2
import numpy as np
import torch

os.environ.setdefault("MFRAME", "1"); os.environ.setdefault("BOX_OUT", "1")
os.environ.setdefault("NO_INST_PROJ", "1"); os.environ.setdefault("COT", "1")
os.environ.setdefault("MF_COT", "1"); os.environ.setdefault("MF_COT_GEN", "1")
os.environ.setdefault("FIXEDQ", "1"); os.environ.setdefault("SEG_TAG", "s4vy")
os.environ.setdefault("MF_TREE", "1")

D = os.environ.get("GROUNDING_DATA_ROOT", "/PATH/TO/data")
DS = os.environ.get("DS", "refdavis"); SPLIT = os.environ.get("SPLIT", "valid")
OUT = _env.require("OUT", "output directory"); os.makedirs(OUT, exist_ok=True)
TMP = os.environ.get("E2E_TMP", f"/tmp/e2e_{DS}"); os.makedirs(TMP, exist_ok=True)
MAXF = int(os.environ.get("MAXF", "96")); LIMIT = int(os.environ.get("LIMIT", "0"))
SHARD = int(os.environ.get("SHARD", "0")); NSHARD = int(os.environ.get("NSHARD", "1"))
NO_CRITIC = os.environ.get("NO_CRITIC") == "1"
DUMP_PRED = os.environ.get("DUMP_PRED") == "1"
CRITIC = os.environ.get("CRITIC_CKPT", "/PATH/TO/checkpoints/agent/critic_2b/full_model.pt")
_cn = os.path.basename(os.path.dirname(CRITIC))
CFMT = (os.environ.get("CRITIC_FMT")
        or ("pairframe" if _cn.endswith("pf") else
            "card3" if "c3" in _cn else
            "perframe5" if "v31" in _cn else "card3"))
CK = (5 if CFMT in ("pairframe", "pfab") else
      int(CFMT[len("perframe"):]) if CFMT.startswith("perframe") else 1)

os.environ.setdefault("DS", DS); os.environ.setdefault("SPLIT", SPLIT)
import torch.nn.functional as F                                              # noqa: E402
from cache_video_ref import _sub, ADAPTERS, _MEVIS_MD
from cache_scene_masks import seg_forward_keepmasks
from cache_scene import _last, MH, MW
from align3d import build_omega                                           # noqa: E402
from grounder_infer import (_agent_mframe, _gen_mf_cot, _gen_mf_box, _gen_mf_box_batch,
                        mf_tree_windows, MF_NF)                              # noqa: E402
from agent_model import AgentGrounder                                        # noqa: E402
from vidref_data import load_gt, jf, up, video_frames
import cards_video as HCV
from cards_video import render, pick_frames, render_perframe, dump_masks  # noqa: E402
os.environ.setdefault("SRC", "/dev/null"); os.environ.setdefault("DST", "/tmp/_pf_unused")
from render_pairframe import pick as pf_pick, overlay as pf_overlay, MAG, CYA
from prompt_infer import load_model as load_prompt, IN_W, IN_H          # noqa: E402  s4vy_prompt
import vidbox as VBS                                                   # noqa: E402


def main():
    seg, _ = build_omega()
    seg.query_decoder.decoder_norm.register_forward_hook(lambda m, i, o: _last.__setitem__("q", o.detach()))
    seg.backbone.register_forward_hook(lambda m, i, o: _last.__setitem__("reg", o["tokens_list"]))
    pm = load_prompt()
    _sd = os.environ.get("SEG_DEV", "cuda")
    if _sd != "cuda":
        seg.to(_sd); pm.to(_sd); print(f"[dev] segmenter/prompt -> {_sd}, grounder/critic -> cuda:0", flush=True)
    ag = AgentGrounder(_env.require("MODEL_PATH", "base Qwen3-VL checkout"), lora=False, loss="matching").cuda().eval()
    _sd = torch.load(f"{os.environ['CKPT']}/full_model.pt", map_location="cpu")
    _r = ag.load_state_dict(_sd, strict=False)
    print(f"[ckpt] {len(_sd)} keys -> missing {len(_r.missing_keys)} unexpected {len(_r.unexpected_keys)}"
          f"{' first-3 missing ' + str(_r.missing_keys[:3]) if _r.missing_keys else ''}"
          f"{' first-3 unexpected ' + str(_r.unexpected_keys[:3]) if _r.unexpected_keys else ''}", flush=True)
    proc = ag.processor
    cm = None
    if not NO_CRITIC:
        from train_critic import Critic, PROMPT, PROMPT_PF, PROMPT_PF2, PROMPT_PF3
        _cb = os.environ.get("CRITIC_BASE") or f"{D}/../hf_models/Qwen3-VL-2B-Instruct"
        cm = Critic(_cb)
        _r = cm.load_state_dict(torch.load(CRITIC, map_location="cpu"), strict=False)
        print(f"[critic] {CRITIC.split('/')[-2]} fmt={CFMT} K={CK} -> "
              f"missing {len(_r.missing_keys)} unexpected {len(_r.unexpected_keys)}", flush=True)
        cm = cm.cuda().eval()
        cm.proc.tokenizer.padding_side = "right"
        from PIL import Image as _I

        @torch.no_grad()
        def _fwd(ims, txt):
            c = [{"type": "image"} for _ in ims] + [{"type": "text", "text": txt}]
            t = cm.proc.apply_chat_template([{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True)
            enc = cm.proc(text=[t], images=[ims], return_tensors="pt", padding=True).to("cuda")
            return float(cm(**dict(enc)))

        @torch.no_grad()
        def duel(pa, pb, desc):
            """pa/pb (card3  1 , perframe  K )
             —— "", """
            s = 0.0
            for first, second, sgn in ((pa, pb, +1), (pb, pa, -1)):
                ims = [_I.open(x).convert("RGB") for x in list(first) + list(second)]
                txt = (PROMPT_PF.format(k=CK, k1=CK + 1, k2=2 * CK, d=desc) if CK > 1
                       else PROMPT.format(d=desc))
                s += sgn * _fwd(ims, txt)
            return s

        @torch.no_grad()
        def duel_pf(P, Q, desc):
            """pairframe:  = ,  [ | 1  | 2 ]
            P  A /B , Q ; >0 """
            txt = (PROMPT_PF3 if CFMT == "pfab" else PROMPT_PF2).format(k=len(P), d=desc)
            return _fwd(P, txt) - _fwd(Q, txt)
    else:
        duel = duel_pf = None
    HCV.OUT = TMP; os.makedirs(f"{TMP}/img", exist_ok=True)
    os.makedirs(f"{OUT}/masks", exist_ok=True)

    def _dump(key, mA, mB, S, MH, MW):
        """dump_masks  HCV.OUT( TMP),  lustre  OUT,
         _raw_dump = dump_masks  UnboundLocalError"""
        _o = HCV.OUT; HCV.OUT = OUT
        try: dump_masks(key, mA, mB, S, MH, MW)
        finally: HCV.OUT = _o

    ds = ADAPTERS[DS]()
    if DS == "mevis":
        _MEVIS_MD[0] = json.load(open(f"{ds.root}/mask_dict.json"))
    vids = ds.videos()
    vids = [v for i, v in enumerate(vids) if i % NSHARD == SHARD]
    if LIMIT: vids = vids[:LIMIT]
    print(f"[e2e] {DS}/{SPLIT} {len(vids)} videos -> {OUT}  critic={'off' if NO_CRITIC else 'on'}", flush=True)

    for vi, vid in enumerate(vids):
        fo = f"{OUT}/{vid}.json"
        if os.path.exists(fo) and not (DUMP_PRED and not glob.glob(f"{OUT}/masks/{vid}__*.npz")):
            continue
        ex = ds.exprs(vid)
        if not ex: continue
        try:
            imgs, gtmap, fidx = ds.load(vid)
        except Exception as e:
            print(f"{vid}: LOAD FAIL {type(e).__name__}: {str(e)[:100]}", flush=True); continue
        S = imgs.shape[0]
        cimgs = imgs
        if not NO_CRITIC:
            try:
                cimgs = video_frames(vid, S)
            except Exception as e:
                print(f"{vid}: card frame source failed {type(e).__name__}: {e} -> falling back to raw frames, distribution differs from critic training", flush=True)
        # OFFLOAD=1 parks the grounder on the CPU while the critic runs (needed on a single GPU).
        _do_off = os.environ.get("OFFLOAD", "0") == "1" and os.environ.get("SEG_DEV", "cuda") == "cuda"
        if _do_off:
            ag.to("cpu"); torch.cuda.empty_cache()
        with torch.no_grad():
            masks_full, masks2d, scores, qemb, register = seg_forward_keepmasks(seg, imgs)
        if _do_off:
            ag.to("cuda")
        n = qemb.shape[0]
        if n == 0: print(f"{vid}: 0 instances, skipping", flush=True); continue
        e_norm = F.normalize(torch.from_numpy(qemb).float().cuda(), dim=-1)
        reg_all = register.astype(np.float32)
        rows = []
        for oid, descs in sorted(ex.items()):
            gt = load_gt(vid, fidx, oid)
            if gt is None or not gt.any(): continue
            for di, desc in enumerate(descs):
                cot = _gen_mf_cot(ag, proc, imgs[np.linspace(0, S - 1, min(S, MF_NF)).round().astype(int)], desc)
                wins, bxall = mf_tree_windows(np.arange(S), S, lambda ck: (
                    _gen_mf_box_batch(ag, proc, [reg_all[w] for w in ck], [imgs[w] for w in ck], desc, cot_text=cot)
                    if len(ck) > 1 else [_gen_mf_box(ag, proc, reg_all[ck[0]], imgs[ck[0]], desc, cot_text=cot)]))
                votes = np.zeros(n)
                for w in wins:
                    for idx, cf in _agent_mframe(ag, proc, reg_all[w], imgs[w], desc, e_norm, cot_text=cot):
                        if 0 <= idx < n: votes[idx] += cf
                sel = int(votes.argmax()) if votes.max() > 0 else -1
                mA = masks2d[sel][:len(gt)] if sel >= 0 else np.zeros((len(gt), MH, MW), bool)
                jA, fA = jf(up(mA, gt.shape[1:]), gt)
                items = VBS.boxes_to_items(wins, bxall)
                jB, fB, nb, mBox = VBS.score_box_path(pm, imgs, items, gt, IN_W, IN_H, up, jf,
                                                      tag=f"{vid}/{oid}", return_mask=True)
                rec = dict(video=vid, oid=int(oid), di=di, desc=desc, sel=sel,
                           A=round(((jA or 0.) + (fA or 0.)) / 2, 4), B=round((jB + fB) / 2, 4), n_box=nb,
                           A_J=round(jA or 0., 4), A_F=round(fA or 0., 4),
                           B_J=round(jB, 4), B_F=round(fB, 4))
                key = f"{vid}__{oid}__{di}"
                if mBox is None:
                    mB = np.zeros_like(mA)
                elif mBox.shape[1:] != mA.shape[1:]:
                    mB = np.stack([cv2.resize(m.astype(np.uint8), (mA.shape[2], mA.shape[1]),
                                              interpolation=cv2.INTER_NEAREST).astype(bool) for m in mBox])
                else:
                    mB = mBox.astype(bool)
                if len(mB) != len(mA):
                    t = np.zeros_like(mA); t[:min(len(t), len(mB))] = mB[:min(len(t), len(mB))]; mB = t
                if DUMP_PRED:
                    _dump(key, mA, mB, len(mA), mA.shape[1], mA.shape[2])
                if duel is not None and CFMT in ("pairframe", "pfab"):
                    _S = len(mA)
                    frs, _ = pf_pick(list(range(_S)), mA, mB, 8)
                    frs = [frs[i] for i in np.linspace(0, 7, 5).round().astype(int)]
                    P, Q = [], []
                    for f in frs:
                        im = cimgs[int(f)]
                        a = pf_overlay(im, mA[int(f)], MAG); b = pf_overlay(im, mB[int(f)], CYA)
                        if CFMT == "pfab":
                            P.append(_I.fromarray(np.concatenate([a, b], axis=1)))
                            Q.append(_I.fromarray(np.concatenate([b, a], axis=1)))
                        else:
                            P.append(_I.fromarray(np.concatenate([im, a, b], axis=1)))
                            Q.append(_I.fromarray(np.concatenate([im, b, a], axis=1)))
                    pick = "A" if duel_pf(P, Q, desc) > 0 else "B"
                    rec["pick"] = pick; rec["critic"] = rec[pick]
                elif duel is not None:
                    ps = {}
                    if CK > 1:
                        fr = pick_frames(mA, mB, len(mA), k=CK)
                        for nm, mm in (("A", mA), ("B", mB)):
                            render_perframe(cimgs, fr, mm, key, nm)
                            ps[nm] = [f"{TMP}/img/{key}__{nm}{j}.jpg" for j in range(CK)]
                    else:
                        for nm, mm in (("A", mA), ("B", mB)):
                            p = f"{TMP}/{key}__{nm}.jpg"
                            ar = mm.reshape(len(mm), -1).sum(1)
                            fs = sorted(int(x) for x in np.argsort(-ar)[:3] if ar[int(x)] > 0) or [0]
                            render(cimgs, fs, mm[fs], p); ps[nm] = [p]
                    pick = "A" if duel(ps["A"], ps["B"], desc) > 0 else "B"
                    rec["pick"] = pick; rec["critic"] = rec[pick]
                rows.append(rec)
        json.dump(rows, open(fo, "w"))
        print(f"[{vi+1}/{len(vids)}] {vid}: {len(rows)} expressions", flush=True)

    res = []
    for f in glob.glob(f"{OUT}/*.json"):
        try: res += json.load(open(f))
        except Exception: pass
    if not res: print("[e2e] no results"); return
    g = lambda k: np.mean([r[k] for r in res if k in r])
    print(f"\n=== {DS}/{SPLIT} end-to-end (no cache)  n={len(res)} ===")
    print(f"  A qemb {g('A'):.4f}   B box {g('B'):.4f}" + (f"   critic {g('critic'):.4f}" if not NO_CRITIC else ""))
    print("E2E_DONE", flush=True)


if __name__ == "__main__":
    main()
