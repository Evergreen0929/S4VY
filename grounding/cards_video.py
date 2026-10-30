#!/usr/bin/env python
"""videos split  critic cards (MeViS / LVOS / Ref-DAVIS train)


Env: DS(mevis|lvos|refdavis), SPLIT(train), CKPT, OUT, POOL(jsonl), SHARD/NSHARD, PROMPT_K(5), PICK(area)
"""
import os, sys, json, glob
import _env
os.environ.setdefault("MFRAME", "1"); os.environ.setdefault("BOX_OUT", "1")
os.environ.setdefault("NO_INST_PROJ", "1"); os.environ.setdefault("COT", "1")
os.environ.setdefault("MF_COT", "1"); os.environ.setdefault("MF_COT_GEN", "1")
os.environ.setdefault("FIXEDQ", "1"); os.environ.setdefault("SEG_TAG", "s4vy")
AG = os.environ.get("GROUNDING_REPO", os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, AG)
D = _env.data_root()
os.environ.setdefault("MODEL_PATH", os.environ.get("QWEN3_VL_2B", ""))
import numpy as np, torch, cv2
from PIL import Image
from grounder_infer import (_agent_mframe, _gen_mf_box, _gen_mf_box_batch, _gen_mf_cot, mf_tree_windows,
                        MF_NF, MF_OVERLAP, MF_COT, MF_COT_GEN, MF_TREE)
from agent_model import AgentGrounder
from prompt_infer import load_model, IN_W, IN_H
import torch.nn.functional as F

DS = os.environ.get("DS", "mevis"); SPLIT = os.environ.get("SPLIT", "train")
CACHE = os.environ.get("CACHE", f"{D}/vidref_cache_{DS}_{SPLIT}")
OUT = _env.require("OUT", "output directory for the rendered cards"); os.makedirs(f"{OUT}/img", exist_ok=True); os.makedirs(f"{OUT}/masks", exist_ok=True)
POOL = os.environ.get("POOL", "")
PK = int(os.environ.get("PROMPT_K", "5")); PICK = os.environ.get("PICK", "area")
FR_CAP = int(os.environ.get("FR_CAP", "48"))
SHARD = int(os.environ.get("SHARD", "0")); NSHARD = int(os.environ.get("NSHARD", "1"))
MAG = np.array([255, 0, 255], np.float32)

sys.path.insert(0, AG)
import vidref_data as EV

# Loaded lazily by main(). `render` is pure compositing, and the inference entry points import it
# for that alone -- building a second grounder and prompt model here would double their GPU use.
ag = proc = pm_model = None


def _load_models():
    global ag, proc, pm_model
    if ag is not None:
        return
    ag = AgentGrounder(_env.require("MODEL_PATH", "base Qwen3-VL checkout"), lora=False, loss="matching").cuda().eval()
    ag.load_state_dict(torch.load(f"{_env.require('CKPT', 'directory holding the trained grounder')}/full_model.pt",
                                  map_location="cpu"), strict=False)
    proc = ag.processor
    pm_model = load_model()                                # s4vy_prompt: box -> masklet


_MD = [None]


def _mevis_gt(vid, fidx):
    """MeViS  GT: meta_expressions  anno_id, mask_dict.json  RLE
    cache_video_ref.MeViS.load ,  cache  frame_idx  _sub"""
    from pycocotools import mask as mu
    root = f"{D}/RefVOS/mevis/{SPLIT}"
    if _MD[0] is None:
        _MD[0] = json.load(open(f"{root}/mask_dict.json"))
    try:
        meta = json.load(open(f"{root}/meta_expressions.json"))["videos"][vid]
    except Exception:
        return {}
    md = _MD[0]; out = {}
    for e in meta["expressions"].values():
        if len(e.get("obj_id", [])) != 1: continue
        oid = int(e["obj_id"][0]); aid = str(e["anno_id"][0])
        if oid in out or aid not in md: continue
        rles = md[aid]
        fr = [int(i) for i in fidx]
        m = []
        for i in fr:
            if i < len(rles) and rles[i]: m.append(mu.decode(rles[i]).astype(bool))
            else: m.append(None)
        sh = next((x.shape for x in m if x is not None), None)
        if sh is None: continue
        out[oid] = np.stack([(x if x is not None else np.zeros(sh, bool)) for x in m])
    return out


def render(imgs, fs, mask, path):
    """ prompt_infer.render_card , (videos SC2D )"""
    area = mask.reshape(len(fs), -1).sum(1)
    top = np.argsort(-area)[:3]
    if area[top[0]] == 0: top = np.linspace(0, len(fs) - 1, min(3, len(fs))).astype(int)
    rows = []
    for ti in top:
        im = imgs[int(fs[int(ti)])]
        m = cv2.resize(mask[int(ti)].astype(np.float32), (im.shape[1], im.shape[0]),
                       interpolation=cv2.INTER_LINEAR) > 0.5
        o = im.astype(np.float32).copy()
        o[m] = o[m] * 0.6 + MAG * 0.4
        cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(o, cont, -1, tuple(int(c) for c in MAG), 2)
        rows.append(np.concatenate([im, o.astype(np.uint8)], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path, quality=90)
    return [int(fs[int(t)]) for t in top]


PERFRAME = os.environ.get("PERFRAME", "0") == "1"
NFRAME = int(os.environ.get("NFRAME", "5"))


def pick_frames(mA, mB, S, k=None):
    """A/B :  masklet ,  k

    """
    k = k or NFRAME
    aA = mA.reshape(S, -1).any(1); aB = mB.reshape(S, -1).any(1)
    cand = np.where(aA & aB)[0]
    if len(cand) < k: cand = np.where(aA | aB)[0]
    if len(cand) < k: cand = np.arange(S)
    if len(cand) < k: return [int(cand[min(i, len(cand) - 1)]) for i in range(k)]
    return [int(cand[i]) for i in np.linspace(0, len(cand) - 1, k).round().astype(int)]


def render_perframe(imgs, frames, mask, key, tag):
    """ [ | ],

    """
    out = []
    for j, f in enumerate(frames):
        im = imgs[int(f)]
        m = cv2.resize(mask[int(f)].astype(np.float32), (im.shape[1], im.shape[0]),
                       interpolation=cv2.INTER_LINEAR) > 0.5
        o = im.astype(np.float32).copy()
        o[m] = o[m] * 0.6 + MAG * 0.4
        cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(o, cont, -1, tuple(int(c) for c in MAG), 2)
        Image.fromarray(np.concatenate([im, o.astype(np.uint8)], axis=1)).save(
            f"{OUT}/img/{key}__{tag}{j}.jpg", quality=90)
        out.append(int(f))
    return out


def dump_masks(key, mA, mB, S, MH, MW):
    """ masklet  IoU  3 , ;
    /,  GPU"""
    np.savez_compressed(f"{OUT}/masks/{key}.npz",
                        A=np.packbits(mA.reshape(S, -1), axis=1),
                        B=np.packbits(mB.reshape(S, -1), axis=1),
                        shape=np.array([S, MH, MW], np.int32))


def pick_boxes(items):
    if PK <= 0 or len(items) <= PK: return items
    if PICK == "area":
        a = [(it[3] - it[1]) * (it[4] - it[2]) for it in items]
        return [items[int(i)] for i in sorted(np.argsort(-np.asarray(a))[:PK])]
    return [items[i] for i in np.linspace(0, len(items) - 1, PK).round().astype(int)]


@torch.no_grad()
def main():
    _load_models()
    want = None
    if POOL:
        want = {}
        for l in open(POOL):
            r = json.loads(l); want.setdefault(r["video"], []).append(r)
    vids = sorted(want) if want else sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{CACHE}/*.npz"))
    vids = [v for i, v in enumerate(vids) if i % NSHARD == SHARD]
    meta = open(f"{OUT}/meta_{SHARD:02d}.jsonl", "a")
    print(f"[vidcards] {DS}/{SPLIT} {len(vids)} videos MF_TREE={int(MF_TREE)} PICK={PICK} K={PK}", flush=True)
    for vi, vid in enumerate(vids):
        try:
            z = np.load(f"{CACHE}/{vid}.npz", allow_pickle=True)
            rows = json.loads(str(z["rows"]))
            n, S, MH, MW = [int(x) for x in z["pred_shape"]]
            pmask = np.unpackbits(z["pred_masks"], axis=1)[:, :S * MH * MW].reshape(n, S, MH, MW).astype(bool)
            reg_all = z["register"].astype(np.float32); fidx = z["frame_idx"]
            qemb = z["qemb"].astype(np.float32)
            imgs = EV.video_frames(vid, S)
            e_norm = F.normalize(torch.from_numpy(qemb).float().cuda(), dim=-1)
        except Exception as e:
            print(f"{vid}: LOAD FAIL {type(e).__name__}: {e}", flush=True); continue
        stride = max(1, MF_NF - MF_OVERLAP)
        starts = list(range(0, max(1, S - MF_NF + 1), stride))
        if S > MF_NF and starts[-1] != S - MF_NF: starts.append(S - MF_NF)
        gtmap = None
        if DS == "mevis":
            gtmap = _mevis_gt(vid, fidx)
            if not gtmap: print(f"{vid}: no MeVIS GT, skipping", flush=True); continue
        gps = EV.gt_paths(vid) if DS != "mevis" else None
        for r in rows:
            oid = int(r["oid"])
            if DS == "mevis":
                if oid not in gtmap: continue
                gt = gtmap[oid]
            else:
                gl = []
                for i in fidx:
                    if int(i) >= len(gps): break
                    gl.append(np.array(Image.open(gps[int(i)])) == oid)
                if not gl: continue
                gt = np.stack(gl)
            if not gt.any(): continue
            H, W = gt.shape[1:]
            if want:
                descs = sorted(((int(x["di"]), x["desc"]) for x in want.get(vid, []) if int(x["oid"]) == oid))
            else:
                descs = list(enumerate(r.get("descs") or []))
            for di, desc in descs:
                key = f"{DS}_{vid}__{oid}__{di}"
                _sent = f"{OUT}/img/{key}__B{NFRAME-1}.jpg" if PERFRAME else f"{OUT}/img/{key}__B.jpg"
                if os.path.exists(_sent): continue
                cot = ""
                if MF_COT and MF_COT_GEN:
                    gw = np.linspace(0, S - 1, min(S, MF_NF)).round().astype(int)
                    cot = _gen_mf_cot(ag, proc, imgs[gw], desc)
                votes = np.zeros(n)
                if MF_TREE:
                    _w, _h = mf_tree_windows(np.arange(S), S, lambda ck: [
                        [(t if t[0] >= 0 else None)
                         for t in _agent_mframe(ag, proc, reg_all[w], imgs[w], desc, e_norm, cot_text=cot)]
                        for w in ck])
                    for hh in _h:
                        for t in hh:
                            if t is not None and 0 <= t[0] < n: votes[t[0]] += t[1]
                else:
                    for s0 in starts:
                        w = np.arange(s0, min(s0 + MF_NF, S))
                        for idx, cf in _agent_mframe(ag, proc, reg_all[w], imgs[w], desc, e_norm, cot_text=cot):
                            if 0 <= idx < n: votes[idx] += cf
                _pe = next((x for x in want.get(vid, []) if int(x["oid"]) == oid and int(x["di"]) == di), None) if want else None
                if _pe is not None and _pe.get("sel") is not None:
                    selA = int(_pe["sel"])
                else:
                    selA = int(votes.argmax()) if votes.max() > 0 else -1
                mA = pmask[selA] if selA >= 0 else np.zeros((S, MH, MW), bool)
                bxwin = {}
                if MF_TREE:
                    _wins, _bxall = mf_tree_windows(np.arange(S), S, lambda ck: (
                        _gen_mf_box_batch(ag, proc, [reg_all[w] for w in ck], [imgs[w] for w in ck], desc, cot_text=cot)
                        if len(ck) > 1 else
                        [_gen_mf_box(ag, proc, reg_all[ck[0]], imgs[ck[0]], desc, cot_text=cot)]))
                else:
                    _wins = [np.arange(s0, min(s0 + MF_NF, S)) for s0 in starts]
                    _bxall = [_gen_mf_box(ag, proc, reg_all[w], imgs[w], desc, cot_text=cot) for w in _wins]
                for w, bxs in zip(_wins, _bxall):
                    for j, f in enumerate(w):
                        bxwin.setdefault(int(f), []).append(bxs[j] if j < len(bxs) else None)
                items = []
                for f in sorted(bxwin):
                    bs = [b for b in bxwin[f] if b is not None]
                    if not bs: continue
                    items.append([int(f)] + [float(np.median([b[k] for b in bs])) for k in range(4)])
                items = [it for it in items if (it[3] - it[1]) > 1e-3 and (it[4] - it[2]) > 1e-3]
                mB = np.zeros((S, MH, MW), bool)
                if items:
                    prompts = pick_boxes(items)
                    pf = sorted({int(it[0]) for it in prompts})
                    rest = [f for f in range(S) if f not in set(pf)]
                    room = max(1, FR_CAP - len(pf))
                    for ch in ([rest[i:i + room] for i in range(0, len(rest), room)] or [[]]):
                        fs = sorted(set(pf) | set(ch)); loc = {fr: i for i, fr in enumerate(fs)}
                        im = torch.stack([torch.from_numpy(np.asarray(
                                Image.fromarray(imgs[fr]).resize((IN_W, IN_H)), dtype=np.float32) / 255.
                             ).permute(2, 0, 1) for fr in fs]).unsqueeze(0).cuda()
                        try:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                pr = pm_model(im, prompts=[[{"points": [], "boxes": [
                                    (loc[int(it[0])], *[float(x) for x in it[1:5]]) for it in prompts]}]])
                            mk = (pr["prompt_mask_logits"][0, 0].float() > 0).cpu().numpy()
                        except Exception as e:
                            print(f"{vid} {key}: PROMPT FAIL {type(e).__name__}: {e}", flush=True); break
                        for i, fr in enumerate(fs):
                            if fr < S:
                                mB[fr] |= cv2.resize(mk[i].astype(np.float32), (MW, MH)) > .5
                jA, fA = EV.jf(EV.up(mA, (H, W)), gt)
                jB, fB = EV.jf(EV.up(mB, (H, W)), gt)
                sA = ((jA or 0.) + (fA or 0.)) / 2; sB = ((jB or 0.) + (fB or 0.)) / 2
                if PERFRAME:
                    dump_masks(key, mA, mB, S, MH, MW)
                    frs = pick_frames(mA, mB, S)
                    vA = render_perframe(imgs, frs, mA, key, "A")
                    vB = render_perframe(imgs, frs, mB, key, "B")
                    _n_common = int((mA.reshape(S, -1).any(1) & mB.reshape(S, -1).any(1)).sum())
                else:
                    vA = render(imgs, np.arange(S), mA, f"{OUT}/img/{key}__A.jpg")
                    vB = render(imgs, np.arange(S), mB, f"{OUT}/img/{key}__B.jpg")
                    _n_common = -1
                meta.write(json.dumps({
                    "key": key, "scene": vid, "oid": oid, "ann": di, "ds": DS, "cat": "",
                    "desc": desc,
                    "A_biou": round(sA, 4), "A_miou": round(jA or 0., 4),
                    "B_biou": round(sB, 4), "B_miou": round(jB or 0., 4),
                    "metric": "jf2d",
                    "fmt": (f"perframe{NFRAME}" if PERFRAME else "card3"),
                    "S": int(S), "n_common": _n_common,
                    "label": "A" if sA >= sB else "B", "margin": round(abs(sA - sB), 4),
                    "viewsA": vA, "viewsB": vB}) + "\n"); meta.flush()
        print(f"[{vi+1}/{len(vids)}] {vid}", flush=True)
    meta.close()


if __name__ == "__main__": main()
