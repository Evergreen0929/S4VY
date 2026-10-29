#!/usr/bin/env python
"""End-to-end language-grounding demo: video + referring expression -> grounded masklet + overlay mp4.

Pipeline (one forward, no cache) — mirrors grounding/run_pipeline_video.py:
  frozen segmenter (s4vy) -> per-frame instances + query embeddings + registers
      -> grounder (Qwen3-VL): CoT -> box-driven tree windows
            A (qemb): per-frame <GROUND> matches a s4vy query -> vote -> selected instance masklet
            B (box) : text boxes -> s4vy_prompt prompt encoder -> masklet
      -> critic (2B): picks A or B  ->  final masklet

Weights auto-download from HF (jdzhang0929/S4VY): segmenter + prompt encoder, the 4B grounder and
the 2B critic. The Qwen3-VL base VLMs are NOT redistributed here; point --base at a local or HF
Qwen3-VL-4B-Instruct and --critic_base at a Qwen3-VL-2B-Instruct. Requires the VGGT-Omega backbone ckpt
(VGGT_OMEGA_CKPT) and the grounding/ code (imported below).

Usage:
  python demo/grounding_video.py --video clip.mp4 --expr "the person in red" --out out/clip \
         --base /path/to/Qwen3-VL-4B-Instruct

Env expected by the imported grounding code: VGGT_OMEGA_ROOT, VGGT_OMEGA_CKPT.
"""
import os, sys, argparse
# fixed grounding regime (same switches as run_pipeline_video.py)
for k, v in dict(MFRAME="1", BOX_OUT="1", NO_INST_PROJ="1", COT="1", MF_COT="1", MF_COT_GEN="1",
                 FIXEDQ="1", SEG_TAG="s4vy", MF_TREE="1", SEG_NO_FRAMEPOS="1").items():
    os.environ.setdefault(k, v)
import numpy as np, cv2, torch, torch.nn.functional as F
from huggingface_hub import hf_hub_download

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# reuse the grounding primitives; GROUNDING_SRC lets you point at a configured checkout
sys.path.insert(0, os.environ.get("GROUNDING_SRC", os.path.join(REPO, "grounding")))
HF = "jdzhang0929/S4VY"
GROUNDER_CKPT = "grounder/full_model.pt"       # the 4B dual-stream grounder
CRITIC_CKPT = "critic/full_model.pt"          # the 2B critic

_rng = np.random.RandomState(0)
PAL = ((_rng.rand(1024, 3) * 0.7 + 0.3) * 255).astype(np.uint8); PAL[0] = 0


def load_frames(path, maxf, long_side=512):
    """-> (S, H, W, 3) uint8 RGB (same format the grounding code expects; it resizes internally)."""
    cap = cv2.VideoCapture(path)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    want = sorted(set(int(i) for i in np.unique(np.linspace(0, max(tot - 1, 0), min(maxf, tot or maxf)).round().astype(int))))
    ws = set(want); frames = {}; i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in ws:
            frames[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        i += 1
        if len(frames) == len(ws):
            break
    cap.release()
    ims = np.stack([frames[k] for k in sorted(frames)])
    h0, w0 = ims.shape[1:3]
    if max(h0, w0) > long_side:                              # cap size, keep aspect
        Wr, Hr = (long_side, round(h0 / w0 * long_side)) if w0 >= h0 else (round(w0 / h0 * long_side), long_side)
        ims = np.stack([cv2.resize(f, (Wr, Hr), interpolation=cv2.INTER_AREA) for f in ims])
    return ims


def overlay_mp4(origs, idmap, out_path, fps=8.0):
    Ho, Wo = origs.shape[1:3]
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (Wo, Ho))
    for f in range(len(origs)):
        o = origs[f].astype(np.float32).copy()
        m = idmap[f].astype(bool)
        if m.any():
            o[m] = 0.5 * origs[f][m] + 0.5 * PAL[1]
        vw.write(cv2.cvtColor(o.astype(np.uint8), cv2.COLOR_RGB2BGR))
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--expr", required=True, help="referring expression")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", required=True, help="path or id of the Qwen3-VL-4B-Instruct base for the grounder")
    ap.add_argument("--critic_base", default=None, help="Qwen3-VL-2B for the critic (default: --base if size 2b else required)")
    ap.add_argument("--maxf", type=int, default=int(os.environ.get("MAXF", "96")))
    ap.add_argument("--no_critic", action="store_true")
    args = ap.parse_args()

    # weights from HF (segmenter/prompt are self-contained; grounder/critic need their base VLM)
    os.environ["INFER_CKPT"] = hf_hub_download(HF, "segmenter/s4vy.pt")
    _pe = hf_hub_download(HF, "segmenter/s4vy_prompt.pt")
    os.environ["MODEL_PATH"] = args.base
    grounder_ckpt = hf_hub_download(HF, GROUNDER_CKPT)
    os.environ["CKPT"] = os.path.dirname(grounder_ckpt)       # grounder_infer reads CKPT (dir holding full_model.pt) at import
    # the box-path prompt model = segmenter (backbone+LoRA+decoder+head) + prompt encoder; merge them for the loader
    import tempfile
    _pf = os.path.join(tempfile.gettempdir(), "s4vy_prompt_full.pt")
    if not os.path.exists(_pf):
        torch.save({"model": {**torch.load(os.environ["INFER_CKPT"], map_location="cpu")["model"],
                              **torch.load(_pe, map_location="cpu")["model"]}}, _pf)
    os.environ["PROMPT_CKPT"] = _pf

    from align3d import build_omega
    from cache_scene_masks import seg_forward_keepmasks
    from cache_scene import _last, MH, MW
    from grounder_infer import (_agent_mframe, _gen_mf_cot, _gen_mf_box, _gen_mf_box_batch, mf_tree_windows, MF_NF)
    from agent_model import AgentGrounder
    from prompt_infer import load_model as load_prompt, IN_W, IN_H
    import vidbox as VBS

    # ---- build the four models once
    seg, _ = build_omega()
    seg.query_decoder.decoder_norm.register_forward_hook(lambda m, i, o: _last.__setitem__("q", o.detach()))
    seg.backbone.register_forward_hook(lambda m, i, o: _last.__setitem__("reg", o["tokens_list"]))
    pm = load_prompt()
    ag = AgentGrounder(args.base, lora=False, loss="matching").cuda().eval()
    r = ag.load_state_dict(torch.load(grounder_ckpt, map_location="cpu"), strict=False)
    print(f"[grounder] missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)}", flush=True)
    proc = ag.processor

    critic = None
    if not args.no_critic:
        from train_critic import Critic, PROMPT
        cbase = args.critic_base
        assert cbase, "--critic_base (a Qwen3-VL-2B-Instruct checkout) is required"
        critic = Critic(cbase)
        critic.load_state_dict(torch.load(hf_hub_download(HF, CRITIC_CKPT), map_location="cpu"), strict=False)
        critic = critic.cuda().eval(); critic.proc.tokenizer.padding_side = "right"
        os.environ["OUT"] = os.path.join(args.out, "_cards"); os.makedirs(os.environ["OUT"], exist_ok=True)
        from cards_video import render          # module reads OUT/MODEL_PATH/CKPT at import
        from PIL import Image as _I

        @torch.no_grad()
        def _fwd(ims, txt):
            c = [{"type": "image"} for _ in ims] + [{"type": "text", "text": txt}]
            t = critic.proc.apply_chat_template([{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True)
            enc = critic.proc(text=[t], images=[ims], return_tensors="pt", padding=True).to("cuda")
            return float(critic(**dict(enc)))

        @torch.no_grad()
        def duel(pa, pb, desc):                                    # card3: each side its own 3-frame card, run both orders
            s = 0.0
            for first, second, sgn in ((pa, pb, +1), (pb, pa, -1)):
                ims = [_I.open(x).convert("RGB") for x in list(first) + list(second)]
                s += sgn * _fwd(ims, PROMPT.format(d=desc))
            return s

    # ---- inference on the clip
    imgs = load_frames(args.video, args.maxf)                 # (S,H,W,3) uint8, used by seg + grounder + overlay
    S = imgs.shape[0]
    print(f"[demo] {S} frames, expr='{args.expr}'", flush=True)
    with torch.no_grad():
        masks_full, masks2d, scores, qemb, register = seg_forward_keepmasks(seg, imgs)
    n = qemb.shape[0]
    if n == 0:
        print("[demo] segmenter found 0 instances"); return
    e_norm = F.normalize(torch.from_numpy(qemb).float().cuda(), dim=-1)
    reg_all = register.astype(np.float32)
    desc = args.expr

    cot = _gen_mf_cot(ag, proc, imgs[np.linspace(0, S - 1, min(S, MF_NF)).round().astype(int)], desc)
    wins, bxall = mf_tree_windows(np.arange(S), S, lambda ck: (
        _gen_mf_box_batch(ag, proc, [reg_all[w] for w in ck], [imgs[w] for w in ck], desc, cot_text=cot)
        if len(ck) > 1 else [_gen_mf_box(ag, proc, reg_all[ck[0]], imgs[ck[0]], desc, cot_text=cot)]))
    votes = np.zeros(n)
    for w in wins:
        for idx, cf in _agent_mframe(ag, proc, reg_all[w], imgs[w], desc, e_norm, cot_text=cot):
            if 0 <= idx < n:
                votes[idx] += cf
    sel = int(votes.argmax()) if votes.max() > 0 else -1
    mA = masks2d[sel][:S] if sel >= 0 else np.zeros((S, MH, MW), bool)                 # A: qemb-path masklet

    items = VBS.boxes_to_items(wins, bxall)
    _, _, nb, mBox = VBS.score_box_path(pm, imgs, items, np.zeros((S, MH, MW), bool),   # dummy gt: we only want the mask
                                        IN_W, IN_H, up=lambda m, hw: m, jf=lambda a, b: (0., 0.),
                                        tag="demo", return_mask=True)
    mB = np.zeros_like(mA) if mBox is None else (mBox.astype(bool) if mBox.shape[1:] == mA.shape[1:] else
        np.stack([cv2.resize(m.astype(np.uint8), (mA.shape[2], mA.shape[1]), interpolation=cv2.INTER_NEAREST).astype(bool) for m in mBox]))
    if len(mB) != len(mA):
        t = np.zeros_like(mA); t[:min(len(t), len(mB))] = mB[:min(len(t), len(mB))]; mB = t

    pick, final = "A", mA
    if critic is not None:
        ps = {}
        for nm, mm in (("A", mA), ("B", mB)):
            p = os.path.join(os.environ["OUT"], f"demo__{nm}.jpg")
            ar = mm.reshape(len(mm), -1).sum(1)
            fs = sorted(int(x) for x in np.argsort(-ar)[:3] if ar[int(x)] > 0) or [0]
            render(imgs, fs, mm[fs], p); ps[nm] = [p]
        pick = "A" if duel(ps["A"], ps["B"], desc) > 0 else "B"
        final = mA if pick == "A" else mB
    print(f"[demo] A(qemb) sel={sel}  B(box) n_box={nb}  critic pick={pick}", flush=True)

    # upsample the chosen masklet to original resolution and write the overlay
    os.makedirs(args.out, exist_ok=True)
    Ho, Wo = imgs.shape[1:3]
    up_final = np.stack([cv2.resize(m.astype(np.uint8), (Wo, Ho), interpolation=cv2.INTER_NEAREST).astype(bool) for m in final[:S]])
    name = os.path.splitext(os.path.basename(args.video))[0]
    overlay_mp4(imgs, up_final, os.path.join(args.out, f"{name}_grounded.mp4"))
    np.savez_compressed(os.path.join(args.out, f"{name}_masklet.npz"), mask=up_final)
    print(f"[demo] wrote {name}_grounded.mp4 + {name}_masklet.npz -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
