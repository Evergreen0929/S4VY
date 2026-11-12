#!/usr/bin/env python
"""Dump segmenter predictions as per-clip idmap (S,H,W)+scores on the SAME exported frames the baselines used,
so paper-aligned metrics (T-mIoU/T-SR/J&F via score_metrics.py) compare segmenter vs baselines apples-to-apples."""
import argparse, os, sys, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VGGT_OMEGA_ROOT", "/PATH/TO/codebase/vggt-omega")
import torch
from PIL import Image
from segomega.models import SegOmega
from segomega.engine import setup_trainable
from segomega.engine.decode import edge_snap, masks_to_idmap, masks_to_idmap_owned, model_input_hw
from segomega.engine.fast_inference import GraphRunner, optimize_for_inference, run as forward

CKPT = os.environ.get("S4VY_CKPT", "/PATH/TO/checkpoints/S4VY/s4vy/segomega_final.pt")
OUT_NAME = os.environ.get("DUMP_NAME", "segmenter")   # <name>.npz per clip


def build(fast="full"):
    m = SegOmega(ckpt_path=os.environ.get("VGGT_OMEGA_CKPT"), num_queries=400, num_dec_layers=9, mask_dim=256,
                 dec_dim=256, supervise_depth=False, supervise_camera=False, use_teacher=False,
                 grad_checkpointing=True, load_pretrained=True, mask_feat_ratio=4)
    m = setup_trainable(m, mode="lora", lora_rank=256, lora_alpha=256.0)
    sd = torch.load(CKPT, map_location="cpu"); m.load_state_dict(sd.get("model", sd), strict=False)
    m = m.cuda().eval()
    if fast == "off":
        return m
    optimize_for_inference(m, exact=(fast == "exact"))
    return GraphRunner(m, capacity=2)   # benchmark clips repeat a few shapes


@torch.no_grad()
def run(model, cdir, decoder="owned", long_side="native"):
    pngs = sorted(glob.glob(os.path.join(cdir, "frames", "*.png")))
    origs = np.stack([np.asarray(Image.open(p).convert("RGB"), np.uint8) for p in pngs])
    S, H, W = origs.shape[:3]
    # The backbone needs its input capped and floored to a multiple of 16; feeding the exported frames
    # at whatever size they happen to be is what this used to do.
    Hr, Wr = model_input_hw(H, W, long_side)
    x = np.stack([np.asarray(Image.fromarray(f).resize((Wr, Hr), Image.BILINEAR), np.float32) / 255 for f in origs])
    imgs = torch.from_numpy(x).permute(0, 3, 1, 2)
    x = imgs.cuda().unsqueeze(0)
    preds = model(x) if isinstance(model, GraphRunner) else forward(model, x)
    if decoder == "legacy":                                   # area-ordered paint, then snap to image edges
        idmap, sc = masks_to_idmap(preds["objectness"], preds["mask_logits"], (H, W), score_thr=0.3)
        idmap = np.stack([edge_snap(idmap[f], origs[f]) for f in range(S)])
    else:
        idmap, sc = masks_to_idmap_owned(preds["objectness"], preds["mask_logits"], (H, W))
    np.savez_compressed(os.path.join(cdir, f"{OUT_NAME}.npz"), idmap=idmap, scores=sc)
    return S, H, W, int((sc > 0).sum())


def long_side_arg(value):
    return value if value == "native" else int(value)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/PATH/TO/data/baseline_eval")
    ap.add_argument("--benches", default="iggt_tracking,davis,ytvos,vipseg")
    ap.add_argument("--input-long-side", type=long_side_arg, default="native",
                    help='model input long side, up to 1024, or "native" (default) for the frames\' own size; '
                         '512 is what this script used before')
    ap.add_argument("--decoder", default="owned", choices=["owned", "legacy"],
                    help="owned: per-pixel score ownership (default); legacy: area-ordered paint + edge_snap")
    ap.add_argument("--fast", default="full", choices=["off", "exact", "full"],
                    help="full (default): merged LoRA, BF16 weights, torch.compile, CUDA graphs; "
                         "exact: bitwise-identical speedups only; off: the plain eager forward")
    args = ap.parse_args()
    model = build(args.fast)
    for b in args.benches.split(","):
        cds = sorted(glob.glob(os.path.join(args.root, b, "clip_*")))
        _mc = int(os.environ.get("MAX_CLIPS", "0"))
        if _mc > 0: cds = cds[:_mc]   # fast subset for quick probes
        for i, cd in enumerate(cds):
            try:
                S, H, W, n = run(model, cd, args.decoder, args.input_long_side)
                if i % 25 == 0: print(f"[{OUT_NAME} {b}] {i}/{len(cds)} S={S} {H}x{W} #inst={n}", flush=True)
            except Exception as e:
                print(f"[{OUT_NAME} {b}] FAIL {cd}: {e}", flush=True)
        print(f"[{OUT_NAME} {b}] DONE {len(cds)} clips", flush=True)
    print("[segmenter] ALL DONE")
