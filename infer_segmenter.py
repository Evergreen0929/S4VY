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
from segomega.engine.decode import edge_snap, masks_to_idmap, model_input_hw

CKPT = os.environ.get("S4VY_CKPT", "/PATH/TO/checkpoints/S4VY/s4vy/segomega_final.pt")
OUT_NAME = os.environ.get("DUMP_NAME", "segmenter")   # <name>.npz per clip


def build():
    m = SegOmega(ckpt_path=os.environ.get("VGGT_OMEGA_CKPT"), num_queries=400, num_dec_layers=9, mask_dim=256,
                 dec_dim=256, supervise_depth=False, supervise_camera=False, use_teacher=False,
                 grad_checkpointing=True, load_pretrained=True, mask_feat_ratio=4)
    m = setup_trainable(m, mode="lora", lora_rank=256, lora_alpha=256.0)
    sd = torch.load(CKPT, map_location="cpu"); m.load_state_dict(sd.get("model", sd), strict=False)
    return m.cuda().eval()


@torch.no_grad()
def run(model, cdir):
    pngs = sorted(glob.glob(os.path.join(cdir, "frames", "*.png")))
    origs = np.stack([np.asarray(Image.open(p).convert("RGB"), np.uint8) for p in pngs])
    S, H, W = origs.shape[:3]
    # The backbone needs its input capped and floored to a multiple of 16; feeding the exported frames
    # at whatever size they happen to be is what this used to do.
    Hr, Wr = model_input_hw(H, W)
    x = np.stack([np.asarray(Image.fromarray(f).resize((Wr, Hr), Image.BILINEAR), np.float32) / 255 for f in origs])
    imgs = torch.from_numpy(x).permute(0, 3, 1, 2)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        preds = model(imgs.cuda().unsqueeze(0))
    idmap, sc = masks_to_idmap(preds["objectness"], preds["mask_logits"], (H, W), score_thr=0.3)
    idmap = np.stack([edge_snap(idmap[f], origs[f]) for f in range(S)])   # snap to image edges
    np.savez_compressed(os.path.join(cdir, f"{OUT_NAME}.npz"), idmap=idmap, scores=sc)
    return S, H, W, len(sc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/PATH/TO/data/baseline_eval")
    ap.add_argument("--benches", default="iggt_tracking,davis,ytvos,vipseg"); args = ap.parse_args()
    model = build()
    for b in args.benches.split(","):
        cds = sorted(glob.glob(os.path.join(args.root, b, "clip_*")))
        _mc = int(os.environ.get("MAX_CLIPS", "0"))
        if _mc > 0: cds = cds[:_mc]   # fast subset for quick probes
        for i, cd in enumerate(cds):
            try:
                S, H, W, n = run(model, cd)
                if i % 25 == 0: print(f"[{OUT_NAME} {b}] {i}/{len(cds)} S={S} {H}x{W} #inst={n}", flush=True)
            except Exception as e:
                print(f"[{OUT_NAME} {b}] FAIL {cd}: {e}", flush=True)
        print(f"[{OUT_NAME} {b}] DONE {len(cds)} clips", flush=True)
    print("[segmenter] ALL DONE")
