#!/usr/bin/env python
"""Dump segmenter predictions as per-clip idmap (S,H,W)+scores on the SAME exported frames the baselines used,
so paper-aligned metrics (T-mIoU/T-SR/J&F via score_metrics.py) compare segmenter vs baselines apples-to-apples."""
import argparse, os, sys, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VGGT_OMEGA_ROOT", "/PATH/TO/codebase/vggt-omega")
import torch, torch.nn.functional as F
from PIL import Image
from segomega.models import SegOmega
from segomega.engine import setup_trainable
from segomega.engine.decode import decode_instances

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
    imgs = torch.from_numpy(np.stack([np.asarray(Image.open(p).convert("RGB"), np.float32) / 255 for p in pngs])).permute(0, 3, 1, 2)
    S, _, H, W = imgs.shape
    with torch.autocast("cuda", dtype=torch.bfloat16):
        preds = model(imgs.cuda().unsqueeze(0))
    scores, masks = decode_instances(preds["objectness"], preds["mask_logits"], score_thr=0.3, mask_thr=0.5)[0]
    if masks.numel() == 0:
        idmap = np.zeros((S, H, W), np.int32); sc = np.zeros((0,), np.float32)
    else:
        up = F.interpolate(masks.float().reshape(-1, 1, *masks.shape[-2:]), size=(H, W), mode="nearest").reshape(masks.shape[0], S, H, W).bool().cpu().numpy()
        sc = scores.float().cpu().numpy()
        order = np.argsort([m.sum() for m in up])                # small first -> big overwrites
        idmap = np.zeros((S, H, W), np.int32)
        for newid, qi in enumerate(order, start=1):
            idmap[up[qi]] = newid
        sc = sc[order]
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
