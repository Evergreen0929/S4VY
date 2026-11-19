#!/usr/bin/env python
"""Dump segmenter predictions as per-clip idmap (S,H,W)+scores on the SAME exported frames the baselines used,
so paper-aligned metrics (T-mIoU/T-SR/J&F via score_metrics.py) compare segmenter vs baselines apples-to-apples.

Two independent switches, both ON by default and both NOT part of the paper:
  --speedup on|off          faster inference (segomega/engine/fast_inference.py); LoRA is never merged
  --mask-id-enhance on|off  owned id-map decoding (masks_to_idmap_owned) instead of the paper's decoder
With both off (and the default --input-long-side native) this reproduces the paper's run exactly.
"""
import argparse, os, sys, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VGGT_OMEGA_ROOT", "/PATH/TO/codebase/vggt-omega")
os.environ.setdefault("SEG_NO_FRAMEPOS", "1")           # weights were trained with decoder frame_pos dropped
import torch
from PIL import Image
from segomega.models import SegOmega
from segomega.engine import setup_trainable
from segomega.engine.decode import decode_idmap, model_input_hw
from segomega.engine.fast_inference import GraphRunner, enable_speedup, run as forward
from segomega.engine.inference_memory import configure_inference_memory

CKPT = os.environ.get("S4VY_CKPT", "/PATH/TO/checkpoints/S4VY/s4vy/segomega_final.pt")
OUT_NAME = os.environ.get("DUMP_NAME", "segmenter")   # <name>.npz per clip


def build(speedup=True, low_memory=False, offload_backbone=False):
    m = SegOmega(ckpt_path=os.environ.get("VGGT_OMEGA_CKPT"), num_queries=400, num_dec_layers=9, mask_dim=256,
                 dec_dim=256, supervise_depth=False, supervise_camera=False, use_teacher=False,
                 grad_checkpointing=True, load_pretrained=True, mask_feat_ratio=4)
    m = setup_trainable(m, mode="lora", lora_rank=256, lora_alpha=256.0)
    sd = torch.load(CKPT, map_location="cpu"); m.load_state_dict(sd.get("model", sd), strict=False)
    m = m.cuda().eval()
    if low_memory or offload_backbone:
        configure_inference_memory(m, offload_backbone=offload_backbone)
    if not speedup:
        return m
    enable_speedup(m)
    return GraphRunner(m, capacity=2)   # benchmark clips repeat a few shapes


def load_clip(cdir, long_side="native"):
    """-> model input (1,S,3,Hr,Wr) in [0,1] on the GPU, and the exported frames' (H, W)."""
    pngs = sorted(glob.glob(os.path.join(cdir, "frames", "*.png")))
    origs = np.stack([np.asarray(Image.open(p).convert("RGB"), np.uint8) for p in pngs])
    S, H, W = origs.shape[:3]
    # "native" keeps the exported frames as they are (what the paper fed the model); an integer long side
    # resizes them, capped and floored to a multiple of 16.
    Hr, Wr = model_input_hw(H, W, long_side)
    if (Hr, Wr) != (H, W):
        origs = np.stack([np.asarray(Image.fromarray(f).resize((Wr, Hr), Image.BILINEAR)) for f in origs])
    x = torch.from_numpy(origs.astype(np.float32) / 255).permute(0, 3, 1, 2)
    return x.cuda().unsqueeze(0), (H, W)


@torch.no_grad()
def predict(model, x):
    return model(x) if isinstance(model, GraphRunner) else forward(model, x)


@torch.no_grad()
def run(model, cdir, mask_id_enhance=True, long_side="native"):
    x, (H, W) = load_clip(cdir, long_side)
    idmap, sc = decode_idmap(predict(model, x), (H, W), mask_id_enhance)
    np.savez_compressed(os.path.join(cdir, f"{OUT_NAME}.npz"), idmap=idmap, scores=sc)
    return idmap.shape[0], H, W, int((sc > 0).sum())


def long_side_arg(value):
    return value if value == "native" else int(value)


def on_off(value):
    if value not in ("on", "off"):
        raise argparse.ArgumentTypeError('expected "on" or "off"')
    return value == "on"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/PATH/TO/data/baseline_eval")
    ap.add_argument("--benches", default="iggt_tracking,davis,ytvos,vipseg")
    ap.add_argument("--speedup", type=on_off, default=True, metavar="{on,off}",
                    help="on (default): exact mask downsample, BF16 weight cache, torch.compile, CUDA graphs; "
                         "LoRA is not merged. off: the paper's plain eager forward. Not part of the paper.")
    ap.add_argument("--mask-id-enhance", type=on_off, default=True, metavar="{on,off}",
                    help="on (default): owned decoding, each pixel goes to the most confident query. "
                         "off: the paper's decoder (area-ordered paint). Not part of the paper.")
    ap.add_argument("--input-long-side", type=long_side_arg, default="native",
                    help='"native" (default): the frames as exported, as in the paper; or a long side up to 1024')
    ap.add_argument("--low-memory", action="store_true", help="discard unused inference auxiliaries and pyramid features")
    ap.add_argument("--offload-backbone", action="store_true", help="also store cached features in CPU RAM; disables CUDA graph capture")
    args = ap.parse_args()
    model = build(args.speedup, args.low_memory, args.offload_backbone)
    print(f"[{OUT_NAME}] speedup={'on' if args.speedup else 'off'} "
          f"mask_id_enhance={'on' if args.mask_id_enhance else 'off'} input={args.input_long_side}", flush=True)
    for b in args.benches.split(","):
        cds = sorted(glob.glob(os.path.join(args.root, b, "clip_*")))
        _mc = int(os.environ.get("MAX_CLIPS", "0"))
        if _mc > 0: cds = cds[:_mc]   # fast subset for quick probes
        for i, cd in enumerate(cds):
            try:
                S, H, W, n = run(model, cd, args.mask_id_enhance, args.input_long_side)
                if i % 25 == 0: print(f"[{OUT_NAME} {b}] {i}/{len(cds)} S={S} {H}x{W} #inst={n}", flush=True)
            except Exception as e:
                print(f"[{OUT_NAME} {b}] FAIL {cd}: {e}", flush=True)
        print(f"[{OUT_NAME} {b}] DONE {len(cds)} clips", flush=True)
    print("[segmenter] ALL DONE")
