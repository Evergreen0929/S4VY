#!/usr/bin/env python
"""Check final-logit and instance-ID parity of memory modes on a short clip.

Use a released S4VY (prompt-free) checkpoint and a directory of source RGB PNGs.
All modes use the same sampled frames and eager numerical operations.
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SEG_NO_FRAMEPOS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

from segomega.engine import setup_trainable
from segomega.engine.decode import masks_to_idmap_owned, model_input_hw
from segomega.engine.fast_inference import run
from segomega.engine.inference_memory import configure_inference_memory
from segomega.models import SegOmega


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--nframes", type=int, default=10)
    parser.add_argument("--long-side", type=int, default=1024)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.frames.glob("*.png"))
    if not files or args.nframes < 1:
        parser.error("Provide source PNGs and a positive frame count")
    indices = np.unique(np.linspace(0, len(files) - 1, min(args.nframes, len(files))).astype(int))
    rgb = [cv2.cvtColor(cv2.imread(str(files[i])), cv2.COLOR_BGR2RGB) for i in indices]
    height, width = rgb[0].shape[:2]
    hr, wr = model_input_hw(height, width, args.long_side)
    resized = np.stack([cv2.resize(frame, (wr, hr), interpolation=cv2.INTER_AREA) for frame in rgb])
    images = torch.from_numpy(resized).permute(0, 3, 1, 2).cuda().float().div_(255).unsqueeze(0)
    model = SegOmega(ckpt_path=None, num_queries=400, num_dec_layers=9, mask_dim=256, dec_dim=256,
                     supervise_depth=False, supervise_camera=False, use_teacher=False,
                     grad_checkpointing=True, load_pretrained=False, mask_feat_ratio=4)
    model = setup_trainable(model, mode="lora", lora_rank=256, lora_alpha=256.)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model", state), strict=True)
    del state
    model = model.cuda().eval().requires_grad_(False)
    results = []
    reference = None
    for mode in ["default", "low_memory", "offload_backbone"]:
        if mode != "default":
            configure_inference_memory(model, offload_backbone=mode == "offload_backbone")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        pred = run(model, images)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated()
        final = {key: pred[key].cpu() for key in ["objectness", "mask_logits", "mask_embed_final"]}
        ids, _ = masks_to_idmap_owned(pred["objectness"], pred["mask_logits"], (height, width))
        if reference is None:
            reference, reference_ids = final, ids
        parity = {key: torch.equal(value, reference[key]) for key, value in final.items()}
        parity["instance_ids"] = bool(np.array_equal(ids, reference_ids))
        row = dict(mode=mode, seconds=elapsed, peak_allocated_bytes=peak, bitwise_equal=parity,
                   auxiliary_layers=len(pred["aux_outputs"]))
        results.append(row)
        print(json.dumps(row), flush=True)
        if not all(parity.values()):
            raise AssertionError(f"Prediction parity failed for {mode}")
        del pred, final, ids
    result = dict(status="pass", frames=[files[i].name for i in indices], model_hw=[hr, wr],
                  torch_version=torch.__version__, device=torch.cuda.get_device_name(),
                  timing_note="First mode includes cold initialization; this is a parity/memory check, not a speed comparison.",
                  modes=results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
