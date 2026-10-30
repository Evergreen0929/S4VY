#!/usr/bin/env python
"""Run a video through S4VY and write per-frame instance masks + an RGB/mask overlay mp4.

Weights are pulled from Hugging Face (jdzhang0929/S4VY). Our checkpoints already bundle the
frozen VGGT-Omega backbone + LoRA + decoder, so nothing else needs downloading.

Usage:
  python demo/forward_video.py --video clip.mp4 --out out/clip [--model s4vy|joint] [--nframes 80]

Outputs (under --out):
  <name>_overlay.mp4      RGB + guided-filter mask overlay
  masks/frame_XXXX.png    per-frame instance id map (uint16; 0 = background)

Requires VGGT_OMEGA_ROOT to point at the VGGT-Omega code repo (see segomega/paths.py).
"""
import os, sys, argparse
os.environ.setdefault("SEG_NO_FRAMEPOS", "1")           # weights were trained with decoder frame_pos dropped
import numpy as np, cv2, torch, torch.nn.functional as F
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
from segomega.models import SegOmega
from segomega.engine import setup_trainable
from segomega.engine.decode import edge_snap, masks_to_idmap, model_input_hw

WEIGHTS_REPO = "jdzhang0929/S4VY"
WEIGHTS = {"s4vy": "segmenter/s4vy.pt", "joint": "segmenter/s4vy_joint.pt"}

_rng = np.random.RandomState(0)
PAL = ((_rng.rand(1024, 3) * 0.7 + 0.3) * 255).astype(np.uint8); PAL[0] = 0   # instance colour palette


def refine_and_overlay(idm, rgb, alpha=0.5):
    """Edge-snap each instance, then composite it -> (hard idmap, rgb+mask overlay u8)."""
    hard = edge_snap(idm, rgb)
    over = rgb.astype(np.float32) / 255.0
    for i in [int(x) for x in np.unique(hard) if x > 0]:
        a = (hard == i).astype(np.float32)[..., None] * alpha
        over = over * (1 - a) + (PAL[i % 1024].astype(np.float32) / 255.0) * a
    return hard, (over * 255).astype(np.uint8)


def load_frames(path, n, long_side=512):
    """subsample n frames from a video; return (model-res tensor S,3,Hr,Wr float[0,1], original-res RGB S,Ho,Wo,3)."""
    cap = cv2.VideoCapture(path)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    want = set(int(i) for i in np.unique(np.linspace(0, max(tot - 1, 0), min(n, tot or n)).round().astype(int)))
    frames = {}; i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            frames[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        i += 1
        if len(frames) == len(want):
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {path}")
    origs = np.stack([frames[k] for k in sorted(frames)])                       # (S,Ho,Wo,3)
    h0, w0 = origs.shape[1:3]
    Hr, Wr = model_input_hw(h0, w0, long_side)
    imgs = np.stack([cv2.resize(f, (Wr, Hr), interpolation=cv2.INTER_AREA) for f in origs]).astype(np.float32) / 255.
    return torch.from_numpy(imgs).permute(0, 3, 1, 2), origs


def build_model(which):
    w = hf_hub_download(WEIGHTS_REPO, WEIGHTS[which])          # bundles backbone + LoRA + decoder
    # s4vy_joint was trained jointly with the prompt branch, so its checkpoint carries a prompt
    # encoder; build it so every tensor in the checkpoint has a home.
    m = SegOmega(ckpt_path=None, num_queries=400, num_dec_layers=9, mask_dim=256, dec_dim=256,
                 supervise_depth=False, supervise_camera=False, use_teacher=False,
                 grad_checkpointing=True, load_pretrained=False, mask_feat_ratio=4,
                 enable_prompt=(which == "joint"), prompt_arch="v1lp",
                 prompt_init_gain_content=25.0, prompt_init_gain_pos=17.0)
    m = setup_trainable(m, mode="lora", lora_rank=256, lora_alpha=256.0)   # inject LoRA to match the ckpt structure
    sd = torch.load(w, map_location="cpu")
    missing, unexpected = m.load_state_dict(sd.get("model", sd), strict=False)
    print(f"[demo] loaded {WEIGHTS[which]}  (missing={len(missing)} unexpected={len(unexpected)})", flush=True)
    return m.cuda().eval()


@torch.no_grad()
def forward_masks(model, imgs, out_hw):
    """imgs (S,3,H,W) -> per-frame instance id map (S,Ho,Wo), background = 0."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        s = model(imgs.cuda().unsqueeze(0))
    return masks_to_idmap(s["objectness"], s["mask_logits"], out_hw, score_thr=0.3)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="input mp4")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--model", default="s4vy", choices=["s4vy", "joint"])
    ap.add_argument("--nframes", type=int, default=int(os.environ.get("NFRAMES", "80")))
    ap.add_argument("--long_side", type=int, default=512)
    args = ap.parse_args()

    imgs, origs = load_frames(args.video, args.nframes, args.long_side)
    S, Ho, Wo = origs.shape[:3]
    name = os.path.splitext(os.path.basename(args.video))[0]
    print(f"[demo] {name}: {S} frames, orig={Wo}x{Ho}", flush=True)

    model = build_model(args.model)
    idmap = forward_masks(model, imgs, (Ho, Wo))                                # instance masks at original res

    mdir = os.path.join(args.out, "masks"); os.makedirs(mdir, exist_ok=True)
    cap = cv2.VideoCapture(args.video); nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    ov_fps = max(1.0, S / ((nfr / fps) if fps and nfr else (S / 12.0)))         # match input duration
    vw = cv2.VideoWriter(os.path.join(args.out, name + "_overlay.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), ov_fps, (Wo, Ho))
    for f in range(S):
        hard, over = refine_and_overlay(idmap[f], origs[f])                     # edge-snap + overlay
        cv2.imwrite(os.path.join(mdir, f"frame_{f:04d}.png"), hard.astype(np.uint16))
        vw.write(cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"[demo] wrote {name}_overlay.mp4 ({S}f {Wo}x{Ho}, {ov_fps:.1f}fps) + masks/ ({S} png)", flush=True)


if __name__ == "__main__":
    main()
