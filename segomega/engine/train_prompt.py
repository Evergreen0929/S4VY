"""promptable training: freeze the frozen base (backbone+LoRA+decoder+head), train ONLY the PromptEncoder.
Each step samples point/box prompts online from the batch's GT instance masklets, runs the frozen base +
prompt query (unidirectional self-attn), and supervises the prompt query's mask against the prompted
object's masklet (point-sampled BCE+dice, incl. deep supervision on decoder layers). All-mask-labeled data
at the frozen base sampling ratios. Separate prompt-eval viz dumped to its own synced dir (NOT mixed with eval)."""
from __future__ import annotations

import os, time, glob

import torch
from torch.utils.data import DataLoader, DistributedSampler

from ..data import InsSceneDataset, collate_fn
from ..models.instseg_model import SegOmega
from ..engine.optimizer import setup_trainable, WarmupCosine
from .distributed import is_dist, setup_dist, move_to
from ..engine.train_segmenter import WeightedDistributedSampler
from ..data.prompt_sampling import sample_prompts, prompt_mask_loss


def _build_loader(cfg, world, rank, is_main):
    d = cfg["data"]
    _DS, _extra = InsSceneDataset, {}
    if d.get("video_subsets"):
        from ..data.seg4d_video import Seg4dDataset
        _DS = Seg4dDataset
        _extra = dict(video_subsets=d["video_subsets"], sparse_window_mult=d.get("sparse_window_mult", 8),
                      video_weights=d.get("video_weights"), video_split=d.get("video_split", "all"),
                      pseudo_prompt_root=d.get("pseudo_prompt_root"),          # offline pseudo id-maps (dl3dv/matterport)
                      pseudo_prompt_subsets=d.get("pseudo_prompt_subsets", []))
    ds = _DS(root=d["root"], subsets=d.get("subsets", ["infinigen", "scannetpp_v2"]),
             num_frames=d.get("num_frames"), resolution=d.get("resolution", 512),
             frame_sampling=d.get("frame_sampling", "random"), split=d.get("split", "all"),
             test_frac=d.get("test_frac", 0.0), min_frames=d.get("min_frames", 2),
             max_frames=d.get("max_frames", 12), scannet_root=d.get("scannet_root"),
             augment=d.get("augment", True), subset_sample_weights=d.get("subset_sample_weights"),
             # CORRECTNESS: base backbone was trained on balanced (aspect-preserving) resize; Seg4dDataset forces
             # aspect_resize=True for static subsets, but plain InsSceneDataset (video_subsets=[]) defaults False
             # (square distort). Match the base. (This is a real bug, but NOT why v1 fails to converge.)
             aspect_resize=d.get("aspect_resize", True), **_extra)
    sw = d.get("sampling_weights")
    if sw:
        subs_of = lambda e: (e[0][2:] if isinstance(e[0], str) and e[0].startswith("v:") else e[0])
        item_w = torch.tensor([float(sw.get(subs_of(e), 1.0)) for e in ds.items], dtype=torch.double)
        sampler = (WeightedDistributedSampler(item_w, len(ds), world, rank, seed=0) if is_dist()
                   else __import__("torch").utils.data.WeightedRandomSampler(item_w, len(ds), replacement=True))
        if is_main:
            print(f"[prompt] weighted sampler over {len(ds)} items (weights={sw})", flush=True)
    else:
        sampler = DistributedSampler(ds) if is_dist() else None
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=(sampler is None), sampler=sampler,
                        num_workers=cfg["train"].get("num_workers", 8), collate_fn=collate_fn,
                        drop_last=True, pin_memory=True)
    return ds, loader, sampler


def _prompt_total_loss(preds, targets, num_points, w_mask, w_dice, aux_w=0.4):
    """final-layer loss + deep supervision on all decoder layers (prompt_aux_outputs)."""
    out = prompt_mask_loss(preds["prompt_mask_logits"], targets, num_points, w_mask, w_dice)
    total = out["loss_total"]
    aux = preds.get("prompt_aux_outputs") or []
    for a in aux:
        la = prompt_mask_loss(a["mask_logits"], targets, num_points, w_mask, w_dice)
        total = total + aux_w * la["loss_total"]
    out["loss_total"] = total
    return out



def _det_prompt(mask, W, H, kind):
    """deterministic eval prompt: point=centroid pixel / box=tight bbox, on the object's biggest frame."""
    import numpy as np
    areas = [int(mask[s].sum()) for s in range(mask.shape[0])]; f = int(np.argmax(areas))
    ys, xs = torch.where(mask[f])
    if kind == "point":
        cx, cy = float(xs.float().mean()), float(ys.float().mean())
        j = int(((xs - cx) ** 2 + (ys - cy) ** 2).argmin())
        return {"points": [(f, (int(xs[j]) + .5) / W, (int(ys[j]) + .5) / H, 1)], "boxes": []}
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return {"points": [], "boxes": [(f, x0 / W, y0 / H, x1 / W, y1 / H)]}


def _grab_eval_clips(cfg, n_per=3):
    """a few HELD-OUT val clips per bench (+ IGGT tracking) for in-training prompt-mIoU eval."""
    import numpy as np
    from collections import defaultdict
    d = cfg["data"]
    n_per = int(cfg["train"].get("eval_clips_per_bench", n_per))   # bigger per-bench eval sample
    from ..data.seg4d_video import Seg4dDataset
    vids = [s for s in d.get("video_subsets", []) if s in ("davis", "lvos", "vipseg", "cityscapes", "kitti")]
    clips = []
    if vids:                                              # skip video eval-set entirely when no video benches (iggt-only ablation)
        vds = Seg4dDataset(root=d["root"], subsets=[], video_subsets=vids, video_split="val", num_frames=None,
                           resolution=512, min_frames=8, max_frames=12, video_weights={s: 1 for s in vids})
        by = defaultdict(list)
        for j, e in enumerate(vds.items):
            s = e[0][2:] if isinstance(e[0], str) and e[0].startswith("v:") else e[0]; by[s].append(j)
        for s in vids:
            js = by.get(s, [])
            for t in np.linspace(0, len(js) - 1, min(n_per, len(js))).astype(int) if js else []:
                try: clips.append(vds[js[int(t)]])
                except Exception: pass
    try:
        from ..data.iggt_benchmark import IGGTBenchmarkDataset
        ig = IGGTBenchmarkDataset(root=os.environ.get("IGGT_BENCHMARK_ROOT", "/PATH/TO/data/IGGT_Benchmark"),
                                  mode="tracking", resolution=512, num_frames=8)
        iggt_n = cfg["train"].get("eval_iggt_clips", n_per)   # ablation: iggt-only -> use more clips for a stable curve
        for t in np.linspace(0, len(ig) - 1, min(iggt_n, len(ig))).astype(int):
            try: clips.append(ig[int(t)])
            except Exception: pass
    except Exception:
        pass
    return clips


@torch.no_grad()
def _prompt_eval(core, clips, device, wb, step):
    """per-bench prompt-mIoU (point/box). v2: pick the candidate with max predicted IoU (realistic inference)."""
    import numpy as np, torch.nn.functional as F
    from collections import defaultdict
    core.eval(); res = defaultdict(lambda: {"point": [], "box": []})
    for s in clips:
        imgs = s["images"].unsqueeze(0).to(device); inst = s["inst_ids"].to(device); K = int(s["num_instances"])
        if K == 0: continue
        sub = s.get("subset", "?"); sub = "iggt_track" if sub.startswith("track_") else sub
        H, W = imgs.shape[3], imgs.shape[4]
        objs = [k for k in range(K) if int((inst == k + 1).sum()) >= 64][:4]
        for kind in ("point", "box"):
            prompts = [[_det_prompt((inst == k + 1), W, H, kind) for k in objs]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pr = core(imgs, prompts=prompts)
            pm = F.interpolate(pr["prompt_mask_logits"][0].float(), size=(H, W), mode="bilinear", align_corners=False).sigmoid() > 0.5
            for pi, k in enumerate(objs):
                gt = (inst == k + 1)
                cm = pm[pi]
                inter = (cm & gt).sum().item(); uni = (cm | gt).sum().item()
                if uni > 0: res[sub][kind].append(inter / uni)
    core.train()
    log = {}; allp, allb = [], []
    for sub, dd in res.items():
        p, b = np.array(dd["point"]), np.array(dd["box"]); allp += list(p); allb += list(b)
        if len(p): log[f"eval/{sub}/point_miou"] = float(p.mean())
        if len(b): log[f"eval/{sub}/box_miou"] = float(b.mean())
    if allp: log["eval/ALL/point_miou"] = float(np.mean(allp))
    if allb: log["eval/ALL/box_miou"] = float(np.mean(allb))
    if wb is not None and log: wb.log(log, step=step)
    return log


def train(cfg):
    local_rank, rank, world = setup_dist()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    is_main = (rank == 0)
    torch.manual_seed(int(cfg["train"].get("seed", 0)) + rank)

    ds, loader, sampler = _build_loader(cfg, world, rank, is_main)

    m = cfg["model"]
    model = SegOmega(ckpt_path=m.get("ckpt_path"), num_queries=m.get("num_queries", 400),
                     num_dec_layers=m.get("num_dec_layers", 9), mask_dim=m.get("mask_dim", 256),
                     dec_dim=m.get("dec_dim", 256), supervise_depth=False, supervise_camera=False,
                     use_teacher=False, grad_checkpointing=m.get("grad_checkpointing", True),
                     freeze_backbone=False, load_pretrained=m.get("load_pretrained", True),
                     mask_feat_ratio=m.get("mask_feat_ratio", 4), enable_prompt=True,
                     prompt_arch=m.get("prompt_arch", "v1lp"),
                     prompt_init_gain_content=m.get("prompt_init_gain_content", 1.0),
                     prompt_init_gain_pos=m.get("prompt_init_gain_pos", 1.0)).to(device)
    model = setup_trainable(model, mode="lora", lora_rank=m.get("lora_rank", 256),
                            lora_alpha=m.get("lora_alpha", 256.0)).to(device)
    # load frozen base (the frozen base); prompt_encoder is new -> stays at init
    init_from = cfg["train"].get("init_from")
    if init_from and os.path.exists(init_from):
        sd = torch.load(init_from, map_location="cpu"); sd = sd.get("model", sd)
        miss, unexp = model.load_state_dict(sd, strict=False)
        if is_main:
            npe = sum(1 for k in miss if k.startswith("prompt_encoder."))
            print(f"[prompt] base<-{os.path.basename(init_from)} missing={len(miss)}(prompt_encoder new={npe}) unexpected={len(unexp)}", flush=True)
    # FREEZE everything except the prompt encoder
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith("prompt_encoder."))
    ntr = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad) / 1e6
    if is_main:
        print(f"[prompt] trainable = prompt_encoder only: {ntr:.2f}M", flush=True)

    if is_dist():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    core = model.module if is_dist() else model

    total_steps = cfg["train"]["total_steps"]
    opt = torch.optim.AdamW([p for p in core.parameters() if p.requires_grad],
                            lr=cfg["train"]["lr_head"], weight_decay=cfg["train"].get("weight_decay", 0.05), betas=(0.9, 0.95))
    sched = WarmupCosine(opt, total_steps, warmup_frac=cfg["train"].get("warmup_frac", 0.05))
    ckpt_dir = cfg["train"]["ckpt_dir"]; os.makedirs(ckpt_dir, exist_ok=True) if is_main else None
    npnts = cfg["loss"].get("num_points", 12544); wmk = cfg["loss"].get("w_mask", 5.0); wdc = cfg["loss"].get("w_dice", 5.0)
    smpl = cfg["train"].get("prompt", {})

    wb = None
    if is_main and cfg.get("wandb", {}).get("enabled", False):
        try:
            import wandb as wb
            wb.init(project=cfg["wandb"].get("project", "segvggtomega"), entity=cfg["wandb"].get("entity") or None,
                    name=cfg["wandb"].get("run_name") or os.path.basename(ckpt_dir),
                    id=cfg["wandb"].get("run_id") or None, resume="allow", config=cfg)
        except Exception as e:
            print(f"[wandb] off ({e})"); wb = None

    start_step = 0
    cks = sorted(glob.glob(os.path.join(ckpt_dir, "prompt_*.pt")), key=os.path.getmtime)
    if cfg["train"].get("resume", "auto") and cks:
        ck = torch.load(cks[-1], map_location="cpu")
        core.load_state_dict(ck["model"], strict=False)
        if "optimizer" in ck: opt.load_state_dict(ck["optimizer"])
        start_step = ck.get("step", 0); sched.t = start_step
        if is_main: print(f"[resume] {cks[-1]} @ step {start_step}", flush=True)

    # fixed held-out clips for the prompt-eval viz + in-training numeric eval (sampled once, deterministic)
    viz_clips = _grab_viz_clips(ds, cfg) if is_main else None
    eval_clips = _grab_eval_clips(cfg) if is_main else None
    if is_main and eval_clips:
        print(f"[run] in-training eval on {len(eval_clips)} held-out clips (per-bench prompt-mIoU -> wandb)", flush=True)

    model.train(); step = start_step; t0 = time.time(); it = iter(loader)
    while step < total_steps:
        try:
            batch = next(it)
        except StopIteration:
            if sampler is not None and hasattr(sampler, "set_epoch"): sampler.set_epoch(step)
            it = iter(loader); batch = next(it)
        batch = move_to(batch, device)
        prompts, targets = sample_prompts(batch["inst_ids"], batch["num_instances"],
                                          max_objs=smpl.get("max_objs", 4), p_box=smpl.get("p_box", 0.4),
                                          p_neg=smpl.get("p_neg", 0.3), max_prompt_frames=smpl.get("max_prompt_frames", 3),
                                          single_frame_box=smpl.get("single_frame_box", False),
                                          noise_aug=smpl.get("noise_aug", 0.0))
        if all(len(p) == 0 for p in prompts):        # no valid objects this step (all-bg clip) -> skip cleanly
            step += 1; sched.step(); continue
        preds = model(batch["images"], prompts=prompts)
        losses = _prompt_total_loss(preds, targets, npnts, wmk, wdc, aux_w=cfg["loss"].get("aux_w", 0.4))
        loss = losses["loss_total"]
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad], cfg["train"].get("grad_clip", 1.0))
        opt.step(); lr_scale = sched.step(); step += 1

        if is_main and step % cfg["train"].get("log_every", 20) == 0:
            ips = step * cfg["train"]["batch_size"] * world / (time.time() - t0)
            ld = {k: float(v) for k, v in losses.items() if k.startswith("loss_")}
            print(f"[step {step}/{total_steps}] " + " ".join(f"{k}={v:.3f}" for k, v in ld.items()) +
                  f" lr_scale={lr_scale:.3f} ips={ips:.1f}", flush=True)
            if wb is not None: wb.log({**ld, "lr_scale": lr_scale}, step=step)
        if is_main and step % cfg["train"].get("save_every", 500) == 0:
            torch.save({"model": core.state_dict(), "optimizer": opt.state_dict(), "step": step},
                       os.path.join(ckpt_dir, f"prompt_{step}.pt"))
            for old in sorted(glob.glob(os.path.join(ckpt_dir, "prompt_*.pt")), key=os.path.getmtime)[:-2]:
                os.remove(old)
        if is_main and viz_clips is not None and step % cfg["train"].get("viz_every", 500) == 0:
            try:
                _dump_prompt_viz(core, viz_clips, cfg["train"].get("viz_dir"), step, device)
            except Exception as e:
                print(f"[viz] skip ({e})", flush=True)
        if is_main and eval_clips and step % cfg["train"].get("eval_every", 2000) == 0:
            try:
                el = _prompt_eval(core, eval_clips, device, wb, step)
                print(f"[eval step {step}] " + " ".join(f"{k.replace('eval/','')}={v:.3f}" for k, v in el.items()), flush=True)
            except Exception as e:
                print(f"[eval] skip ({e})", flush=True)

    if is_main:
        torch.save({"model": core.state_dict(), "step": step}, os.path.join(ckpt_dir, f"prompt_{step}.pt"))
        open(os.path.join(ckpt_dir, "DONE"), "w").close()
        print("[prompt] DONE", flush=True)


# ------------------------------------------------------------------ viz (separate, synced) -------------
def _grab_viz_clips(ds, cfg, n=4):
    """deterministically pick n clips for the prompt-eval viz (held fixed across steps)."""
    import numpy as np
    idx = np.linspace(0, len(ds) - 1, n).astype(int).tolist()
    clips = []
    for i in idx:
        try:
            clips.append(ds[i])
        except Exception:
            continue
    return clips


@torch.no_grad()
def _dump_prompt_viz(core, clips, viz_dir, step, device):
    """For each fixed clip: drop ONE point prompt on a GT object, run prompt query, render prompt+mask
    overlays to viz_dir/step_{step}/. Writes an auto-refreshing index.html listing the latest step."""
    import numpy as np
    from PIL import Image
    if not viz_dir:
        return
    outd = os.path.join(viz_dir, f"step_{step}"); os.makedirs(outd, exist_ok=True)
    core.eval()
    for ci, s in enumerate(clips):
        imgs = s["images"].unsqueeze(0).to(device)                 # (1,S,3,H,W)
        inst = s["inst_ids"].unsqueeze(0).to(device); K = int(s["num_instances"])
        if K == 0:
            continue
        prompts, targets = sample_prompts(inst, [K], max_objs=1, p_box=0.0, p_neg=0.0, max_prompt_frames=1)
        if len(prompts[0]) == 0:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds = core(imgs, prompts=prompts)
        pm = preds["prompt_mask_logits"][0, 0].float().sigmoid()   # (S,h,w)
        S, H, W = imgs.shape[1], imgs.shape[3], imgs.shape[4]
        pm = torch.nn.functional.interpolate(pm.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)[:, 0]
        rgb = (imgs[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        pf, px, py, _ = prompts[0][0]["points"][0]
        for f in range(S):
            o = rgb[f].copy()
            mm = (pm[f] > 0.5).cpu().numpy()
            o[mm] = (0.45 * o[mm] + np.array([255, 40, 40]) * 0.55).astype(np.uint8)
            if f == pf:                                            # draw the prompt point
                cx, cy = int(px * W), int(py * H)
                o[max(0, cy - 4):cy + 4, max(0, cx - 4):cx + 4] = [0, 255, 0]
            Image.fromarray(o).resize((512, int(H * 512 / W))).save(os.path.join(outd, f"clip{ci}_{f:03d}.png"))
    core.train()
    # write/refresh an index listing all step_* dirs, newest first
    steps = sorted([d for d in os.listdir(viz_dir) if d.startswith("step_")],
                   key=lambda x: int(x.split("_")[1]), reverse=True)
    html = ["<!doctype html><meta charset=utf-8><title>prompt-eval</title>",
            "<meta http-equiv=refresh content=60><body style='background:#111;color:#ddd;font-family:sans-serif'>",
            "<h2>promptable — prompt-eval viz (green=prompt point, red=predicted masklet)</h2>"]
    for d in steps[:8]:
        pngs = sorted(glob.glob(os.path.join(viz_dir, d, "*.png")))
        html.append(f"<h3>{d}</h3>" + "".join(
            f"<img src='{d}/{os.path.basename(p)}' width=180 style='margin:2px'>" for p in pngs[:32]))
    open(os.path.join(viz_dir, "index.html"), "w").write("\n".join(html))
    print(f"[viz] dumped step_{step} -> {viz_dir}", flush=True)
