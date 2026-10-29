"""Training loop for s4vy: DEFAULT SegOmega Option-B head, FULL fine-tune, geometry
heads/teacher dropped, mask supervised at FULL res (SetCriterion). Separate engine so earlier hot
paths are untouched. Reuses dist/save/eval/optimizer helpers."""
from __future__ import annotations

import os
import time

import torch
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from ..data import InsSceneDataset, collate_fn
from ..models.instseg_model import SegOmega
from ..losses.criterion import SetCriterion
from .optimizer import build_optimizer, WarmupCosine, setup_trainable
from .distributed import is_dist, setup_dist, move_to, save_ckpt

try:                                  # optional: the benchmark evaluation harness (not released)
    from .eval import run_full_eval as _  # noqa: F401
    _HAS_EVAL = True
except ImportError:
    _HAS_EVAL = False


def _prompt_schedule(jp):
    """Return a validated [(start_step, prompt_every), ...] schedule.

    ``prompt_schedule`` is the canonical form.  The legacy
    ``prompt_every[_late]`` fields remain supported so older configs keep the
    exact same behavior.
    """
    if not jp:
        return [(0, 2)]
    raw = jp.get("prompt_schedule")
    if raw:
        schedule = [(int(x["start_step"]), int(x["prompt_every"])) for x in raw]
    else:
        early = int(jp.get("prompt_every", 2))
        late = int(jp.get("prompt_every_late", early))
        switch = int(jp.get("prompt_every_switch_step", 10**9))
        schedule = [(0, early)]
        if switch < 10**9 and late != early:
            schedule.append((switch, late))
    schedule = sorted(schedule)
    if not schedule or schedule[0][0] != 0:
        raise ValueError("joint_prompt.prompt_schedule must start at step 0")
    if len({s for s, _ in schedule}) != len(schedule):
        raise ValueError("joint_prompt.prompt_schedule has duplicate start_step values")
    if any(e < 2 for _, e in schedule):
        raise ValueError("joint_prompt.prompt_every must be >=2 (one prompt step per N steps)")
    return schedule


def _prompt_every_at(step, schedule):
    every = schedule[0][1]
    for start, candidate in schedule:
        if step < start:
            break
        every = candidate
    return every


class WeightedDistributedSampler(Sampler):
    """Per-item weighted sampling (proportional to subset weight), DDP-sharded. ALL items
    participate (nonzero weight) -- supports fractional weights (e.g. savx 0.36) that integer replication can't.
    Only used when cfg.data.sampling_weights is set; teacher keeps the plain DistributedSampler unchanged."""
    def __init__(self, weights, num_samples, num_replicas, rank, seed=0, weight_fn=None):
        self.weights = weights.double(); self.total = int(num_samples)
        self.nr = max(1, num_replicas); self.rank = rank; self.seed = seed; self.epoch = 0
        self.weight_fn = weight_fn        # s4vy dynamic mix: (global step at epoch start) -> per-item weights
    def set_epoch(self, e): self.epoch = e
    def __iter__(self):
        if self.weight_fn is not None:
            self.weights = self.weight_fn(self.epoch).double()   # epoch is keyed by global STEP (see train loop)
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        idx = torch.multinomial(self.weights, self.total, replacement=True, generator=g)
        return iter(idx[self.rank::self.nr].tolist())          # each rank takes a disjoint shard of the same draw
    def __len__(self):
        return self.total // self.nr


# online photometric aug applied IN THE LOOP (after the EMA teacher forward) so the TEACHER sees
# CLEAN frames (reliable pseudo) while the STUDENT trains on augmented frames. Matches the dataset's aug
# (ColorJitter(0.5,0.5,0.5,0.1) + RandomGrayscale(0.2)), per-frame. Only used when ema_pseudo is on
# (with dataset augment=false); teacher keeps augmentation in the dataset -> unchanged.
from torchvision import transforms as _TV
_ONLINE_AUG = _TV.Compose([_TV.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                           _TV.RandomGrayscale(p=0.2)])
def _online_photo_aug(imgs):   # (S,3,H,W) in [0,1] -> per-frame photometric aug
    return torch.stack([_ONLINE_AUG(imgs[i]) for i in range(imgs.shape[0])], 0)


def train_segmenter(cfg: dict):
    local_rank, rank, world = setup_dist()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    # GATED (default None = OFF, existing runs unaffected): fix global RNG so model init (LoRA + heads + the 400
    # static query embeddings) is reproducible across runs. Data order was already deterministic (sampler seed=0);
    # only model init was unseeded -> the run-to-run early-mAP variance. Set cfg.train.seed for a clean A/B.
    _seed = cfg["train"].get("seed")
    if _seed is not None:
        import numpy as _np, random as _random
        torch.manual_seed(int(_seed)); torch.cuda.manual_seed_all(int(_seed))
        _np.random.seed(int(_seed)); _random.seed(int(_seed))
        if is_main:
            print(f"[run] global manual_seed={_seed} (model init reproducible)", flush=True)

    d = cfg["data"]
    _DS = InsSceneDataset
    _extra = {}
    if d.get("video_subsets"):                                   # Seg4D-Anything: add 4D/video subsets
        from ..data.seg4d_video import Seg4dDataset
        _DS = Seg4dDataset
        _extra = dict(video_subsets=d["video_subsets"], sparse_window_mult=d.get("sparse_window_mult", 8),
                      video_weights=d.get("video_weights"), video_split=d.get("video_split", "all"),
                      pseudo_prompt_root=d.get("pseudo_prompt_root"),          # pure-pseudo subsets (dl3dv/matterport) -> v3 offline pseudo
                      pseudo_prompt_subsets=tuple(d.get("pseudo_prompt_subsets", ())),
                      gt_pseudo_root=d.get("gt_pseudo_root"),                   # non-exh subsets WITH real GT -> real GT covers pseudo, pseudo fills
                      gt_pseudo_subsets=tuple(d.get("gt_pseudo_subsets", ())))
    ds = _DS(root=d["root"], subsets=d.get("subsets", ["infinigen", "scannetpp_v2"]),
                         num_frames=d.get("num_frames"), resolution=d.get("resolution", 512),
                         frame_sampling=d.get("frame_sampling", "random"),
                         split=d.get("split", "all"), test_frac=d.get("test_frac", 0.0),
                         min_frames=d.get("min_frames", 2), max_frames=d.get("max_frames", 12),
                         scannet_root=d.get("scannet_root"), augment=d.get("augment", True),  # train-only photometric aug
                         subset_sample_weights=d.get("subset_sample_weights"), **_extra)
    if is_main and d.get("video_subsets"):
        print(f"[run] train dataset: {len(ds)} items (video_split={d.get('video_split','all')})", flush=True)
    sw = d.get("sampling_weights")                              # fractional per-subset weights
    if sw:
        subs_of = lambda e: (e[0][2:] if isinstance(e[0], str) and e[0].startswith("v:") else e[0])
        item_sub = [subs_of(e) for e in ds.items]
        item_w = torch.tensor([float(sw.get(s, 1.0)) for s in item_sub], dtype=torch.double)
        # s4vy DYNAMIC MIX (gated; absent -> behavior unchanged): exhaustive:non-exhaustive MASS ratio r(t)
        # linear from r0 (at mix_start) to r1 (at mix_end). Non-exh items keep their static config weights;
        # exhaustive items get w_e(t) = r(t) * mass_non / n_exh so the mass ratio is exactly r(t).
        dm = d.get("dynamic_mix"); weight_fn = None
        if dm and dm.get("enable", True):
            _exh = set(dm["exhaustive_subsets"])
            _is_exh = torch.tensor([s in _exh for s in item_sub])
            _mass_non = float(item_w[~_is_exh].sum()); _n_exh = int(_is_exh.sum())
            _r0, _r1 = float(dm["r0"]), float(dm["r1"])
            _t0 = int(dm.get("mix_start", 0)); _t1 = int(dm.get("mix_end") or cfg["train"]["total_steps"])
            def weight_fn(gstep, _base=item_w, _m=_is_exh):
                f = min(max((gstep - _t0) / max(_t1 - _t0, 1), 0.0), 1.0)
                r = _r0 + (_r1 - _r0) * f
                w = _base.clone(); w[_m] = r * _mass_non / max(_n_exh, 1)
                return w
            if is_main:
                print(f"[run] DYNAMIC MIX exh:non-exh {_r0}:1 -> {_r1}:1 over steps [{_t0},{_t1}] "
                      f"(n_exh={_n_exh}, mass_non={_mass_non:.0f})", flush=True)
        if is_dist():
            sampler = WeightedDistributedSampler(item_w, num_samples=len(ds), num_replicas=world, rank=rank,
                                                 seed=0, weight_fn=weight_fn)
        else:
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(item_w, num_samples=len(ds), replacement=True)
        if is_main:
            print(f"[run] weighted sampler over {len(ds)} items (sampling_weights={sw})", flush=True)
    else:
        sampler = DistributedSampler(ds) if is_dist() else None   # teacher path: unchanged
    _nw = cfg["train"].get("num_workers", 8)
    _dl_kw = dict(batch_size=cfg["train"]["batch_size"], shuffle=(sampler is None), sampler=sampler,
                  num_workers=_nw, collate_fn=collate_fn, drop_last=True, pin_memory=True)
    if _nw > 0:   # pure-IO knobs (no training-semantics change): hide disk-decode latency for video-heavy mixes.
        _dl_kw["persistent_workers"] = cfg["train"].get("persistent_workers", False)   # default False -> teacher unchanged
        _dl_kw["prefetch_factor"] = cfg["train"].get("prefetch_factor", 2)             # default 2 -> teacher unchanged
    loader = DataLoader(ds, **_dl_kw)

    # ---- (gated; absent -> everything below unchanged) -------------------------------
    # Second loader for the point/box PROMPT task (1:1 with exhaustive via step parity). Own subset pool
    # (all seg subsets; dl3dv/matterport = offline pseudo, down-weighted) + own static weights.
    jp = cfg["train"].get("joint_prompt")
    p_loader = p_sampler = None
    if jp and jp.get("enable", True):
        from ..data.seg4d_video import Seg4dDataset as _S4D
        jd = jp["data"]
        p_ds = _S4D(root=jd.get("root", d["root"]), subsets=jd.get("subsets", []),
                    num_frames=jd.get("num_frames"), resolution=jd.get("resolution", d.get("resolution", 512)),
                    frame_sampling=jd.get("frame_sampling", "random"), split="all", test_frac=0.0,
                    min_frames=jd.get("min_frames", 8), max_frames=jd.get("max_frames", 16),
                    augment=jd.get("augment", True), video_subsets=jd.get("video_subsets", []),
                    sparse_window_mult=jd.get("sparse_window_mult", 8), video_split=jd.get("video_split", "train"),
                    pseudo_prompt_root=jd.get("pseudo_prompt_root"),
                    pseudo_prompt_subsets=tuple(jd.get("pseudo_prompt_subsets", ())))
        psw = jd.get("sampling_weights", {})
        _subs_of = lambda e: (e[0][2:] if isinstance(e[0], str) and e[0].startswith("v:") else e[0])
        p_w = torch.tensor([float(psw.get(_subs_of(e), 1.0)) for e in p_ds.items], dtype=torch.double)
        if is_dist():
            p_sampler = WeightedDistributedSampler(p_w, num_samples=len(p_ds), num_replicas=world, rank=rank, seed=1)
        else:
            from torch.utils.data import WeightedRandomSampler
            p_sampler = WeightedRandomSampler(p_w, num_samples=len(p_ds), replacement=True)
        _pkw = dict(_dl_kw); _pkw["sampler"] = p_sampler; _pkw["shuffle"] = False
        p_loader = DataLoader(p_ds, **_pkw)
        if is_main:
            _sched = _prompt_schedule(jp)
            _sched_s = ", ".join(f"step>={s}: {e-1}:1 exhaustive:prompt" for s, e in _sched)
            print(f"[run] joint prompt on: {len(p_ds)} prompt items, schedule=[{_sched_s}], "
                  f"noise_aug={jp.get('prompt', {}).get('noise_aug', 0.05)}, "
                  f"pseudo={jd.get('pseudo_prompt_subsets')}@w={[psw.get(s) for s in jd.get('pseudo_prompt_subsets', [])]}", flush=True)

    eval_loaders = []
    if is_main and cfg.get("eval", {}).get("enabled", False):
        ev = cfg["eval"]; nf = ev.get("num_frames", 8); broot = ev.get("benchmark_root")
        iggt_modes = ev.get("iggt_modes", ["tracking", "geometry"])   # sets [] to disable (Seg4D uses video benchmarks)
        if broot and iggt_modes:
            from ..data.iggt_benchmark import IGGTBenchmarkDataset
            for mode in iggt_modes:
                eds = IGGTBenchmarkDataset(root=broot, mode=mode, resolution=d.get("resolution", 512), num_frames=nf)
                eval_loaders.append((mode, DataLoader(eds, batch_size=1, shuffle=False,
                                     num_workers=ev.get("num_workers", 4), collate_fn=collate_fn), False))  # IGGT -> combined eval/ section
                print(f"[eval] IGGT_Benchmark {mode}: {len(eds)} scenes", flush=True)
        # per-benchmark video-seg eval, EACH its own wandb section eval/<bench>/* (no mixing)
        for bench in ev.get("video_benchmarks", []) or []:
            try:
                from ..data.seg4d_video import Seg4dDataset
                veds = Seg4dDataset(root=d["root"], subsets=[], video_subsets=[bench],
                                    num_frames=nf, frame_sampling="consecutive", resolution=d.get("resolution", 512),
                                    video_split=ev.get("video_split", "all"))   # eval on held-out val (vipseg/cityscapes/kitti)
                eval_loaders.append((bench, DataLoader(veds, batch_size=1, shuffle=False,
                                     num_workers=ev.get("num_workers", 4), collate_fn=collate_fn), True))  # video -> own eval_<bench>/ section
                print(f"[eval] video {bench} (split={ev.get('video_split','all')}): {len(veds)} clips", flush=True)
            except Exception as e:
                print(f"[eval] skip {bench}: {e}", flush=True)   # e.g. not yet extracted

    m = cfg["model"]
    model = SegOmega(
        ckpt_path=m.get("ckpt_path"), num_queries=m.get("num_queries", 400),
        num_dec_layers=m.get("num_dec_layers", 9), mask_dim=m.get("mask_dim", 256),
        dec_dim=m.get("dec_dim", 256), supervise_depth=False, supervise_camera=False,
        use_teacher=False, grad_checkpointing=m.get("grad_checkpointing", True),
        freeze_backbone=False, load_pretrained=m.get("load_pretrained", True),
        mask_feat_ratio=m.get("mask_feat_ratio", 4),   # H/4 (segmenter baseline); configs may pin 2 for H/2
        anyup_enable=m.get("anyup_enable", False),     # AnyUp replaces head's upsampling resize layers
        anyup_ckpt=m.get("anyup_ckpt"), anyup_qchunk=m.get("anyup_qchunk", 8192),
        hr_upsampler=m.get("hr_upsampler"), uplift_cfg=m.get("uplift_cfg"),   # UPLiFT end-branch
        uplift_iters=m.get("uplift_iters", 2), uplift_bf16=m.get("uplift_bf16", True),
        uplift_pyr=m.get("uplift_pyr", False),                               # UPLiFT in head pyramid
        uplift_guidance_ds=m.get("uplift_guidance_ds", 1),                   # uplift_compile=m.get("uplift_compile", False),
        enable_prompt=bool(jp),                                              # prompt branch on
        prompt_arch=m.get("prompt_arch", "v1lp"),
        prompt_init_gain_content=m.get("prompt_init_gain_content", 25.0),
        prompt_init_gain_pos=m.get("prompt_init_gain_pos", 17.0),
    ).to(device)
    ft_mode = m.get("finetune_mode", "full")
    model = setup_trainable(model, mode=ft_mode,
                            lora_rank=m.get("lora_rank", 32), lora_alpha=m.get("lora_alpha", 32.0))
    model = model.to(device)  # LoRA adapters are injected on CPU after the first .to(device); re-move them

    if is_dist():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          find_unused_parameters=True)
    core = model.module if is_dist() else model
    if is_main:
        n_train = sum(p.numel() for p in core.parameters() if p.requires_grad)
        print(f"[run] trainable params: {n_train/1e6:.1f}M (finetune_mode={ft_mode}, seg-only)", flush=True)

    crit = SetCriterion(w_mask=cfg["loss"].get("w_mask", 5.0), w_dice=cfg["loss"].get("w_dice", 5.0),
                         w_obj=cfg["loss"].get("w_obj", 2.0), eos_coef=cfg["loss"].get("eos_coef", 0.1),
                         num_points=cfg["loss"].get("num_points", 12544),
                         ignore_bg_subsets=cfg["loss"].get("ignore_bg_subsets"),
                         pseudo_weight=cfg["loss"].get("pseudo_weight", 0.5),
                         pseudo_no_exh_flip=cfg["train"].get("pseudo_no_exh_flip", False)).to(device)

    # (opt-in via cfg.train.ema_pseudo): online EMA teacher for soft pseudo-labels on
    # non-exhaustive batches. teacher has no such flag -> ema stays None -> loop below is identical to teacher.
    ema = None
    if cfg["train"].get("ema_pseudo"):
        from .ema_pseudo import EMATeacher, generate_pseudo
        _ema_init_ckpt = cfg["train"].get("ema_init_ckpt")
        _ema_lazy = _ema_init_ckpt is None                              # no warm-start -> DELAYED self-teacher (snapshot at update_start)
        _tfp = cfg["train"].get("teacher_framepos", _ema_init_ckpt is not None)  # teacher frame_pos ON only if warm-started from a frame_pos-trained ckpt
        ema = EMATeacher(core, momentum=cfg["train"].get("ema_momentum", 0.999),
                         init_ckpt=_ema_init_ckpt, teacher_framepos=_tfp)
        _pseudo_thr = cfg["train"].get("pseudo_obj_thr", 0.5)
        # GATED (default OFF -> constant _pseudo_thr): linear objectness-threshold schedule. Keep pseudo VERY clean early
        # (hi, e.g. 0.95) so only the teacher's most-confident masks supervise, then relax to lo (0.5) as the student
        # matures. thr(step)= hi for step<=decay_start, lo for step>=decay_end, linear between.
        _thr_sched = bool(cfg["train"].get("pseudo_thr_schedule", False))
        _thr_hi = float(cfg["train"].get("pseudo_thr_hi", _pseudo_thr))
        _thr_lo = float(cfg["train"].get("pseudo_thr_lo", _pseudo_thr))
        _thr_ds = int(cfg["train"].get("pseudo_thr_decay_start", 0))
        _thr_de = int(cfg["train"].get("pseudo_thr_decay_end", 1))
        def _thr_at(_st):
            if not _thr_sched:
                return _pseudo_thr
            if _st <= _thr_ds:
                return _thr_hi
            if _st >= _thr_de:
                return _thr_lo
            return _thr_hi + (_thr_lo - _thr_hi) * (_st - _thr_ds) / max(1, _thr_de - _thr_ds)
        _pseudo_maxfrac = cfg["train"].get("pseudo_max_area_frac", 0.5)
        # THREE-BAND pseudo (gated, default off): obj>obj_thr POSITIVE, ignore_lo<obj<=obj_thr IGNORE (not supervised,
        # not ∅), obj<=ignore_lo suppressed by ∅. Fixes the ∅-on under-seg (∅ pushed teacher-missed objects to bg).
        _three_band = bool(cfg["train"].get("pseudo_three_band", False))
        _ignore_lo = float(cfg["train"].get("pseudo_ignore_lo", 0.3))
        _ema_update_start = cfg["train"].get("ema_update_start", 0)      # step teacher starts EMA-updating (and, if lazy, is snapshot from student)
        # step pseudo supervision begins. LAZY self-teacher: default = update_start (teacher not competent until then).
        # WARM-START teacher (warm-start / frozen-teacher variants): default = 0 (teacher good from step 0). BUG FIX: previously defaulted to
        # _ema_update_start for ALL, so the frozen-teacher variant (update_start=50001 for permanent freeze) got pseudo_start=50001 = pseudo
        # NEVER applied -> lost the pseudo path + its exhaustive-flip -> AP collapsed (mIoU intact). See the frozen-teacher variant @20000 crash.
        _pseudo_start = cfg["train"].get("pseudo_start", _ema_update_start if _ema_lazy else 0)
        _ema_snapped = not _ema_lazy                                    # lazy mode: snapshot teacher from student once at _ema_update_start
        _ema_persist = cfg["train"].get("ema_persist", False)          # save/restore teacher EMA across chain restarts (else teacher re-inits from teacher each link -> per-link sawtooth). Diagnostic: isolates reset-discontinuity vs genuine drift.
        if is_main:
            print(f"[run] EMA-teacher pseudo ENABLED (m={cfg['train'].get('ema_momentum',0.999)}, obj_thr={_pseudo_thr}, "
                  f"lazy_self_teacher={_ema_lazy}, pseudo_start={_pseudo_start}, update_start={_ema_update_start}, teacher_framepos={_tfp}, ema_persist={_ema_persist}, "
                  f"thr_sched={_thr_sched}({_thr_hi}->{_thr_lo} @{_thr_ds}-{_thr_de}), pseudo_no_exh_flip={cfg['train'].get('pseudo_no_exh_flip', False)})", flush=True)

    total_steps = cfg["train"]["total_steps"]
    optimizer = build_optimizer(core, lr_backbone=cfg["train"].get("lr_backbone", 1e-6),
                                lr_head=cfg["train"]["lr_head"], weight_decay=cfg["train"].get("weight_decay", 0.05),
                                lr_uplift=cfg["train"].get("lr_uplift"))   # separate upsampler group
    _uplift_gi = next((i for i, g in enumerate(optimizer.param_groups) if g.get("name") == "uplift"), None)
    scheduler = WarmupCosine(optimizer, total_steps, warmup_frac=cfg["train"].get("warmup_frac", 0.05),
                             uplift_gi=_uplift_gi,                          # upsampler LR anneals to 0 on its own fast cosine
                             uplift_anneal_step=cfg["train"].get("uplift_anneal_step", 0))
    grad_clip = cfg["train"].get("grad_clip", 1.0)
    ckpt_dir = cfg["train"].get("ckpt_dir", "./checkpoints")
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)

    wb = None
    if is_main and cfg.get("wandb", {}).get("enabled", False):
        try:
            import wandb as wb
            wb.init(project=cfg["wandb"].get("project", "segvggtomega"), entity=cfg["wandb"].get("entity") or None,
                    name=cfg["wandb"].get("run_name") or os.path.basename(ckpt_dir),
                    id=cfg["wandb"].get("run_id") or None, resume="allow", config=cfg)
            print(f"[wandb] init run={wb.run.name}", flush=True)
        except Exception as e:
            print(f"[wandb] disabled ({e})", flush=True); wb = None

    start_step = 0
    resumed = False
    if cfg["train"].get("resume", "auto"):
        import glob as _glob
        cks = sorted(_glob.glob(os.path.join(ckpt_dir, "segomega_*.pt")), key=os.path.getmtime)
        if cks:
            ck = torch.load(cks[-1], map_location="cpu")
            core.load_state_dict(ck["model"], strict=False)
            if "optimizer" in ck:
                optimizer.load_state_dict(ck["optimizer"])
            start_step = ck.get("step", 0); scheduler.t = start_step
            resumed = True
            # PERSIST teacher EMA across chain restarts (else EMATeacher just re-inited from teacher -> reset each link).
            if ema is not None and cfg["train"].get("ema_persist", False) and "ema_state" in ck:
                for n in ema.ema:
                    if n in ck["ema_state"]:
                        ema.ema[n].copy_(ck["ema_state"][n].to(ema.ema[n].device, ema.ema[n].dtype))
                if is_main:
                    print(f"[ema] RESTORED teacher EMA from ckpt (persist) -> teacher continuous across links", flush=True)
            if is_main:
                print(f"[resume] from {cks[-1]} at step {start_step}", flush=True)
    # warm-start (step 0) from a prior run's weights (e.g. segmenter final) when no resume ckpt exists.
    # strict=False so a freshly-added head (AnyUp) keeps its pretrained init; everything else inherits segmenter.
    init_from = cfg["train"].get("init_from")
    if not resumed and init_from and os.path.exists(init_from):
        ck = torch.load(init_from, map_location="cpu")
        sd = ck.get("model", ck)
        miss, unexp = core.load_state_dict(sd, strict=False)
        if is_main:
            print(f"[init_from] warm-started from {init_from} (missing={len(miss)} unexpected={len(unexp)})", flush=True)

    # prompt: loss fns + fixed held-out prompt-eval clips (same prompt-eval protocol)
    p_iter = None; jp_eval_clips = None
    if p_loader is not None:
        from ..data.prompt_sampling import sample_prompts as _sample_prompts
        from .train_prompt import _prompt_total_loss as _jp_total_loss, _grab_eval_clips as _jp_grab, _prompt_eval as _jp_eval
        _jpp = jp.get("prompt", {}); _jpl = jp.get("loss", {})
        if hasattr(p_sampler, "set_epoch"): p_sampler.set_epoch(start_step)
        if is_main:
            try:
                jp_eval_clips = _jp_grab(cfg)
                print(f"[run] prompt-eval on {len(jp_eval_clips)} held-out clips", flush=True)
            except Exception as e:
                print(f"[run] prompt-eval clips skipped ({e})", flush=True)

    model.train()
    step = start_step; t0 = time.time()
    if sampler is not None and getattr(sampler, "weight_fn", None) is not None:
        sampler.set_epoch(step)          # dynamic mix: first draw uses the RESUME step's ratio, not step 0's
    data_iter = iter(loader)
    _pe_schedule = _prompt_schedule(jp)
    while step < total_steps:
        _pe_cur = _prompt_every_at(step, _pe_schedule)
        _is_p = p_loader is not None and (step % _pe_cur == _pe_cur - 1)   # 1 PROMPT step per _pe_cur steps (DDP branch-synced: step is deterministic across ranks)
        if _is_p:
            try:
                batch = next(p_iter) if p_iter is not None else next(iter(()))
            except StopIteration:
                if hasattr(p_sampler, "set_epoch"): p_sampler.set_epoch(step)
                p_iter = iter(p_loader); batch = next(p_iter)
        else:
            try:
                batch = next(data_iter)
            except StopIteration:
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(step)
                data_iter = iter(loader); batch = next(data_iter)
        batch = move_to(batch, device)

        # DELAYED self-teacher: once step reaches update_start, snapshot the (now-competent) student as the teacher
        # start (fires on fresh run AND on resume-past-update_start; robust since ema was built from the cold student).
        if ema is not None and _ema_lazy and not _ema_snapped and step >= _ema_update_start:
            ema.reset_from(core); _ema_snapped = True
        # for non-exhaustive (partial) batches, EMA teacher fills soft pseudo instances BEFORE the student
        if ema is not None and step >= _pseudo_start and batch.get("exhaustive") and not bool(batch["exhaustive"][0]):
            t_out = ema.forward_pseudo(core, batch["images"])
            pm, pc, pig = generate_pseudo(t_out, batch["inst_ids"], batch["num_instances"][0], obj_thr=_thr_at(step),
                                          max_area_frac=_pseudo_maxfrac, three_band=_three_band, ignore_lo=_ignore_lo)
            del t_out                                          # NO empty_cache() here: it cudaFrees the whole allocator
            if pm is not None:                                 # pool every non-exh step -> next student fwd/bwd must re-
                batch["pseudo_masks"] = [pm]; batch["pseudo_conf"] = [pc]   # cudaMalloc GBs (this was the ~4.5x slowdown)
            if pig is not None:                                # three-band IGNORE region (0.3-0.7 teacher masks)
                batch["pseudo_ignore"] = [pig]
        if ema is not None:                                    # student sees AUGMENTED frames (teacher saw clean above)
            batch["images"] = _online_photo_aug(batch["images"])

        if _is_p:                                          # point/box prompts from GT/pseudo masklets
            prompts, targets = _sample_prompts(batch["inst_ids"], batch["num_instances"],
                                               max_objs=_jpp.get("max_objs", 4), p_box=_jpp.get("p_box", 0.4),
                                               p_neg=_jpp.get("p_neg", 0.3),
                                               max_prompt_frames=_jpp.get("max_prompt_frames"),
                                               noise_aug=_jpp.get("noise_aug", 0.05))
            _zero = all(len(p) == 0 for p in prompts)
            if _zero:                                      # no valid object this rank -> dummy prompt, zero loss (DDP-safe)
                S, H, W = batch["inst_ids"].shape[1:]
                prompts = [[{"points": [(0, 0.5, 0.5, 1)], "boxes": []}] for _ in range(batch["inst_ids"].shape[0])]
                targets = [batch["inst_ids"].new_zeros(1, S, H, W, dtype=torch.float32) for _ in prompts]
            preds = model(batch["images"], prompts=prompts)
            losses = _jp_total_loss(preds, targets, _jpl.get("num_points", 12544),
                                    _jpl.get("w_mask", 5.0), _jpl.get("w_dice", 5.0), aux_w=_jpl.get("aux_w", 0.4))
            losses = {f"p_{k}" if not k.startswith("p_") else k: v for k, v in losses.items()}
            loss = losses["p_loss_total"] * (0.0 if _zero else 1.0)
            # s4vy_joint ∅-DENSITY FIX (gated by joint_prompt.exh_aux_weight, default 0 = behavior):
            # the prompt forward already produced the FULL static-400 outputs (unidirectional attn keeps them
            # bit-identical to a promptless pass). On EXHAUSTIVE samples (every pixel labeled) also apply the
            # regular Hungarian+∅ criterion to them, so idle queries get ∅ pressure on prompt steps too
            # (∅ coverage 50% -> ~77% of steps). Non-exhaustive samples skip (∅-tax on unlabeled objects).
            _eaw = float(jp.get("exh_aux_weight", 0.0))
            if _eaw > 0.0 and batch.get("exhaustive") and bool(batch["exhaustive"][0]) and not _zero:
                _el, _ = crit(preds, batch)
                losses["loss_exhfull"] = _el["loss_total"]   # name avoids the "aux" log filter
                loss = loss + _eaw * _el["loss_total"]
        else:
            _swc = (cfg.get("stream_window") or {}).get("choices")
            if _swc:                                        # per-step random backbone window (0=full), cleared after fwd so eval stays full
                _w = int(_swc[int(torch.randint(len(_swc), (1,)).item())])
                if _w > 0:
                    os.environ["VGGT_OMEGA_STREAM_WINDOW"] = str(_w); os.environ["VGGT_OMEGA_STREAM_NOCACHE"] = "1"
                else:
                    os.environ.pop("VGGT_OMEGA_STREAM_WINDOW", None)
            preds = model(batch["images"])
            os.environ.pop("VGGT_OMEGA_STREAM_WINDOW", None); os.environ.pop("VGGT_OMEGA_STREAM_NOCACHE", None)
            losses, _ = crit(preds, batch)
            loss = losses["loss_total"]
        # BEFORE the teacher intervenes (step < pseudo_start), the pseudo-destined non-exhaustive data has NO
        # teacher pseudo AND no ∅ signal (exhaustive=False) -> training on it teaches over-emission (AP collapse).
        # Zero its loss so the pre-teacher warmup is effectively pure-exhaustive (DDP-safe: fwd/bwd still run,
        # gradient is zero). At pseudo_start the teacher is snapshot and non-exh joins WITH pseudo+∅.
        if ema is not None and step < _pseudo_start and batch.get("exhaustive") and not bool(batch["exhaustive"][0]):
            loss = loss * 0.0
        # per-subset supervision weight (e.g. down-weight noisier ScanNet200). batch_size=1/GPU -> one subset.
        _sw = cfg["data"].get("subset_loss_weights", {})
        _w = _sw.get(batch["subset"][0], 1.0) if batch.get("subset") else 1.0
        if _w != 1.0:
            loss = loss * _w
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad], grad_clip)
        optimizer.step()
        if ema is not None and step >= _ema_update_start:       # teacher FROZEN at warm-start (teacher) for the first
            ema.update(core)                                    # _ema_update_start steps -> student catches up on clean
        lr_scale = scheduler.step(); step += 1                  # pseudo before EMA lets the teacher co-evolve (avoids collapse)

        # with step parity, a fixed log_every parity would only ever sample ONE task's losses ->
        # keep the latest loss dict of EACH branch and log the merged view.
        _cur = {k: v.item() for k, v in losses.items()
                if (k.startswith("loss_") or k.startswith("p_loss_")) and "aux" not in k}
        if _is_p:
            _last_p = _cur
        else:
            _last_e = _cur
        if is_main and step % cfg["train"].get("log_every", 20) == 0:
            ips = step * cfg["train"]["batch_size"] * world / (time.time() - t0)
            ldict = {}
            try: ldict.update(_last_e)
            except NameError: pass
            try: ldict.update(_last_p)
            except NameError: pass
            print(f"[step {step}/{total_steps}] " + " ".join(f"{k}={v:.3f}" for k, v in ldict.items()) +
                  f" lr_scale={lr_scale:.3f} ips={ips:.1f}", flush=True)
            if wb is not None:
                wb.log({**{f"train/{k}": v for k, v in ldict.items()}, "train/lr_scale": lr_scale, "train/ips": ips}, step=step)

        # Periodic benchmark validation. The evaluation harness that produces the paper's numbers
        # is not part of this release, so the hook is a soft import and training simply skips it.
        if is_main and eval_loaders and step % cfg["eval"].get("eval_every", 2000) == 0 and _HAS_EVAL:
            from .eval import run_full_eval
            from ..data.seg4d_video import VIDEO_SUBSETS
            te = time.time(); all_metrics = {}
            for name, ld, is_video in eval_loaders:
                viz = os.path.join(ckpt_dir, "eval_viz", f"step{step}", name)
                ex = VIDEO_SUBSETS[name][2] if (is_video and name in VIDEO_SUBSETS) else True  # partial video benchmarks -> oracle-ignore unmatched preds (fair AP); IGGT 3D GT is dense
                mm, imgs = run_full_eval(core, ld, device, out_dir=viz, exhaustive=ex,
                                         max_scenes=cfg["eval"].get("max_scenes", 30), n_viz=cfg["eval"].get("n_viz", 4))
                all_metrics.update({f"{name}/{k}": v for k, v in mm.items()})
                if wb is not None:
                    # IGGT (geometry/tracking) -> original combined 'eval/' section; video benchmarks -> own 'eval_<name>/' section
                    pfx, ppfx = (f"eval_{name}/", f"eval_{name}/") if is_video else (f"eval/{name}/", "eval/")
                    wb.log({f"{pfx}{k}": v for k, v in mm.items()}, step=step)
                    if imgs:
                        wb.log({f"{ppfx}panels": [wb.Image(p) for p in imgs]}, step=step)
            print(f"[eval step {step}] " + " ".join(f"{k}={v:.3f}" for k, v in all_metrics.items()) + f" ({time.time()-te:.0f}s)", flush=True)
            # extra: ScanNet200-val prediction visualizations (NO metrics), logged to wandb
            n_sviz = cfg["eval"].get("scannet_val_viz", 0)
            sroot = cfg["data"].get("scannet_root")
            if wb is not None and n_sviz and sroot and _HAS_EVAL:
                try:
                    from .eval import scannet_val_panels
                    sviz_dir = os.path.join(ckpt_dir, "eval_viz", f"step{step}", "scannet_val")
                    sp = scannet_val_panels(core, sroot, device, n_scenes=n_sviz,
                                            resolution=cfg["data"].get("resolution", 512), out_dir=sviz_dir)
                    if sp:
                        wb.log({"eval/scannet200_val_viz": [wb.Image(p) for p in sp]}, step=step)
                        print(f"[eval step {step}] logged {len(sp)} scannet200-val viz", flush=True)
                except Exception as e:
                    print(f"[scannet_val_viz] skipped: {type(e).__name__}: {e}", flush=True)
            model.train()

        # prompt-task eval (point/box mIoU per bench, the prompt-eval protocol) on the same cadence
        if is_main and jp_eval_clips and step % cfg["eval"].get("eval_every", 2000) == 0:
            try:
                pel = _jp_eval(core, jp_eval_clips, device, wb, step)
                print(f"[eval step {step}] PROMPT " +
                      " ".join(f"{k.replace('eval/','')}={v:.3f}" for k, v in pel.items()), flush=True)
            except Exception as e:
                print(f"[prompt eval] skip ({type(e).__name__}: {e})", flush=True)
            model.train()

        _es = ema.ema if (ema is not None and _ema_persist) else None   # persist teacher EMA -> continuous across restarts
        if is_main and step % cfg["train"].get("save_every", 500) == 0:
            save_ckpt(core, optimizer, step, ckpt_dir, os.path.join(ckpt_dir, f"segomega_{step}.pt"), keep_last=2, ema_state=_es)

    if is_main:
        save_ckpt(core, optimizer, step, ckpt_dir, os.path.join(ckpt_dir, "segomega_final.pt"), keep_last=2,
                   ema_state=(ema.ema if (ema is not None and _ema_persist) else None))
        if step >= total_steps:
            open(os.path.join(ckpt_dir, "DONE"), "w").close()
    if is_dist():
        torch.distributed.destroy_process_group()
