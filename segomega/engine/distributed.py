"""Distributed setup and checkpoint I/O shared by the training entry points."""
from __future__ import annotations

import os

import torch


def is_dist():
    return int(os.environ.get("WORLD_SIZE", 1)) > 1


def setup_dist():
    if is_dist():
        torch.distributed.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 0, 1


def move_to(batch, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device, non_blocking=True)
    return batch


def save_ckpt(core, optimizer, step, ckpt_dir, path, keep_last=2, ema_state=None):
    """Save trainable state for resume. Excludes the frozen teacher (== pretrained, reloaded on resume),
    so checkpoints are ~half size. Prunes to the `keep_last` most recent `segomega_*.pt` to bound disk.
    ema_state (optional): the EMA-teacher's trainable-param dict, persisted so the teacher is CONTINUOUS
    across chain restarts (ema_persist=True) instead of re-initing from the teacher recipe on each link."""
    import glob
    sd = {k: v for k, v in core.state_dict().items() if not k.startswith("teacher.")}
    obj = {"model": sd, "optimizer": optimizer.state_dict(), "step": step}
    if ema_state is not None:
        obj["ema_state"] = {n: t.detach().cpu() for n, t in ema_state.items()}
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)  # atomic: a concurrent reader never sees a half-written ckpt
    cks = sorted(glob.glob(os.path.join(ckpt_dir, "segomega_*.pt")), key=os.path.getmtime)
    for old in cks[:-keep_last]:
        try:
            os.remove(old)
        except OSError:
            pass


