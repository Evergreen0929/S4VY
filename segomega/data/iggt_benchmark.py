"""IGGT_Benchmark eval dataset (the official IGGT test set), for IGGT-aligned evaluation.

Two modes (different sub-benchmarks, used as two eval passes):
  mode="geometry"  ->  processed_scannet_eval/<scene>/{images, gt_depth, gt_cam}
                       yields depth + camera pose (-> depth AbsRel/tau, pose ATE/RPE/AUC).
                       gt_cam pose is camera->world; we invert to world->cam (OpenCV) to match
                       VGGT-Omega's encoding_to_camera convention.
  mode="tracking"  ->  "3D Tracking Benchmark"/{scannet,scannetpp}/<scene>/images/{NNNNN.jpg, NNNNN_label.npy}
                       _label.npy holds 3D-CONSISTENT instance/track ids (same id across frames),
                       yields inst_ids (-> matched mIoU/mAcc == IGGT 2D mIoU/mAcc, temporally consistent).

Outputs the same sample dict as InsSceneDataset so collate_fn consumes it unchanged.
Missing modalities are simply omitted (consumers guard per-key).
"""
from __future__ import annotations

import os, sys
_VG = os.environ.get("VGGT_OMEGA_ROOT", "/PATH/TO/codebase/vggt-omega")
if _VG not in sys.path: sys.path.insert(0, _VG)

import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class IGGTBenchmarkDataset(Dataset):
    def __init__(self, root: str, mode: str = "tracking", resolution: int = 512,
                 num_frames: int = 10, max_instances: int = 100, min_instance_pixels: int = 50, aspect_resize: bool = True):
        super().__init__()
        self.root = root
        self.mode = mode
        self.res = resolution
        self.aspect_resize = aspect_resize
        self._cur_thtw = (resolution, resolution)
        self.num_frames = num_frames
        self.max_instances = max_instances
        self.min_instance_pixels = min_instance_pixels
        self.scenes = []  # (dataset_tag, scene_dir)
        if mode == "geometry":
            base = os.path.join(root, "processed_scannet_eval")
            for d in sorted(glob.glob(os.path.join(base, "scene*"))):
                if os.path.isdir(os.path.join(d, "images")):
                    self.scenes.append(("scannet", d))
        elif mode == "tracking":
            for tag in ("scannet", "scannetpp"):
                base = os.path.join(root, "3D Tracking Benchmark", tag)
                for d in sorted(glob.glob(os.path.join(base, "*"))):
                    if os.path.isdir(os.path.join(d, "images")):
                        self.scenes.append((tag, d))
        else:
            raise ValueError(mode)
        if not self.scenes:
            raise RuntimeError(f"no IGGT_Benchmark {mode} scenes under {root}")

    def __len__(self):
        return len(self.scenes)

    # ---- resize helpers (square; bilinear RGB, nearest for depth/labels) ----
    def _balanced_hw(self, h, w):
        from vggt_omega.utils.load_fn import _balanced_target_shape
        return _balanced_target_shape(h / max(w, 1), self.res, 16)

    def _rgb(self, path):
        im = Image.open(path).convert("RGB")
        if self.aspect_resize:
            W, H = im.size; th, tw = self._balanced_hw(H, W); self._cur_thtw = (th, tw)
        else:
            th = tw = self.res
        im = im.resize((tw, th), Image.BILINEAR)
        return torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)

    def _map(self, arr):
        th, tw = self._cur_thtw if self.aspect_resize else (self.res, self.res)
        im = Image.fromarray(arr).resize((tw, th), Image.NEAREST)
        return np.asarray(im)

    def _finalize_inst(self, insts):
        inst = np.stack(insts, 0).astype(np.int64)             # (S,H,W)
        uniq = inst[inst > 0]
        uniq = np.unique(uniq)
        keep = [u for u in uniq if (inst == u).sum() >= self.min_instance_pixels][: self.max_instances]
        out = np.zeros_like(inst)
        for i, u in enumerate(keep):
            out[inst == u] = i + 1
        return torch.from_numpy(out), len(keep)

    def __getitem__(self, i):
        tag, d = self.scenes[i]
        if self.mode == "tracking":
            return self._load_tracking(tag, d)
        return self._load_geometry(tag, d)

    def _pick(self, ids):
        ids = sorted(ids)
        if len(ids) <= self.num_frames:
            return ids
        sel = np.linspace(0, len(ids) - 1, self.num_frames).round().astype(int)
        return [ids[k] for k in sel]

    def _load_tracking(self, tag, scene_dir):
        idir = os.path.join(scene_dir, "images")
        labs = sorted(glob.glob(os.path.join(idir, "*_label.npy")))
        stems = [os.path.basename(p)[:-len("_label.npy")] for p in labs]
        stems = self._pick(stems)
        rgbs, insts = [], []
        for st in stems:
            rgbs.append(self._rgb(os.path.join(idir, st + ".jpg")))
            insts.append(self._map(np.load(os.path.join(idir, st + "_label.npy")).astype(np.int32)))
        inst_t, k = self._finalize_inst(insts)
        return {"subset": f"track_{tag}", "images": torch.stack(rgbs, 0),
                "inst_ids": inst_t, "num_instances": k}

    def _load_geometry(self, tag, scene_dir):
        idir = os.path.join(scene_dir, "images")
        imgs = sorted(glob.glob(os.path.join(idir, "*.jpg")))
        stems = self._pick([os.path.basename(p)[:-4] for p in imgs])
        rgbs, depths, Ks, Ts = [], [], [], []
        for st in stems:
            rgbs.append(self._rgb(os.path.join(idir, st + ".jpg")))
            dp = os.path.join(scene_dir, "gt_depth", st + ".png")
            d = np.asarray(Image.open(dp)).astype(np.float32) / 1000.0
            depths.append(self._map(d))
            cam = np.load(os.path.join(scene_dir, "gt_cam", st + ".npz"))
            Ks.append(np.asarray(cam["intrinsics"], np.float32))
            c2w = np.asarray(cam["pose"], np.float32)            # camera->world
            Ts.append(np.linalg.inv(c2w).astype(np.float32))     # -> world->cam (OpenCV)
        depth = torch.from_numpy(np.stack(depths, 0)).float()
        valid = (depth > 0) & (depth < 1e6)
        depth = torch.where(valid, depth, torch.zeros_like(depth))
        return {"subset": f"geom_{tag}", "images": torch.stack(rgbs, 0),
                "depth": depth, "depth_valid": valid,
                "intrinsics": torch.from_numpy(np.stack(Ks, 0)).float(),
                "pose": torch.from_numpy(np.stack(Ts, 0)).float()}
