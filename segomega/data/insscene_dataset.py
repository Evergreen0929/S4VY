"""InsScene-15K dataset for class-agnostic multi-view instance segmentation.

Two usable subsets (verified by inspecting the downloaded data):

  processed_infinigen/scene_XXX/<hash>.zip      (read directly from the per-sub-scene zip)
      frames/Image/camera_0/Image_<f>_0_<take>_0.png       RGB
      frames/Depth/camera_0/Depth_<f>_0_<take>_0.npy       float32, invalid == 1e10
      frames/ObjectSegmentation/camera_0/...npy            int64 instance ids, 0 == background
      frames/camview/camera_0/...npz                       keys: K(3x3), T(4x4), HW(2,)
      -> instance ids are GLOBALLY 3D-CONSISTENT across the frames of a sub-scene (verified).

  processed_scannetpp_v2/<scene>/                (EXTRACTED tree; reassemble the split zip first --
      images/frame_XXXXXX.jpg                              see scripts/extract_split_zip.sh)
      depth/frame_XXXXXX.png                               uint16 depth (mm)
      refined_ins_ids/frame_XXXXXX.(png|npy)               refined instance ids (SAM2 + projected GT)
      scene_iphone_metadata.npz                            camera intrinsics / poses

A training sample = N frames drawn from ONE sub-scene/scene (so instance ids are comparable),
with everything resized to a square `resolution`.  Depth `1e10` (infinigen) and non-positive depths
are masked out.  Instance ids are remapped to a contiguous 1..K (0 = background) *within the sample*.
"""
from __future__ import annotations

import io
import os
import re
import zipfile
from glob import glob
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

INVALID_DEPTH = 1e6  # anything >= this (covers infinigen's 1e10 sentinel) is invalid


# ------------------------------------------------------------------------------------------------
# indexing
# ------------------------------------------------------------------------------------------------
class _InfinigenIndex:
    """Each entry is one sub-scene zip path."""

    def __init__(self, root: str):
        self.zips = sorted(glob(os.path.join(root, "processed_infinigen", "scene_*", "*.zip")))

    def __len__(self):
        return len(self.zips)


# ScanNet++ scenes reserved for evaluation — excluded from training.
SCANNETPP_EVAL_HOLDOUT = frozenset({
    "0a76e06478", "0a7cc12c0e", "0b031f3119", "0cf2e9402d", "e050c15a8d",
    "e898c76c1f", "e9e16b6043", "ebc200e928", "f6659a3107",
})


class _ScanNetppIndex:
    """Each entry is one extracted scene directory."""

    def __init__(self, root: str):
        base = os.path.join(root, "processed_scannetpp_v2")
        # the split-zip extracts to a NESTED dir (processed_scannetpp_v2/processed_scannetpp_v2/<scene>/),
        # so accept a scene dir (one holding images/) at depth 1 OR 2.
        cand = glob(os.path.join(base, "*")) + glob(os.path.join(base, "*", "*"))
        all_scenes = sorted(
            d for d in cand
            if os.path.isdir(d) and os.path.isdir(os.path.join(d, "images"))
        )
        self.scenes = [d for d in all_scenes if os.path.basename(d) not in SCANNETPP_EVAL_HOLDOUT]

    def __len__(self):
        return len(self.scenes)


class _Re10kIndex:
    """Each entry is one extracted re10k scene directory.

    NOTE: the exact extracted layout is verified against the data after extract_split_zip.sh runs;
    we scan for per-scene dirs and let _load_re10k resolve the modality paths. Kept tolerant so a
    layout mismatch raises a clear error rather than silently yielding nothing.
    """

    def __init__(self, root: str):
        # extracted layout is nested: processed_re10k/processed_re10k/<scene>/{rgb,cam};
        # SAM2 instance masks live in a sibling processed_re10k/processed_re10k/sam2_results/<scene>/auto_masks.json
        base = os.path.join(root, "processed_re10k", "processed_re10k")
        self.mask_root = os.path.join(base, "sam2_results")
        self.scenes = sorted(
            d for d in glob(os.path.join(base, "*"))
            if os.path.isdir(os.path.join(d, "rgb"))
            and os.path.basename(d) != "sam2_results"
            and os.path.exists(os.path.join(self.mask_root, os.path.basename(d), "auto_masks.json"))
        )

    def __len__(self):
        return len(self.scenes)


class _ScannetIndex:
    """ScanNet200 train scenes, projected to per-frame 2D instance masks by tools/scannet_project.py.
    Lives under a SEPARATE root (not InsScene-15K), layout: <scannet_root>/frames/train/<scene>/
    {rgb/<idx>.jpg, inst/<idx>.npz['inst'] (H,W int16, 0=bg, 1..K), cam/<idx>.npz[K,pose(cam2world)]}.
    No depth (seg-only training uses zeros, like re10k)."""

    def __init__(self, scannet_root: str, split: str = "train"):
        base = os.path.join(scannet_root, "frames", split)
        self.scenes = sorted(
            d for d in glob(os.path.join(base, "*"))
            if os.path.isdir(os.path.join(d, "rgb")) and os.path.isdir(os.path.join(d, "inst"))
        )

    def __len__(self):
        return len(self.scenes)


# ------------------------------------------------------------------------------------------------
# dataset
# ------------------------------------------------------------------------------------------------
class InsSceneDataset(Dataset):
    def __init__(
        self,
        root: str,
        subsets: Optional[List[str]] = None,
        num_frames: Optional[int] = None,        # if set, use a FIXED count (legacy / eval)
        min_frames: int = 2,                     # IGGT samples 1-12, SegVGGT 2-24 frames per scene;
        max_frames: int = 12,                    # we randomize #frames per sample within [min,max].
        resolution: int = 512,
        max_instances: int = 100,
        frame_sampling: str = "random",  # "random" | "consecutive"
        min_instance_pixels: int = 50,
        split: str = "all",              # "all" | "train" | "test"
        test_frac: float = 0.05,         # last `test_frac` of each subset (sorted) held out as test
        scannet_root: Optional[str] = None,   # separate root for projected ScanNet200 (the "scannet" subset)
        subset_sample_weights: Optional[dict] = None,  # relative sampling frequency per subset (e.g.
                                                       # {scannet: 0.3} -> scannet scenes drawn 0.3x as often).
                                                       # Realized by integer replication; default uniform.
        aspect_resize: bool = False,     # True -> official VGGT balanced (aspect-preserving) resize like IGGT/video
                                         # subsets, instead of distorting to a square (square = legacy segmenter default).
        augment: bool = False,           # train-only PHOTOMETRIC aug (color jitter + random grayscale), RGB only.
                                         # NO geometric aug (would break multi-view/geometry consistency). matches
                                         # VGGT-Omega supervised-stage jitter (b/c/s=0.5,hue=0.1). eval keeps False.
    ):
        super().__init__()
        self.split = split
        self.test_frac = test_frac
        self.root = root
        # Random frame count per sample (mirrors IGGT/SegVGGT). A fixed num_frames overrides the range.
        self.num_frames = num_frames
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.resolution = resolution
        self.aspect_resize = aspect_resize
        self._cur_thtw = (resolution, resolution)
        self.augment = augment
        self._aug = None
        if augment:                                  # photometric only (RGB); per-frame independent (matches VGGT-Omega)
            import torchvision.transforms as _T
            self._aug = _T.Compose([_T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                                    _T.RandomGrayscale(p=0.2)])
        self.max_instances = max_instances
        self.frame_sampling = frame_sampling
        self.min_instance_pixels = min_instance_pixels
        subsets = ["infinigen", "scannetpp_v2"] if subsets is None else subsets   # [] stays empty (video-only eval)

        self.items = []  # (subset, payload)
        if "infinigen" in subsets:
            self._add(("infinigen", p) for p in _InfinigenIndex(root).zips)
        if "scannetpp_v2" in subsets:
            self._add(("scannetpp_v2", p) for p in _ScanNetppIndex(root).scenes)
        if "re10k" in subsets:
            self._add(("re10k", p) for p in _Re10kIndex(root).scenes)
        if "scannet" in subsets:
            if not scannet_root:
                raise ValueError("subset 'scannet' requires data.scannet_root (projected ScanNet200 dir)")
            self._add(("scannet", p) for p in _ScannetIndex(scannet_root).scenes)

        # mixing ratio: realize per-subset sampling frequency by integer replication of the item list
        # (DDP-safe: the normal DistributedSampler then shuffles/shards the replicated list).
        if subset_sample_weights:
            # weight == #copies of each scene in the flat item list (>=1), SAME convention as the 4D
            # video_weights path (seg4d_video: reps=round(w)). No hidden scale factor: weight=1 -> 1 copy.
            rep = []
            for s, p in self.items:
                mult = max(1, int(round(subset_sample_weights.get(s, 1.0))))
                rep.extend([(s, p)] * mult)
            self.items = rep
        if not self.items and not getattr(self, "_defer_empty", False):
            raise RuntimeError(
                f"No samples found under {root}. For scannetpp, extract the split zip first "
                f"(scripts/extract_split_zip.sh)."
            )

    def _add(self, pairs):
        """Apply the per-subset train/test holdout: the LAST `test_frac` of the sorted scenes of this
        subset are the test split (deterministic, no leakage between train and test)."""
        pairs = sorted(pairs)
        if self.split == "all" or self.test_frac <= 0:
            self.items.extend(pairs)
            return
        n_test = max(1, int(round(len(pairs) * self.test_frac)))
        chosen = pairs[-n_test:] if self.split == "test" else pairs[:-n_test]
        self.items.extend(chosen)

    def __len__(self):
        return len(self.items)

    # --- frame sampling -----------------------------------------------------------------------
    def _n_frames(self) -> int:
        if self.num_frames is not None:
            return self.num_frames
        return int(np.random.randint(self.min_frames, self.max_frames + 1))

    def _pick(self, all_ids: List, n: int) -> List:
        all_ids = sorted(all_ids)
        if len(all_ids) <= n:
            return all_ids
        if self.frame_sampling == "consecutive":
            return all_ids[: n]                              # deterministic (eval): first n consecutive
        idx = sorted(np.random.choice(len(all_ids), size=n, replace=False).tolist())
        return [all_ids[i] for i in idx]

    # --- image / map resizing -----------------------------------------------------------------
    def _balanced_hw(self, h, w):
        from vggt_omega.utils.load_fn import _balanced_target_shape   # official VGGT aspect-preserving resize
        return _balanced_target_shape(h / max(w, 1), self.resolution, 16)   # (th, tw)

    def _resize_rgb(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB")
        if self._aug is not None:                      # train-only photometric aug (RGB); masks/depth untouched
            img = self._aug(img)
        if self.aspect_resize:
            W, H = img.size; th, tw = self._balanced_hw(H, W); self._cur_thtw = (th, tw)  # same target reused by maps
        else:
            th = tw = self.resolution
        img = img.resize((tw, th), Image.BILINEAR)
        return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

    def _resize_map(self, arr: np.ndarray, is_depth: bool) -> np.ndarray:
        # nearest for labels/depth to avoid mixing ids / interpolating across discontinuities
        th, tw = self._cur_thtw if self.aspect_resize else (self.resolution, self.resolution)  # match rgb's balanced target
        im = Image.fromarray(arr).resize((tw, th), Image.NEAREST)
        return np.asarray(im)

    # --- builders -----------------------------------------------------------------------------
    def _finalize(self, rgbs, depths, insts, Ks, Ts):
        images = torch.stack(rgbs, 0)                                  # (S,3,H,W)
        depth = torch.from_numpy(np.stack(depths, 0)).float()          # (S,H,W)
        depth_valid = (depth > 0) & (depth < INVALID_DEPTH)
        depth = torch.where(depth_valid, depth, torch.zeros_like(depth))

        inst = np.stack(insts, 0).astype(np.int64)                     # (S,H,W) raw ids (0=bg)
        # remap nonzero ids -> contiguous 1..K across the sampled frames
        uniq = np.unique(inst)
        uniq = uniq[uniq > 0]
        # drop tiny instances (across all frames)
        keep = [u for u in uniq if (inst == u).sum() >= self.min_instance_pixels]
        keep = keep[: self.max_instances]
        remap = {old: i + 1 for i, old in enumerate(keep)}
        out = np.zeros_like(inst)
        for old, new in remap.items():
            out[inst == old] = new
        inst_t = torch.from_numpy(out)

        sample = {
            "images": images,
            "depth": depth,
            "depth_valid": depth_valid,
            "inst_ids": inst_t,                                        # (S,H,W) long
            "num_instances": int(len(keep)),
        }
        if Ks and Ks[0] is not None:
            sample["intrinsics"] = torch.from_numpy(np.stack(Ks, 0)).float()
        if Ts and Ts[0] is not None:
            sample["pose"] = torch.from_numpy(np.stack(Ts, 0)).float()
        return sample

    def __getitem__(self, i: int):
        # robustness: skip an occasional corrupt sample by retrying another, but BOUND the retries
        # (a systematic loader bug must surface as an error, not recurse/loop until OOM).
        last_exc = None
        for _ in range(8):
            subset, payload = self.items[i]
            try:
                if subset == "infinigen":
                    return {"subset": subset, **self._load_infinigen(payload)}
                if subset == "re10k":
                    return {"subset": subset, **self._load_re10k(payload)}
                if subset == "scannet":
                    return {"subset": subset, **self._load_scannet(payload)}
                return {"subset": subset, **self._load_scannetpp(payload)}
            except Exception as e:
                last_exc = e
                i = int(np.random.randint(0, len(self)))
        raise RuntimeError(f"InsSceneDataset: {8} consecutive load failures; last: {last_exc!r}")

    # --- scannet (ScanNet200, projected per-frame 2D masks) -----------------------------------
    def _load_scannet(self, scene_dir: str):
        """scene_dir/{rgb/<idx>.jpg, inst/<idx>.npz['inst'] (H,W int16, 0=bg, 1..K, 3D-consistent),
        cam/<idx>.npz[K(3x3), pose(4x4 cam2world)]}. No depth GT (seg-only) -> zeros, like re10k."""
        rgb_files = sorted(glob(os.path.join(scene_dir, "rgb", "*.jpg")))
        stems = [os.path.basename(p)[:-4] for p in rgb_files]
        if not stems:
            raise FileNotFoundError(f"no rgb frames in {scene_dir}")
        pos = self._pick(list(range(len(stems))), self._n_frames())
        rgbs, depths, insts, Ks, Ts = [], [], [], [], []
        for p in pos:
            st = stems[p]
            rgbs.append(self._resize_rgb(Image.open(os.path.join(scene_dir, "rgb", st + ".jpg"))))
            idm = np.load(os.path.join(scene_dir, "inst", st + ".npz"))["inst"].astype(np.int32)
            insts.append(self._resize_map(idm, is_depth=False))
            depths.append(np.zeros(self._cur_thtw, np.float32))  # no GT depth
            cam = np.load(os.path.join(scene_dir, "cam", st + ".npz"))
            Ks.append(cam["K"].astype(np.float32))
            Ts.append(cam["pose"].astype(np.float32))
        return self._finalize(rgbs, depths, insts, Ks, Ts)

    # --- infinigen (read from zip) ------------------------------------------------------------
    def _load_infinigen(self, zip_path: str):
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            seg_names = [n for n in names if "/ObjectSegmentation/" in n and n.endswith(".npy")]
            # parse frame id = first integer field after the modality prefix
            def fid(n):
                m = re.search(r"ObjectSegmentation_(\d+)_", os.path.basename(n))
                return int(m.group(1)) if m else -1
            id2seg = {fid(n): n for n in seg_names if fid(n) >= 0}
            chosen = self._pick(list(id2seg.keys()), self._n_frames())

            rgbs, depths, insts, Ks, Ts = [], [], [], [], []
            for f in chosen:
                seg = id2seg[f]
                stem = os.path.basename(seg).replace("ObjectSegmentation_", "").replace(".npy", "")  # f_0_take_0
                def grab(mod, ext):
                    p = f"frames/{mod}/camera_0/{mod}_{stem}.{ext}"
                    return zf.read(p)
                rgb = Image.open(io.BytesIO(grab("Image", "png")))
                rgbs.append(self._resize_rgb(rgb))
                d = np.load(io.BytesIO(grab("Depth", "npy")))
                depths.append(self._resize_map(d.astype(np.float32), is_depth=True))
                s = np.load(io.BytesIO(zf.read(seg)))
                insts.append(self._resize_map(s.astype(np.int32), is_depth=False))
                cam = np.load(io.BytesIO(grab("camview", "npz")))
                Ks.append(cam["K"].astype(np.float32))
                Ts.append(cam["T"].astype(np.float32))
        return self._finalize(rgbs, depths, insts, Ks, Ts)

    # --- scannetpp (read from extracted tree) -------------------------------------------------
    def _load_scannetpp(self, scene_dir: str):
        img_dir = os.path.join(scene_dir, "images")
        frames = sorted(glob(os.path.join(img_dir, "frame_*.jpg")))
        ids = [os.path.basename(p)[:-4] for p in frames]  # frame_XXXXXX
        chosen = self._pick(ids, self._n_frames())

        # metadata stores per-frame arrays (intrinsics (N,3,3), trajectories/c2w (N,4,4)) indexed
        # positionally by the parallel `images` name array -- resolve frame -> row by name.
        meta_path = os.path.join(scene_dir, "scene_iphone_metadata.npz")
        K_all = T_all = name2idx = None
        if os.path.exists(meta_path):
            meta = np.load(meta_path, allow_pickle=True)
            if "intrinsics" in meta:
                K_all = np.asarray(meta["intrinsics"], dtype=np.float32)          # (N,3,3)
            for pk in ("trajectories", "poses", "pose", "extrinsics", "T"):
                if pk in meta:
                    T_all = np.asarray(meta[pk], dtype=np.float32)                 # (N,4,4)
                    break
            if "images" in meta:
                name2idx = {os.path.splitext(str(x))[0]: j for j, x in enumerate(meta["images"])}

        rgbs, depths, insts, Ks, Ts = [], [], [], [], []
        for fr in chosen:
            rgbs.append(self._resize_rgb(Image.open(os.path.join(img_dir, fr + ".jpg"))))
            dp = os.path.join(scene_dir, "depth", fr + ".png")
            d = np.asarray(Image.open(dp)).astype(np.float32) / 1000.0 if os.path.exists(dp) else np.zeros((1, 1), np.float32)
            depths.append(self._resize_map(d, is_depth=True))
            ins = self._read_ins(os.path.join(scene_dir, "refined_ins_ids", fr))
            insts.append(self._resize_map(ins, is_depth=False))
            j = name2idx.get(fr) if name2idx is not None else None
            Ks.append(K_all[j] if (K_all is not None and j is not None) else None)
            Ts.append(T_all[j] if (T_all is not None and j is not None) else None)
        return self._finalize(rgbs, depths, insts, Ks, Ts)

    # --- re10k (read from extracted tree) -----------------------------------------------------
    def _load_re10k(self, scene_dir: str):
        """re10k: rgb/<ts>.png + cam/<ts>.npz (intrinsics,pose), NO depth. SAM2 instance masks live in a
        sibling sam2_results/<scene>/auto_masks.json as SA-V 'masklet' RLEs (mirrors IGGT re10k.py):
        masklet is indexed BY FRAME (len == #frames); masklet[f] is a list of K RLEs decoding to
        (H,W,K); channel j is the SAME instance across frames -> 3D-consistent ids."""
        import json as _json
        import pycocotools.mask as _mask_util

        rgb_files = sorted(glob(os.path.join(scene_dir, "rgb", "*.png")))
        stems = [os.path.basename(p)[:-4] for p in rgb_files]               # timestamps, sorted == masklet order
        name = os.path.basename(scene_dir)
        mask_json = os.path.join(os.path.dirname(scene_dir), "sam2_results", name, "auto_masks.json")
        masklet = _json.load(open(mask_json))["masklet"]                    # len == #frames
        n = min(len(stems), len(masklet))
        # pick frame POSITIONS (masklet is positional), then map to stems
        pos = self._pick(list(range(n)), self._n_frames())

        rgbs, depths, insts, Ks, Ts = [], [], [], [], []
        for p in pos:
            st = stems[p]
            rgbs.append(self._resize_rgb(Image.open(os.path.join(scene_dir, "rgb", st + ".png"))))
            m = _mask_util.decode(masklet[p])                               # (H,W,K) uint8, channel == instance
            if m.ndim == 2:
                m = m[..., None]
            idm = np.zeros(m.shape[:2], np.int32)
            for j in range(m.shape[2]):
                idm[m[:, :, j] > 0] = j + 1                                 # channel j -> consistent id j+1
            insts.append(self._resize_map(idm, is_depth=False))
            depths.append(np.zeros(self._cur_thtw, np.float32))  # no GT depth -> all invalid
            cam = np.load(os.path.join(scene_dir, "cam", st + ".npz"))
            Ks.append(cam["intrinsics"].astype(np.float32))
            Ts.append(cam["pose"].astype(np.float32))
        return self._finalize(rgbs, depths, insts, Ks, Ts)

    @staticmethod
    def _read_ins(stem: str) -> np.ndarray:
        for ext in (".png", ".npy"):
            p = stem + ext
            if os.path.exists(p):
                a = np.load(p) if ext == ".npy" else np.asarray(Image.open(p))
                a = a.astype(np.int32)
                a[a < 0] = 0
                a[a >= 65535] = 0  # common ignore sentinel
                return a
        raise FileNotFoundError(f"no refined_ins_ids for {stem}")

    @staticmethod
    def _meta_K(meta, fr):
        # metadata key layout is dataset-specific; return None if not resolvable (geometry loss skipped)
        if meta is None:
            return None
        for key in ("intrinsics", "K", "intrinsic"):
            if key in meta:
                v = meta[key]
                return (v.item().get(fr) if v.dtype == object else v).astype(np.float32) if hasattr(v, "dtype") else None
        return None

    @staticmethod
    def _meta_T(meta, fr):
        if meta is None:
            return None
        for key in ("poses", "pose", "extrinsics", "T"):
            if key in meta:
                v = meta[key]
                return (v.item().get(fr) if v.dtype == object else v).astype(np.float32) if hasattr(v, "dtype") else None
        return None


# ------------------------------------------------------------------------------------------------
# collate
# ------------------------------------------------------------------------------------------------
def collate_fn(batch):
    """Stack fixed-size tensors; keep per-sample instance metadata as lists.

    NOTE: with random per-sample frame counts, different samples have different S, so they cannot be
    stacked. Use batch_size=1 per GPU (recommended; scale via more GPUs / grad-accum). DDP across
    GPUs is fine because each rank carries its own (possibly different-S) batch.
    """
    if len(batch) > 1:
        s0 = batch[0]["images"].shape[0]
        if any(b["images"].shape[0] != s0 for b in batch):
            raise ValueError(
                "Variable frame counts in a batch cannot be stacked. Use batch_size=1 per GPU "
                "with random frame sampling, or set a fixed num_frames for batched loading."
            )
    out = {
        "images": torch.stack([b["images"] for b in batch], 0),        # (B,S,3,H,W)
        "subset": [b["subset"] for b in batch],
    }
    # modality keys may be absent in eval batches (e.g. tracking has no depth; geometry has no inst)
    if all("depth" in b for b in batch):
        out["depth"] = torch.stack([b["depth"] for b in batch], 0)     # (B,S,H,W)
        out["depth_valid"] = torch.stack([b["depth_valid"] for b in batch], 0)
    if all("inst_ids" in b for b in batch):
        out["inst_ids"] = torch.stack([b["inst_ids"] for b in batch], 0)  # (B,S,H,W)
        out["num_instances"] = [b["num_instances"] for b in batch]
    if "intrinsics" in batch[0] and all("intrinsics" in b for b in batch):
        out["intrinsics"] = torch.stack([b["intrinsics"] for b in batch], 0)
        out["pose"] = torch.stack([b["pose"] for b in batch], 0)
    if all("ignore" in b for b in batch):                              # Seg4D: per-pixel ignore (mask-loss)
        out["ignore"] = torch.stack([b["ignore"] for b in batch], 0)   # (B,S,H,W) bool
    if any("exhaustive" in b for b in batch):                          # Seg4D: partial-annotation flag
        out["exhaustive"] = [bool(b.get("exhaustive", True)) for b in batch]   # missing (3D/InsScene) -> dense, robust to mixed BS>1 batches
    return out
