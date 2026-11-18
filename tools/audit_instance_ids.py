#!/usr/bin/env python
"""Compare object/arm IDs inside eroded reference interiors, not ground truth.

Run on saved predictions to distinguish similar display colors from shared IDs.
Candidate filenames must match reference filenames; missing frames are reported
as missing, allowing temporally subsampled runs without silently shifting indices.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from segomega.utils.mask_io import read_idmap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--object-id", type=int, required=True, help="object ID in reference")
    parser.add_argument("--arm-id", type=int, required=True, help="arm ID in reference")
    parser.add_argument("--erosion", type=int, default=9, help="kernel width at reference resolution")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.erosion < 1 or args.object_id == args.arm_id or min(args.object_id, args.arm_id) <= 0:
        parser.error("Use a positive erosion kernel and distinct foreground reference IDs")
    files = sorted(args.reference.glob("*.png"))
    if not files:
        parser.error("Reference directory has no PNG masks")
    kernel = np.ones((args.erosion, args.erosion), np.uint8)
    rows = []
    for path in files:
        candidate = args.candidate / path.name
        if not candidate.exists():
            rows.append(dict(frame=path.name, status="missing"))
            continue
        reference = read_idmap(path)
        actual = read_idmap(candidate, (reference.shape[1], reference.shape[0]))
        core = cv2.erode((reference == args.object_id).astype(np.uint8), kernel).astype(bool)
        arm = cv2.erode((reference == args.arm_id).astype(np.uint8), kernel).astype(bool)
        if not core.any() or not arm.any():
            rows.append(dict(frame=path.name, status="empty_reference_interior"))
            continue
        ids, counts = np.unique(actual[core], return_counts=True)
        arm_ids, arm_counts = np.unique(actual[arm], return_counts=True)
        object_id, arm_id = int(ids[counts.argmax()]), int(arm_ids[arm_counts.argmax()])
        rows.append(dict(frame=path.name, status="compared", object_majority_id=object_id, arm_majority_id=arm_id,
                         candidate_shared_foreground_id=object_id == arm_id and arm_id != 0,
                         object_core_arm_fraction=float(np.mean(actual[core] == arm_id)) if arm_id else None,
                         object_core_ids={str(int(k)): int(v) for k, v in zip(ids, counts)}))
    result = dict(note="Cross-run reference, not ground truth. Shared foreground IDs flag frames for visual review; occlusion and reference errors can also cause disagreement.",
                  reference=str(args.reference), candidate=str(args.candidate), reference_object_id=args.object_id,
                  reference_arm_id=args.arm_id, erosion=args.erosion, frames=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
