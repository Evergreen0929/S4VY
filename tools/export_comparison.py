#!/usr/bin/env python
"""Export [source | S4VY offline; S4VY streaming | SAM] from saved masks.

Requires ffmpeg, OpenCV, numpy and Pillow. Each directory must contain one PNG
per source frame, sorted in temporal order. Masks may have different spatial
resolutions; IDs are resized with nearest neighbors. This does not run models
or modify masks. Supply labels that describe how the saved predictions were made.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from segomega.utils.mask_io import overlay_ids, palette, parse_color, read_idmap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    for name in ("offline", "stream", "sam"):
        parser.add_argument(f"--{name}", type=Path, required=True, help="directory of ordered instance-ID PNGs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offline-label", default="S4VY offline")
    parser.add_argument("--stream-label", default="S4VY streaming (saved reference)")
    parser.add_argument("--sam-label", default="SAM (saved reference)")
    parser.add_argument("--offline-color", type=parse_color, action="append", default=[], metavar="ID=R,G,B")
    parser.add_argument("--alpha", type=float, default=.5)
    parser.add_argument("--duration", type=float, help="override output duration in seconds (constant frame rate)")
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1 or (args.duration is not None and args.duration <= 0):
        parser.error("alpha must be 0..1 and duration must be positive")
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")
    paths = [sorted(getattr(args, name).glob("*.png")) for name in ("offline", "stream", "sam")]
    count = len(paths[0])
    if not count or any(len(p) != count for p in paths):
        parser.error("All three mask directories must contain the same nonzero number of frames")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        parser.error(f"Cannot open {args.video}")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = count / args.duration if args.duration else source_fps
    if fps <= 0:
        parser.error("Cannot determine source FPS; provide --duration")
    ok, bgr = cap.read()
    if not ok:
        parser.error("Video has no readable frames")
    height, width = bgr.shape[:2]
    header, footer = 64, 48
    canvas_width = (2 * width + 1) // 2 * 2
    canvas_height = (2 * (height + header) + footer + 1) // 2 * 2
    colors = palette()
    offline_colors = palette(args.offline_color)
    labels = ["Source video", args.offline_label, args.stream_label, args.sam_label]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-n", "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", f"{canvas_width}x{canvas_height}", "-framerate", str(fps), "-i", "pipe:0",
               "-i", str(args.video), "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-threads", "8",
               "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
               "-t", str(count / fps), "-movflags", "+faststart", str(args.out)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        index = 0
        while ok:
            if index >= count:
                raise ValueError("Source has more frames than the mask directories")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (height, width):
                raise ValueError("Source resolution changed within video")
            panels = [rgb]
            for panel, files in enumerate(paths):
                ids = read_idmap(files[index], (width, height))
                panels.append(overlay_ids(rgb, ids, offline_colors if panel == 0 else colors, args.alpha))
            canvas = np.full((canvas_height, canvas_width, 3), (16, 21, 30), dtype=np.uint8)
            for panel, (label, pixels) in enumerate(zip(labels, panels)):
                x, y = (panel % 2) * width, (panel // 2) * (height + header)
                canvas[y + header:y + header + height, x:x + width] = pixels
                cv2.putText(canvas, label, (x + 12, y + 39), cv2.FONT_HERSHEY_SIMPLEX, .8, (229, 237, 249), 2)
            cv2.putText(canvas, f"Frame {index + 1}/{count} | Independent colors per model; display overrides preserve IDs",
                        (12, canvas_height - 16), cv2.FONT_HERSHEY_SIMPLEX, .6, (170, 185, 204), 1)
            proc.stdin.write(canvas.tobytes())
            index += 1
            ok, bgr = cap.read()
        if index != count:
            raise ValueError(f"Decoded {index} source frames but found {count} masks per model")
    except BaseException:
        proc.kill()
        raise
    finally:
        cap.release()
        proc.stdin.close()
        returncode = proc.wait()
    if returncode:
        raise RuntimeError(f"ffmpeg failed with exit code {returncode}")
    manifest = dict(video=str(args.video), frames=count, fps=fps, duration_seconds=count / fps,
                    labels=labels, mask_directories=[str(args.offline), str(args.stream), str(args.sam)],
                    offline_color_overrides=dict(args.offline_color), alpha=args.alpha,
                    masks_modified=False, alignment="One sorted PNG per decoded source frame; no temporal resampling")
    args.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
