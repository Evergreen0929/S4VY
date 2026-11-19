"""Saved instance IDs and display colors used by the comparison diagnostics."""
import argparse

import numpy as np
from PIL import Image


def read_idmap(path, size=None):
    """Read uint16 PNGs or legacy packed RGB (ID = R + 256*G).

    Optional size is (width, height); resizing always uses nearest neighbors.
    """
    with Image.open(path) as image:
        ids = np.array(image)
    if ids.ndim == 3 and ids.shape[-1] == 3:
        ids = ids[..., 0].astype(np.uint16) + ids[..., 1].astype(np.uint16) * 256
    if ids.ndim != 2 or ids.min() < 0 or ids.max() > 65535:
        raise ValueError(f"Expected uint16 or packed RGB instance IDs: {path}")
    ids = ids.astype(np.uint16)
    if size is not None and ids.shape != (size[1], size[0]):
        ids = np.array(Image.fromarray(ids).resize(size, Image.Resampling.NEAREST))
    return ids


def parse_color(value):
    """argparse type for ID=R,G,B display overrides."""
    try:
        ident, color = value.split("=")
        ident, color = int(ident), tuple(int(c) for c in color.split(","))
        if not 1 <= ident <= 65535 or len(color) != 3 or any(c < 0 or c > 255 for c in color):
            raise ValueError
        return ident, color
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected ID=R,G,B (ID 1..65535, RGB 0..255)") from error


def palette(overrides=()):
    ids = np.arange(65536, dtype=np.uint32)
    hashes = ids * np.uint32(0x45d9f3b)
    hashes ^= hashes >> 16
    colors = np.stack([70 + (hashes & 255) * .72,
                       70 + ((hashes >> 8) & 255) * .72,
                       70 + ((hashes >> 16) & 255) * .72], -1)
    for ident, color in overrides:
        colors[ident] = color
    return colors


def overlay_ids(rgb, ids, colors, alpha=.5):
    """Color only the display; neither the RGB input nor IDs are mutated."""
    out = rgb.copy()
    foreground = ids > 0
    out[foreground] = np.rint((1 - alpha) * rgb[foreground] + alpha * colors[ids[foreground]]).clip(0, 255).astype(np.uint8)
    return out
