"""Locate the (un-pip-installed) vggt_omega package so we can import the backbone."""
from __future__ import annotations

import os
import sys

# The released VGGT-Omega code lives here (not pip-installed). Override with $VGGT_OMEGA_ROOT.
VGGT_OMEGA_ROOT = os.environ.get(
    "VGGT_OMEGA_ROOT",
    "/PATH/TO/codebase/vggt-omega",
)

# Default checkpoint for the 1B / 512px variant.
VGGT_OMEGA_CKPT = os.environ.get(
    "VGGT_OMEGA_CKPT",
    "/PATH/TO/checkpoints/VGGT-Omega/vggt_omega_1b_512.pt",
)


def ensure_vggt_omega_on_path() -> str:
    """Prepend the vggt-omega repo root to sys.path so `import vggt_omega` works."""
    if VGGT_OMEGA_ROOT not in sys.path:
        sys.path.insert(0, VGGT_OMEGA_ROOT)
    return VGGT_OMEGA_ROOT
