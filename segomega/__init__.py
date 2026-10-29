"""S4VY: class-agnostic query-based 4D instance segmentation on the VGGT-Omega backbone.

Design (Option B, decided with the user):
  - Backbone: VGGT-Omega (register-attention geometry transformer), reused as a frozen / LoRA / low-LR
    feature extractor that also keeps its depth & camera heads for joint geometry supervision.
  - Head:  Mask2Former-style transformer decoder with N learnable *instance queries* that are shared
           across the S input frames (so one query == one 3D-consistent instance). Each query predicts
             * a per-frame binary mask  (via dot-product with a pixel-decoder feature map), and
             * a class-agnostic objectness logit  {object, no-object/varnothing}.
  - Supervision: Hungarian matching (mask BCE + dice + objectness) -- ported from Mask2Former conventions
           because SegVGGT's training/loss code is NOT in its public (eval-only) release.

References folded in:
  - VGGT-Omega backbone API: codebase/vggt-omega/vggt_omega/{models/aggregator.py, models/heads/dense_head.py}
  - SegVGGT query mechanism / mask einsum / objectness-from-background: codebase/SegVGGT/segvggt/{models,heads,eval}
  - IGGT data + multi-view-consistent instance ids + geometry losses: codebase/IGGT_official
"""

from .paths import VGGT_OMEGA_ROOT, ensure_vggt_omega_on_path  # noqa: F401

__all__ = ["VGGT_OMEGA_ROOT", "ensure_vggt_omega_on_path"]
