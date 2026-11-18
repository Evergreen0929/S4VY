"""Lossless host storage for cached inference features (no persistent disk cache)."""
from collections.abc import Sequence


class HostTokens(Sequence):
    """Upload only the requested layer, back to the device it came from.

    Call under no_grad after the backbone finishes. This reduces head-stage GPU
    residency, not the peak inside the backbone, and needs enough CPU RAM for
    all cached layers. Blocking CPU transfers cannot be CUDA-graph captured.
    """

    def __init__(self, tokens):
        self._values = [None if t is None else (t.cpu(), t.device) for t in tokens]

    def __len__(self):
        return len(self._values)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        entry = self._values[index]
        return None if entry is None else entry[0].to(entry[1])
