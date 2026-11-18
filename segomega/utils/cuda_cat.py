"""Concatenate large CUDA features without the affected 32-bit cat kernel."""

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.overrides import TorchFunctionMode


_CUDA_CAT_BYTE_LIMIT = 1 << 32


def safe_cuda_cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    """Use explicit slice copies for homogeneous CUDA outputs of at least 4 GiB.

    The CUDA cat kernel in the NVIDIA 25.10 PyTorch build can silently corrupt
    these outputs even when their element count is 32-bit indexable. Small,
    CPU, mixed-dtype and invalid inputs retain native torch.cat behavior. The
    large-tensor path produces a contiguous output and supports autograd.
    """
    tensors = tuple(tensors)
    if not tensors:
        return torch.cat(tensors, dim=dim)
    first = tensors[0]
    if (
        not first.is_cuda
        or not -first.ndim <= dim < first.ndim
        or any(t.device != first.device or t.dtype != first.dtype or
               t.layout != torch.strided or t.ndim != first.ndim for t in tensors)
        or sum(t.numel() for t in tensors) * first.element_size() < _CUDA_CAT_BYTE_LIMIT
    ):
        return torch.cat(tensors, dim=dim)
    dim %= first.ndim
    if any(t.shape[d] != first.shape[d] for t in tensors
           for d in range(first.ndim) if d != dim):
        return torch.cat(tensors, dim=dim)
    shape = list(first.shape)
    shape[dim] = sum(t.shape[dim] for t in tensors)
    result = first.new_empty(shape)
    start = 0
    for tensor in tensors:
        size = tensor.shape[dim]
        result.narrow(dim, start, size).copy_(tensor)
        start += size
    return result


class SafeCudaCat(TorchFunctionMode):
    """Protect torch.cat within a backbone forward without modifying PyTorch.

    This also covers the separately installed VGGT-Omega package. The mode
    is scoped to the current thread/context and works with compiled blocks
    and CUDA graph capture. Calls with an explicit output retain native cat
    semantics; VGGT-Omega's feature concatenations do not use that overload.
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.cat and kwargs.get("out") is None:
            return safe_cuda_cat(*args, **{k: v for k, v in kwargs.items() if k != "out"})
        return func(*args, **kwargs)
