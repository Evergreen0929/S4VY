"""Inference-only speedups for the offline segmenter, ported from the S4VY streaming service.

    report = optimize_for_inference(model)         # after load_state_dict(...), .cuda().eval()
    runner = GraphRunner(model)                    # optional: CUDA graph per input shape
    preds = runner(images)                         # (1,S,3,H,W) in [0,1]; same keys as model(images)

Transforms, all opt-out:
  fast_mask_downsample  the decoder's per-layer mask -> attention-mask bilinear downsample as four
                 strided slices; bitwise identical, and was ~30% of a long clip's forward
  merge_lora     fold every backbone LoRA adapter into its base Linear (one matmul instead of three)
  bf16_weights   keep BF16 copies of the frozen backbone matmul weights, so autocast stops
                 re-casting ~2 GB of FP32 weights on every forward; registered parameters stay FP32
  compile        torch.compile the DINO patch-embed blocks and the frame / inter-frame blocks
                 (fixed shapes, precision-preserving Inductor options)

fast_mask_downsample and GraphRunner are exact; pass exact=True to use only those. The others
change floating-point order, so masks can differ at threshold ties (~99.7% of pixels keep their
instance). GraphRunner replays a captured CUDA graph while its input shape remains cached. Cache
misses pay warm-up and capture, plus compilation for new shapes in full mode, so graphs only pay
off when shapes repeat before eviction. Outputs of a
replay alias the graph's static buffers and are overwritten by the next call on that shape --
clone them if you keep more than one.
"""
from __future__ import annotations

from collections import OrderedDict
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.lora import LoRALinear

COMPILE_OPTIONS = {"triton.cudagraphs": False, "max_autotune": False,
                   "force_same_precision": True, "emulate_precision_casts": True}


def merge_lora(model: nn.Module) -> int:
    """Replace each LoRALinear with its base Linear holding W + scaling * B @ A (FP32)."""
    merged = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if not isinstance(child, LoRALinear):
                continue
            base = child.base
            if child.enabled:
                with torch.no_grad():
                    delta = (child.lora_b.weight.float() @ child.lora_a.weight.float()) * child.scaling
                    base.weight.add_(delta.to(base.weight.dtype))
            setattr(parent, name, base)
            merged += 1
    return merged


def _bf16_linear(self, x):
    return F.linear(x, self._bf16_weight, self._bf16_bias)


def _bf16_conv(self, x):
    return self._conv_forward(x, self._bf16_weight, self._bf16_bias)


def cache_bf16_weights(backbone: nn.Module) -> int:
    """BF16 operands for backbone Linear/Conv2d, matching what BF16 autocast would compute.

    Only valid under BF16 autocast with a frozen model; the backbone always runs that way.
    """
    cached = 0
    for module in backbone.modules():
        if type(module) not in (nn.Linear, nn.Conv2d) and type(module).__name__ != "LinearKMaskedBias":
            continue
        with torch.no_grad():
            bias = module.bias
            if bias is not None and hasattr(module, "bias_mask"):
                bias = bias * module.bias_mask.to(bias.dtype)          # FP32 mask first, as the original
            module.register_buffer("_bf16_weight", module.weight.detach().to(torch.bfloat16), persistent=False)
            module.register_buffer("_bf16_bias", None if bias is None else bias.detach().to(torch.bfloat16),
                                   persistent=False)
        module.forward = MethodType(_bf16_conv if isinstance(module, nn.Conv2d) else _bf16_linear, module)
        cached += 1
    return cached


def _attn_mask_exact_downsample(self, mask_logits, gh, gw):
    """Bitwise-identical _attn_mask_from_logits for the 4x case (mask grid H/4, patch grid H/16).

    Bilinear, align_corners=False, at exactly 1/4 scale samples source coordinate 4i+1.5: a 0.5/0.5
    blend of pixels 4i+1 and 4i+2 on each axis. The expression below reproduces the CUDA kernel's
    order (horizontal lerps, then vertical), so results are bitwise equal, ~60x faster.
    """
    B, N, S, h, w = mask_logits.shape
    if h != 4 * gh or w != 4 * gw:
        return type(self)._attn_mask_from_logits(self, mask_logits, gh, gw)
    m = mask_logits
    m = 0.5 * (0.5 * m[..., 1::4, 1::4] + 0.5 * m[..., 1::4, 2::4]) \
        + 0.5 * (0.5 * m[..., 2::4, 1::4] + 0.5 * m[..., 2::4, 2::4])
    m = m.reshape(B, N, S * gh * gw)
    block = ~(m.sigmoid() > self.mask_attn_threshold)
    block = block & ~block.all(dim=-1, keepdim=True)
    return block.unsqueeze(1).expand(-1, self.num_heads, -1, -1).reshape(B * self.num_heads, N, S * gh * gw)


def compile_backbone_blocks(backbone: nn.Module) -> int:
    agg = backbone.aggregator
    blocks = [*agg.frame_blocks, *agg.inter_frame_blocks]
    blocks = [getattr(b, "block", b) for b in blocks]                  # unwrap grad-checkpoint wrappers
    patch_blocks = getattr(agg.patch_embed, "blocks", None)
    if patch_blocks is not None and not getattr(agg.patch_embed, "chunked_blocks", False):
        blocks += list(patch_blocks)
    # Every distinct (S,H,W) is a new static shape. Allow plenty; past the limit GraphRunner/run()
    # fall back to the eager blocks instead of failing.
    cfg = torch._dynamo.config
    for names, floor in ((("recompile_limit", "cache_size_limit"), 1024),          # renamed after 2.5
                         (("accumulated_recompile_limit", "accumulated_cache_size_limit"), 4096)):
        name = next(n for n in names if hasattr(cfg, n))
        setattr(cfg, name, max(getattr(cfg, name), floor))
    known = torch._inductor.config.get_config_copy()
    options = {k: v for k, v in COMPILE_OPTIONS.items() if k in known}            # 2.5 lacks some options
    for block in blocks:
        block.compile(backend="inductor", fullgraph=True, dynamic=False, options=options)
    return len(blocks)


def optimize_for_inference(model: nn.Module, *, exact: bool = False, merge_lora_adapters: bool = True,
                           bf16_weights: bool = True, compile: bool = True,
                           fast_mask_downsample: bool = True) -> dict:
    """Apply the streaming service's inference transforms in place. Irreversible; eval-only."""
    if model.training:
        raise ValueError("optimize_for_inference requires model.eval()")
    if exact:
        merge_lora_adapters = bf16_weights = compile = False
    model.requires_grad_(False)
    report = {"merged_lora": 0, "bf16_weight_modules": 0, "compiled_blocks": 0,
              "fast_mask_downsample": fast_mask_downsample}
    if fast_mask_downsample:
        model.query_decoder._attn_mask_from_logits = MethodType(_attn_mask_exact_downsample, model.query_decoder)
    if merge_lora_adapters:
        report["merged_lora"] = merge_lora(model.backbone)
    if bf16_weights:
        if not merge_lora_adapters and any(isinstance(m, LoRALinear) for m in model.backbone.modules()):
            raise ValueError("bf16_weights needs merged adapters")
        report["bf16_weight_modules"] = cache_bf16_weights(model.backbone)
    if compile:
        report["compiled_blocks"] = compile_backbone_blocks(model.backbone)
    return report


def disable_compile(model: nn.Module) -> int:
    disabled = 0
    for module in model.modules():
        if getattr(module, "_compiled_call_impl", None) is not None:
            module._compiled_call_impl = None
            disabled += 1
    return disabled


def run(model: nn.Module, images: torch.Tensor) -> dict:
    """model(images) under BF16 autocast; drops compilation if the recompile limit is exhausted."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        try:
            return model(images)
        # Older PyTorch releases lack this exception; preserve their original forward errors.
        except getattr(torch._dynamo.exc, "FailOnRecompileLimitHit", ()):
            disable_compile(model)
            return model(images)


class GraphRunner:
    """Bounded per-shape CUDA graph cache around model(images) under BF16 autocast."""

    def __init__(self, model: nn.Module, capacity: int = 4, warmup: int = 2):
        self.model, self.capacity, self.warmup = model, capacity, warmup
        self.graphs: OrderedDict = OrderedDict()

    def _forward(self, images):
        return run(self.model, images)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> dict:
        if images.dim() == 4:
            images = images.unsqueeze(0)
        key = (tuple(images.shape), images.dtype, images.device)
        entry = self.graphs.pop(key, None)
        if entry is None:
            while len(self.graphs) >= self.capacity:
                self.graphs.popitem(last=False)
            static = images.clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):                            # compile + allocator warm-up
                for _ in range(self.warmup):
                    self._forward(static)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = self._forward(static)
            entry = (graph, static, output)
        self.graphs[key] = entry
        graph, static, output = entry
        static.copy_(images)
        graph.replay()
        return output
