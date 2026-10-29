#!/usr/bin/env python
"""vLLM plugin: s4vy registers as a custom modality on top of stock Qwen3-VL.

WHY A PLUGIN AND NOT PRECOMPUTED EMBEDDINGS. Qwen3-VL is deepstack -- image features are re-injected at
several intermediate layers (`deepstack_visual_embeds`, one tensor per layer), so an input-embedding sequence
cannot carry the image contribution and the EmbedsPrompt route is structurally impossible here. Letting vLLM
own the vision path is the only correct option, and it is also the cheap one: images, deepstack, M-RoPE and
paged KV all stay native. <REG> is an ordinary text token as far as position encoding is concerned, so nothing
about RoPE changes.

WHAT WE ADD. Exactly the eight lines of agent_model.gen_boxes that turn s4vy features into <REG> embeddings:
    normalize -> reg_proj -> (+reg_slot_emb) -> reg_ln -> rescale to the mean text-token norm
Copied rather than re-derived, because a reward/eval that silently uses a slightly different injection is the
failure mode this whole exercise is meant to avoid.

NOT COVERED: the <GROUND> readout. vLLM's generate returns logprobs, not arbitrary hidden states, so the qemb
match stays in torch -- it is one forward per window, not autoregressive, so it is the small half anyway.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Mapping, Sequence

from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration, Qwen3VLProcessingInfo,
    Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import MultiModalDataParser, EmbeddingItems, DictEmbeddingItems
from vllm.multimodal.processing import PromptReplacement
from vllm import ModelRegistry

INJ = os.environ.get("INJ_CKPT", "/PATH/TO/checkpoints/agent/g2_inject.pt")
MF_NF_MAX = int(os.environ.get("MF_NF_MAX", "8"))    # frames per window; <REG> count = MF_NF_MAX * 17
RPF = 17                                             # <REG> per frame -- the prompt writes them in 8 runs of
NREG = MF_NF_MAX * RPF                               # 17, separated by <GROUND> and " Frame j: " text
HIDDEN = int(os.environ.get("VLM_HIDDEN", "2048"))   # register embeds arrive ALREADY projected to hidden


class RegProcessingInfo(Qwen3VLProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        lim = dict(super().get_supported_mm_limits())
        # ONE item PER FRAME, each spanning that frame's 17 <REG> tokens -- the same shape as an image item,
        # which is one item spanning many <|image_pad|> tokens. Item granularity must match the prompt's actual
        # contiguous runs: NREG single-token items make each register a separate non-empty REPLACE, and
        # _find_matches applies at most one of those per pass over a SHARED prev_end_idx (144 interleaved passes
        # corrupted the first image: [1720,192x7] with registers vs [192x8] without), while a single NREG-token
        # item never matches at all because the runs are broken up by <GROUND>.
        lim["register"] = MF_NF_MAX
        return lim


class RegDummyInputsBuilder(Qwen3VLDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return super().get_dummy_text({k: v for k, v in mm_counts.items() if k != "register"})

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int], mm_options=None):
        d = dict(super().get_dummy_mm_data(seq_len, {k: v for k, v in mm_counts.items() if k != "register"},
                                           mm_options))
        if mm_counts.get("register"):
            d["register"] = torch.zeros(mm_counts["register"], RPF, HIDDEN)
        return d


class RegDataParser(MultiModalDataParser):
    """vLLM only knows how to parse audio/image/video; a custom modality needs its own subparser (this is the
    route the base class's own docstring points at). Registers are already embeddings -- no HF processor is
    involved -- so EmbeddingItems is the right item type."""
    def _parse_register(self, data):
        return EmbeddingItems(data, "register")

    def _get_subparsers(self):
        return {**super()._get_subparsers(), "register": self._parse_register}


class RegMultiModalProcessor(Qwen3VLMultiModalProcessor):
    def _get_data_parser(self) -> MultiModalDataParser:
        return RegDataParser()

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        """The field is `register_embeds`, NOT `register`. EmbeddingItems ships its data through the
        PASSTHROUGH channel (`get_passthrough_data` -> {modality}_embeds), which bypasses the HF processor
        entirely -- so overriding _call_hf_processor achieves nothing, and a field named `register` has no
        data behind it (vLLM then reports "0 of 136 items found")."""
        cfg = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
        cfg["register_embeds"] = MultiModalFieldConfig.batched("register")
        return cfg

    def _hf_processor_applies_updates(self, prompt_text, mm_items, hf_processor_mm_kwargs,
                                      tokenization_kwargs):
        """THE FIX. The base implementation is all-or-nothing across modalities:

            return not any(isinstance(items, (EmbeddingItems, DictEmbeddingItems))
                           for items in mm_items.values())

        Registers are embeddings, so this returned False for the WHOLE request and vLLM re-applied every
        modality's updates -- including the images, which the HF processor had already expanded. Measured:
        1536 image tokens in, 3064 out, first image ballooning to 1720 while the other seven stayed at 192.
        Registers need no expansion at all (their <REG> placeholders are written into the prompt by the eval),
        so they must not veto the flag; the images alone decide it.
        """
        return not any(isinstance(items, (EmbeddingItems, DictEmbeddingItems))
                       for m, items in mm_items.items() if m != "register")


    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        """vLLM builds the item -> token-position map from these updates. Our <REG> placeholders are ALREADY in
        the prompt (the eval writes them), and there is exactly one register item per <REG>, so each item maps
        to itself: a one-token replacement that leaves the prompt unchanged."""
        ups = list(super()._get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs))
        tok = self.info.get_tokenizer()
        reg_id = tok.convert_tokens_to_ids("<REG>")
        ups.append(PromptReplacement(modality="register", target=[reg_id] * RPF,
                                     replacement=[reg_id] * RPF))
        return ups


@MULTIMODAL_REGISTRY.register_processor(RegMultiModalProcessor,
                                        info=RegProcessingInfo,
                                        dummy_inputs=RegDummyInputsBuilder)
class Qwen3VLWithRegisters(Qwen3VLForConditionalGeneration):
    """Stock Qwen3-VL + the s4vy register injection at <REG> positions."""

    def iter_mm_grid_hw(self, input_tokens, mm_features):
        """M-RoPE walks EVERY multimodal feature to read its (h,w) grid, and `register` has no grid, so stock
        Qwen3-VL raises `Unsupported modality: register` here. Filtering it out is not a workaround -- <REG> is
        an ordinary text token for position purposes and must NOT contribute a vision grid; images keep their
        native M-RoPE treatment untouched."""
        return super().iter_mm_grid_hw(
            input_tokens, [f for f in mm_features if getattr(f, "modality", None) != "register"])


    def embed_multimodal(self, **kwargs):
        """vLLM treats `register` as a modality WITH an encoder: it calls this to get the embeddings and then
        merges them at the <REG> placeholders itself. So the projection is done CLIENT-SIDE and handed over
        finished -- which also makes it bit-identical to the HF path, because it is literally the same code
        (agent_model's reg_proj -> reg_ln -> rescale to the sequence's mean token norm) rather than a
        re-implementation living inside vLLM. tok_norm depends on the sequence's own token embeddings, which
        the mm-encoder stage cannot see, so computing it here would have been an approximation."""
        reg = kwargs.pop("register_embeds", None)
        out = super().embed_multimodal(**kwargs) if kwargs else None
        if reg is None:
            return out
        r = reg if isinstance(reg, torch.Tensor) else torch.as_tensor(reg)
        r = r.reshape(-1, r.shape[-1]).to(dtype=next(self.parameters()).dtype,
                                          device=next(self.parameters()).device)
        # Qwen3-VL is deepstack: an image item is [main | per-layer features] concatenated, width
        # visual_dim + multiscale_dim, and _compute_deepstack_embeds cats ALL mm items before splitting on that
        # width. Registers carry no intermediate-layer contribution, so they are padded with ZEROS there --
        # any other value would inject something into layers 5/11/17 that the SFT model never saw.
        need = int(self.visual_dim) + int(self.multiscale_dim)
        if r.shape[-1] < need:
            r = torch.cat([r, r.new_zeros(r.shape[0], need - r.shape[-1])], dim=-1)
        regs = tuple(r[i * RPF:(i + 1) * RPF] for i in range(r.shape[0] // RPF))   # one (RPF, hidden) per frame
        if out is None:
            return regs
        return tuple(out) + regs


def register():
    ModelRegistry.register_model("Qwen3VLWithRegisters", Qwen3VLWithRegisters)
