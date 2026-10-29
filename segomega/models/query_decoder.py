"""Class-agnostic, multi-frame query decoder (Mask2Former-style).

N learnable instance queries are *shared across the S frames* and refined by L decoder layers of
  (1) masked cross-attention  -> queries attend to ALL frames' patch tokens jointly
                                  (this is what makes one query == one 3D-consistent instance),
  (2) query self-attention,
  (3) FFN.
Each layer predicts, per query:
  - mask logits per frame  :  <mask_embed(query), mask_feature>  -> (B, N, S, h, w)
  - objectness logit       :  {object, no-object}  (class-agnostic; the "no-object/varnothing" slot
                              is exactly SegVGGT's background logit used as 1-objectness at inference).
Mask-attention (Mask2Former / SegVGGT `mask_attention`) restricts each query's cross-attention to the
foreground of its own current mask prediction.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers=3):
        super().__init__()
        h = [dim_hidden] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip([dim_in] + h, h + [dim_out]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class _DecoderLayer(nn.Module):
    """One Mask2Former decoder layer: masked-cross-attn -> self-attn -> FFN (all pre-LN residual)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ca = nn.LayerNorm(dim)
        self.norm_sa = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(
        self,
        q: torch.Tensor,          # (B, N, C) query features
        q_pos: torch.Tensor,      # (B, N, C) query positional embeddings
        mem: torch.Tensor,        # (B, M, C) memory (all frames' tokens)
        mem_pos: torch.Tensor,    # (B, M, C) memory positional embeddings
        attn_mask: Optional[torch.Tensor],  # (B*num_heads, N, M) bool, True = block
        sa_mask: Optional[torch.Tensor] = None,  # (N, N) bool self-attn mask, True=block (prompt: unidirectional)
    ) -> torch.Tensor:
        # (1) masked cross-attention
        h = self.norm_ca(q)
        out, _ = self.cross_attn(
            query=h + q_pos, key=mem + mem_pos, value=mem, attn_mask=attn_mask, need_weights=False
        )
        q = q + out
        # (2) self-attention among queries (sa_mask=None -> teacher/23c UNCHANGED full self-attn)
        h = self.norm_sa(q)
        out, _ = self.self_attn(query=h + q_pos, key=h + q_pos, value=h, attn_mask=sa_mask, need_weights=False)
        q = q + out
        # (3) FFN
        q = q + self.ffn(self.norm_ffn(q))
        return q


class QueryInstanceDecoder(nn.Module):
    def __init__(
        self,
        token_dim: int = 2048,     # backbone token dim (2*embed_dim)
        mask_dim: int = 256,       # must match MaskFeatureHead.mask_dim
        dec_dim: int = 256,
        num_queries: int = 400,    # SegVGGT default
        num_layers: int = 9,       # Mask2Former default
        num_heads: int = 8,
        memory_layer_idx: int = 23,  # which cached backbone layer feeds the decoder memory
        max_frames_pos: int = 64,
        mask_attn_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.memory_layer_idx = memory_layer_idx
        self.mask_attn_threshold = mask_attn_threshold

        self.query_feat = nn.Embedding(num_queries, dec_dim)
        self.query_pos = nn.Embedding(num_queries, dec_dim)
        nn.init.normal_(self.query_feat.weight, std=1e-6)  # SegVGGT inits queries tiny

        self.mem_proj = nn.Linear(token_dim, dec_dim)
        self.mem_norm = nn.LayerNorm(dec_dim)
        self.frame_pos = nn.Embedding(max_frames_pos, dec_dim)  # per-frame embedding for memory

        self.layers = nn.ModuleList(_DecoderLayer(dec_dim, num_heads) for _ in range(num_layers))
        self.decoder_norm = nn.LayerNorm(dec_dim)

        # prediction heads (shared across layers, applied after each)
        self.objectness = nn.Linear(dec_dim, 2)            # {object, no-object}
        self.mask_embed = MLP(dec_dim, dec_dim, mask_dim, num_layers=3)

    # ---- helpers -----------------------------------------------------------------------------
    def _patch_grid(self, patch_tokens: torch.Tensor, B: int, S: int) -> Tuple[int, int]:
        # patch_tokens: (B, S, Np, C). assume square-ish; caller passes ph, pw separately normally.
        raise NotImplementedError

    def _predict(self, q: torch.Tensor, mask_feat: torch.Tensor):
        """q:(B,N,C) -> objectness (B,N,2), mask_logits (B,N,S,h,w)."""
        qn = self.decoder_norm(q)
        obj = self.objectness(qn)
        me = self.mask_embed(qn)                                    # (B,N,mask_dim)
        mask_logits = torch.einsum("bnd,bsdhw->bnshw", me, mask_feat)
        return obj, mask_logits

    def _attn_mask_from_logits(self, mask_logits: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        """mask_logits:(B,N,S,h,w) -> cross-attn mask (B*heads, N, S*gh*gw) bool, True=block.

        Downsample predicted masks to the *memory* (patch) grid (gh,gw), threshold the sigmoid.
        A query attends only where its own predicted mask is foreground.  Queries whose mask is
        empty everywhere are allowed to attend everywhere (avoid dead queries) -- as in SegVGGT.
        """
        B, N, S, h, w = mask_logits.shape
        m = mask_logits.reshape(B * N * S, 1, h, w)
        m = F.interpolate(m, size=(gh, gw), mode="bilinear", align_corners=False)
        m = m.reshape(B, N, S * gh * gw)
        fg = (m.sigmoid() > self.mask_attn_threshold)
        block = ~fg                                                 # True where we block attention
        empty = block.all(dim=-1, keepdim=True)                     # query fully blocked -> unblock all
        block = block & ~empty
        block = block.unsqueeze(1).expand(-1, self.num_heads, -1, -1).reshape(B * self.num_heads, N, S * gh * gw)
        return block.detach()

    # ---- forward -----------------------------------------------------------------------------
    def _build_sa_mask(self, B, N, P, prompt_valid, device):
        """UNIDIRECTIONAL self-attn mask over [N static ; P prompt] tokens, True=block.
        static rows are BLOCKED from prompt columns (400 never see the prompt -> outputs bit-identical to
        the frozen base); prompt rows attend to everything. invalid (padded) prompt columns are blocked for all."""
        NP = N + P
        sa = torch.zeros(B, NP, NP, dtype=torch.bool, device=device)
        sa[:, :N, N:] = True                                   # static -> prompt : blocked
        if prompt_valid is not None:
            for b in range(B):
                inv = (~prompt_valid[b]).nonzero(as_tuple=True)[0]
                if inv.numel():
                    sa[b, :, N + inv] = True                    # nobody attends padded prompt cols
        return sa.unsqueeze(1).expand(-1, self.num_heads, -1, -1).reshape(B * self.num_heads, NP, NP)

    def forward(
        self,
        tokens_list: List[torch.Tensor],
        patch_token_start: int,
        mask_feat: torch.Tensor,           # (B, S, mask_dim, h, w) from MaskFeatureHead
        patch_hw: Tuple[int, int],         # (ph, pw) patch grid of the backbone
        prompt_q: Optional[torch.Tensor] = None,       # (B, P, dec) prompt-query content
        prompt_q_pos: Optional[torch.Tensor] = None,   # (B, P, dec) prompt-query position
        prompt_valid: Optional[torch.Tensor] = None,   # (B, P) bool
    ) -> dict:
        ph, pw = patch_hw
        mem_tokens = tokens_list[self.memory_layer_idx]            # (B,S,T,2048)
        B, S, _, _ = mem_tokens.shape
        patches = mem_tokens[:, :, patch_token_start:].float()     # (B,S,Np,2048)
        Np = patches.shape[2]
        assert Np == ph * pw, f"patch count {Np} != {ph}*{pw}"

        mem = self.mem_proj(patches)                               # (B,S,Np,dec)
        # per-frame positional embedding added to memory (encodes frame-SEPARATION; MATTERS: S=16 WITH vs
        # WITHOUT -> mask IoU 0.10). SEG_FRAMEPOS_MODE ablation:
        #   embed (default) = learned nn.Embedding(64), cycle-padded for S>64 (segmenter original; frames capped/16-trained)
        #   none            = dropped (also via legacy SEG_NO_FRAMEPOS=1)
        #   sin             = fixed sinusoidal PE of frame index: UNBOUNDED #frames, no cap, trains fresh
        C = mem.shape[-1]
        mode = os.environ.get("SEG_FRAMEPOS_MODE", "embed")
        if os.environ.get("SEG_NO_FRAMEPOS") or mode == "none":
            fpos = mem.new_zeros(1, 1, 1, C)
        elif mode == "sin":
            pos = torch.arange(S, device=mem.device, dtype=torch.float32).unsqueeze(1)          # (S,1)
            div = torch.exp(torch.arange(0, C, 2, device=mem.device, dtype=torch.float32) * (-math.log(10000.0) / C))
            pe = torch.zeros(S, C, device=mem.device, dtype=torch.float32)
            pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
            fpos = pe.to(mem.dtype).view(1, S, 1, C)                                             # any S, no cap
        else:
            n = self.frame_pos.num_embeddings
            idx = torch.arange(S, device=self.frame_pos.weight.device) % n   # cycle-pad for S>n (S<=n: == [:S])
            fpos = self.frame_pos.weight[idx].view(1, S, 1, -1)
        mem = self.mem_norm(mem) + fpos
        mem_pos = fpos.expand(B, S, Np, -1).reshape(B, S * Np, -1)
        mem = mem.reshape(B, S * Np, -1)

        q = self.query_feat.weight.unsqueeze(0).expand(B, -1, -1)  # (B,N,dec)
        q_pos = self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)
        N = self.num_queries

        # append prompt queries -> [400 static ; P prompt], UNIDIRECTIONAL self-attn (sa_mask)
        P = 0
        sa_mask = None
        if prompt_q is not None and prompt_q.shape[1] > 0:
            P = prompt_q.shape[1]
            q = torch.cat([q, prompt_q], dim=1)                    # (B, N+P, dec)
            q_pos = torch.cat([q_pos, prompt_q_pos], dim=1)
            sa_mask = self._build_sa_mask(B, N, P, prompt_valid, q.device)

        aux = []
        # initial prediction (layer 0 input) for the first mask-attention
        obj, mask_logits = self._predict(q, mask_feat)
        aux.append({"objectness": obj, "mask_logits": mask_logits})

        for layer in self.layers:
            attn_mask = self._attn_mask_from_logits(mask_logits, ph, pw)
            q = layer(q, q_pos, mem, mem_pos, attn_mask, sa_mask=sa_mask)
            obj, mask_logits = self._predict(q, mask_feat)
            aux.append({"objectness": obj, "mask_logits": mask_logits})

        # split static (first N -> bit-identical to the frozen base) vs prompt (last P) outputs
        def _split(d):
            return ({"objectness": d["objectness"][:, :N], "mask_logits": d["mask_logits"][:, :N]},
                    {"objectness": d["objectness"][:, N:], "mask_logits": d["mask_logits"][:, N:]})
        stat_aux = [_split(a)[0] for a in aux]
        out = stat_aux[-1].copy()
        out["aux_outputs"] = stat_aux[:-1]
        out["mask_embed_final"] = self.mask_embed(self.decoder_norm(q[:, :N]))
        if P > 0:
            prompt_aux = [_split(a)[1] for a in aux]
            out["prompt_objectness"] = prompt_aux[-1]["objectness"]     # (B,P,2)
            out["prompt_mask_logits"] = prompt_aux[-1]["mask_logits"]   # (B,P,S,h,w)
            out["prompt_aux_outputs"] = prompt_aux[:-1]                 # deep supervision on prompt queries
            out["prompt_mask_embed_final"] = self.mask_embed(self.decoder_norm(q[:, N:]))
        return out
