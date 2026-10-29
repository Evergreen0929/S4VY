"""SAM-style prompt encoder for the frozen S4VY segmenter.

Given a point / box prompt on 1+ frames, produce ONE object query that is fed into the SAME frozen
QueryInstanceDecoder alongside the 400 static queries (UNIDIRECTIONAL self-attn: the prompt query READS
the 400 + image, the 400 do NOT see it, so the 400 outputs stay bit-identical to the frozen base). The prompt query
resolves to the prompted object's 4D masklet (mask on all S frames).

Design:
  - the query CONTENT comes from the FEATURES THEMSELVES (not abstract Fourier tokens): we bilinear-sample
    the frozen mask_feat at the prompt location(s). This anchors the query in the space the frozen decoder
    reads, so it routes to the right object with a tiny trainable head.
      * point  -> bilinear sample mask_feat at (x,y) on that frame.
      * box    -> CENTER-TO-EDGE gaussian-weighted pooling over the box (down-weights corner background).
      * multi-point / multi-frame -> sample each, then mean-pool.
  - a learned type embedding {point-pos, point-neg, box} is added to each sampled token before pooling.
  - the query POSITION is a Fourier PE of the prompt's mean (x,y) -> MLP (so multiple prompts are distinct).
Only THIS module trains; the segmenter is fully frozen.
"""
from __future__ import annotations

from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .query_decoder import MLP

T_POINT_POS, T_POINT_NEG, T_BOX = 0, 1, 2


class PromptEncoder(nn.Module):
    def __init__(self, feat_dim: int = 256, dec_dim: int = 256, n_fourier: int = 64,
                 box_grid: int = 12, box_sigma_scale: float = 0.5,
                 init_gain_content: float = 1.0, init_gain_pos: float = 1.0,
                 learnable_pool: bool = False):
        super().__init__()
        self.feat_dim = feat_dim
        self.box_grid = box_grid
        self.box_sigma_scale = box_sigma_scale
        # v1lp: LEARNABLE pooling. Init is chosen so step-0 output == plain v1 exactly:
        #   point: 3x3 patch conv, dirac init (center tap) == old single bilinear sample
        #   box:   14x14 grid conv, identity-channel x normalized-gaussian spatial init == old gaussian pool
        #          (old weights are exp(-0.5(u^2+v^2)) on a linear u,v in [-1,1] grid -- box-size independent)
        #   frames: softmax(score) weighted mean, zero-init score == uniform == old mean
        self.learnable_pool = learnable_pool
        if learnable_pool:
            self.box_grid = 14
            self.point_conv = nn.Conv2d(feat_dim, feat_dim, 3, bias=False)
            self.box_conv = nn.Conv2d(feat_dim, feat_dim, self.box_grid, bias=False)
            self.tok_score = nn.Linear(feat_dim, 1, bias=False)
            with torch.no_grad():
                idx = torch.arange(feat_dim)
                self.point_conv.weight.zero_()
                self.point_conv.weight[idx, idx, 1, 1] = 1.0
                u = torch.linspace(-1.0, 1.0, self.box_grid)
                g = torch.exp(-0.5 * (u.view(-1, 1) ** 2 + u.view(1, -1) ** 2))
                g = g / g.sum()
                self.box_conv.weight.zero_()
                self.box_conv.weight[idx, idx] = g
                self.tok_score.weight.zero_()
        self.type_embed = nn.Embedding(3, feat_dim)          # pos / neg / box
        nn.init.normal_(self.type_embed.weight, std=0.02)
        self.content_mlp = MLP(feat_dim, dec_dim, dec_dim, num_layers=3)   # sampled feat -> query content
        self.pos_mlp = MLP(2 * n_fourier, dec_dim, dec_dim, num_layers=3)  # fourier(xy) -> query pos
        # fixed gaussian Fourier basis for positional encoding of (x,y) in [0,1]
        self.register_buffer("pe_basis", torch.randn(2, n_fourier) * 2.0 * 3.14159265)
        # INIT-SCALE fix (measured): default kaiming puts the query at out-norm ~1.7 -- a whisper the frozen
        # decoder ignores (empty masks -> tiny grads -> stuck). The converged solution sits 25x/17x higher
        # (content 43.6, pos 19.8). Scaling the FINAL Linear at init drops the encoder straight into the
        # decoder-audible basin so static-only training can bootstrap. Default 1.0 = old behavior.
        if init_gain_content != 1.0:
            with torch.no_grad():
                self.content_mlp.layers[-1].weight.mul_(init_gain_content)
                self.content_mlp.layers[-1].bias.mul_(init_gain_content)
        if init_gain_pos != 1.0:
            with torch.no_grad():
                self.pos_mlp.layers[-1].weight.mul_(init_gain_pos)
                self.pos_mlp.layers[-1].bias.mul_(init_gain_pos)

    # ---- feature sampling helpers (all differentiable, operate on ONE frame's mask_feat (C,h,w)) ----
    def _sample_point(self, feat: torch.Tensor, x: float, y: float) -> torch.Tensor:
        """feat:(C,h,w) -> (C,) bilinear sample at normalized (x,y) in [0,1]."""
        grid = feat.new_tensor([[2 * x - 1, 2 * y - 1]]).view(1, 1, 1, 2)      # (1,1,1,2) in [-1,1]
        s = F.grid_sample(feat.unsqueeze(0), grid, mode="bilinear", align_corners=False)  # (1,C,1,1)
        return s.view(-1)

    def _pool_box(self, feat: torch.Tensor, box) -> torch.Tensor:
        """feat:(C,h,w), box=(x0,y0,x1,y1) normalized -> (C,) center-weighted gaussian pool."""
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        sx = max((x1 - x0) * self.box_sigma_scale, 1e-3)
        sy = max((y1 - y0) * self.box_sigma_scale, 1e-3)
        K = self.box_grid
        gx = torch.linspace(x0, x1, K, device=feat.device)
        gy = torch.linspace(y0, y1, K, device=feat.device)
        yy, xx = torch.meshgrid(gy, gx, indexing="ij")                        # (K,K)
        w = torch.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))  # gaussian, center-heavy
        w = (w / w.sum().clamp(min=1e-6)).reshape(-1)                          # (K*K,)
        grid = torch.stack([2 * xx - 1, 2 * yy - 1], dim=-1).reshape(1, K * K, 1, 2)
        s = F.grid_sample(feat.unsqueeze(0), grid, mode="bilinear", align_corners=False)  # (1,C,K*K,1)
        s = s.view(feat.shape[0], K * K)                                       # (C,K*K)
        return (s * w).sum(-1)                                                 # (C,)

    def _sample_point_lp(self, feat: torch.Tensor, x: float, y: float) -> torch.Tensor:
        """feat:(C,h,w) -> (C,): 3x3 one-texel-spaced patch around (x,y), pooled by learnable conv."""
        C, h, w = feat.shape
        dx, dy = 1.0 / w, 1.0 / h
        xs = feat.new_tensor([x - dx, x, x + dx])
        ys = feat.new_tensor([y - dy, y, y + dy])
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([2 * xx - 1, 2 * yy - 1], dim=-1).view(1, 3, 3, 2)
        s = F.grid_sample(feat.unsqueeze(0), grid, mode="bilinear", align_corners=False)   # (1,C,3,3)
        return self.point_conv(s).view(-1)                                                 # (1,C,1,1)->(C,)

    def _pool_box_lp(self, feat: torch.Tensor, box) -> torch.Tensor:
        """feat:(C,h,w), box normalized -> (C,): 14x14 grid over the box, pooled by learnable conv."""
        x0, y0, x1, y1 = box
        K = self.box_grid
        gx = torch.linspace(x0, x1, K, device=feat.device)
        gy = torch.linspace(y0, y1, K, device=feat.device)
        yy, xx = torch.meshgrid(gy, gx, indexing="ij")
        grid = torch.stack([2 * xx - 1, 2 * yy - 1], dim=-1).view(1, K, K, 2)
        s = F.grid_sample(feat.unsqueeze(0), grid, mode="bilinear", align_corners=False)   # (1,C,K,K)
        return self.box_conv(s).view(-1)

    def _fourier(self, x: float, y: float, device) -> torch.Tensor:
        p = torch.tensor([x, y], device=device, dtype=torch.float32) @ self.pe_basis   # (n_fourier,)
        return torch.cat([torch.sin(p), torch.cos(p)], -1)                     # (2*n_fourier,)

    def encode_one(self, mask_feat_b: torch.Tensor, obj: Dict) -> (torch.Tensor, torch.Tensor):
        """mask_feat_b:(S,C,h,w) for ONE sample. obj: {'points':[(f,x,y,label)], 'boxes':[(f,x0,y0,x1,y1)]}.
        Returns (content_feat (C,), mean_xy tuple) -> caller builds query."""
        toks, xs, ys = [], [], []
        _pt = self._sample_point_lp if self.learnable_pool else self._sample_point
        _bx = self._pool_box_lp if self.learnable_pool else self._pool_box
        for (f, x, y, label) in obj.get("points", []):
            v = _pt(mask_feat_b[f], x, y) + self.type_embed.weight[T_POINT_POS if label else T_POINT_NEG]
            toks.append(v); xs.append(x); ys.append(y)
        for (f, x0, y0, x1, y1) in obj.get("boxes", []):
            v = _bx(mask_feat_b[f], (x0, y0, x1, y1)) + self.type_embed.weight[T_BOX]
            toks.append(v); xs.append((x0 + x1) / 2); ys.append((y0 + y1) / 2)
        t = torch.stack(toks, 0)                                              # (N,C)
        if self.learnable_pool:
            wgt = torch.softmax(self.tok_score(t).squeeze(-1), 0)             # (N,) soft frame/token weights
            content = (t * wgt.unsqueeze(-1)).sum(0)
        else:
            content = t.mean(0)                                               # (C,) pool all tokens
        mx, my = float(sum(xs) / len(xs)), float(sum(ys) / len(ys))
        return content, (mx, my)

    def forward(self, mask_feat: torch.Tensor, prompts: List[List[Dict]]):
        """mask_feat:(B,S,C,h,w). prompts: list[B] of list[P] object-prompt dicts.
        Returns pq (B,Pmax,dec), pq_pos (B,Pmax,dec), pq_valid (B,Pmax) bool."""
        B = mask_feat.shape[0]
        Pmax = max((len(p) for p in prompts), default=0)
        dec = self.content_mlp.layers[-1].out_features
        pq = mask_feat.new_zeros(B, Pmax, dec)
        pq_pos = mask_feat.new_zeros(B, Pmax, dec)
        pq_valid = torch.zeros(B, Pmax, dtype=torch.bool, device=mask_feat.device)
        for b in range(B):
            for i, obj in enumerate(prompts[b]):
                content, (mx, my) = self.encode_one(mask_feat[b].float(), obj)
                pq[b, i] = self.content_mlp(content.to(pq.dtype))
                pq_pos[b, i] = self.pos_mlp(self._fourier(mx, my, mask_feat.device).to(pq.dtype))
                pq_valid[b, i] = True
        return pq, pq_pos, pq_valid
