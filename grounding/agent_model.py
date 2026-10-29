#!/usr/bin/env python
"""Agentic-loop grounder on frozen segmenter.
Tokens: <REG> (placeholders where segmenter camera+register tokens are injected), <TEMP> (temporal
readout), <GROUND> (spatial readout).
Heads:
  reg_proj      2048 -> H   : inject segmenter register/camera tokens as VLM input embeddings
  temporal_head [maxpool(frame's <REG> hiddens) ; <TEMP> hidden] -> logit  (per-frame, option (c), BCE)
  text_proj     H -> P      : <GROUND> -> instance-match space   (G1/G3 matching)
  inst_proj     256 -> P    : segmenter query -> match space          (G1/G3 matching)
  gq_proj       H -> 256     : <GROUND> -> segmenter query/mask-embed space (G2/G4 dense decode)
  mask_embed    256 -> 256   : segmenter mask head (init from segmenter; frozen G2 / trainable G4)
LOSS env: matching (G1/G3) | dense (G2/G4).  segmenter backbone always frozen/cached."""
import os, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText
REG, TEMP, GROUND = "<REG>", "<TEMP>", "<GROUND>"
INST = "<INST>"   # INST_APPEND: inject the teacher dynamic instance queries (qemb) as input tokens so the VLM contextualizes them; <GROUND> matches their output hiddens
POINT = "<POINT>"   # dual-output (POINT_HEAD=1): independent token that regresses a 2D coord, decoded SEPARATELY from <GROUND> (which matches qemb) -> no task conflict; their (dis)agreement = self-correction signal
THINK_S, THINK_E = "<think>", "</think>"   # E5 (COT): natural-language geometry+appearance reasoning before <GROUND>


class MLP(nn.Module):
    def __init__(self, di, dh, do):
        super().__init__(); self.net = nn.Sequential(nn.Linear(di, dh), nn.GELU(), nn.Linear(dh, do))
    def forward(self, x): return self.net(x)


class PointHead(nn.Module):
    """Arm B: <POINT> hidden cross-attends 's per-frame PATCH-feature grid (visual pool, reuse seg features
    -- not image downsample) -> (x, y, null_logit). q:(M,H) point-token hidden; patch:(M,P,C=2048) pooled grid.
    VGGT patch features have huge magnitude -> F.normalize like reg_proj. 3rd out = P(absent) logit (null gate)."""
    def __init__(self, H, C=2048, P=192, heads=8):
        super().__init__()
        self.patch_proj = nn.Linear(C, H)
        self.pos = nn.Parameter(torch.randn(P, H) * 0.02)              # learned grid position embedding
        self.attn = nn.MultiheadAttention(H, heads, batch_first=True)
        self.norm = nn.LayerNorm(H)
        self.out = MLP(H, H // 2, 3)                                    # x, y, null_logit
    def forward(self, q, patch):                                       # q:(M,H)  patch:(M,P,C)
        kv = self.patch_proj(F.normalize(patch.float(), dim=-1).to(self.patch_proj.weight.dtype)) + self.pos  # (M,P,H)
        a, _ = self.attn(q[:, None], kv, kv)                           # (M,1,H)
        h = self.norm(q[:, None] + a).squeeze(1)                       # (M,H) residual
        return self.out(h)                                            # (M,3): [:2]=xy(pre-sigmoid), [2]=null logit


class MaskEmbedMLP(nn.Module):
    """Exactly matches segmenter query_decoder.mask_embed (3-layer ReLU MLP, `.layers` ModuleList)
    so its state_dict loads directly. Used for G2/G4 dense decode."""
    def __init__(self, di, dh, do, num_layers=3):
        super().__init__()
        h = [dh] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip([di] + h, h + [do]))
    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = F.relu(l(x)) if i < len(self.layers) - 1 else l(x)
        return x


class AgentGrounder(nn.Module):
    def __init__(self, model_path, inst_dim=256, proj_dim=256, dec_dim=256, lora=True,
                 lora_rank=32, lora_alpha=64, temp=0.07, loss="matching",
                 mask_embed_trainable=False, cot=None):
        super().__init__()
        self.loss = loss
        self.cot = (os.environ.get("COT") == "1") if cot is None else bool(cot)   # E5: add <think>/</think> for CoT
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tok = self.processor.tokenizer
        self.inst_append = (os.environ.get("INST_APPEND") == "1")   # ExpA: append instance-query tokens; <GROUND> matches their contextualized hiddens
        self.no_inst_proj = (os.environ.get("NO_INST_PROJ") == "1")  # ablation: drop the query-side proj -> match text_proj(ground) vs RAW teacher qemb (force alignment to teacher space)
        self.mf_bidir = (os.environ.get("MF_BIDIR") == "1")          # MFRAME: full BIDIRECTIONAL attention over the window (non-causal); each <GROUND> sees ALL 8 frames' registers (forward is non-generative -> causality unneeded)
        self.point_out = (os.environ.get("POINT_HEAD") == "1")       # dual-output: per-frame <POINT> token + coord regression head, decoded independently from <GROUND>
        specials = [REG, TEMP, GROUND] + ([THINK_S, THINK_E] if self.cot else []) + ([INST] if self.inst_append else []) + ([POINT] if self.point_out else [])
        add = [t for t in specials if t not in tok.get_vocab()]
        if add: tok.add_special_tokens({"additional_special_tokens": add})
        self.reg_id, self.temp_id, self.ground_id = (tok.convert_tokens_to_ids(t) for t in (REG, TEMP, GROUND))
        self.inst_id = tok.convert_tokens_to_ids(INST) if self.inst_append else -1
        self.point_id = tok.convert_tokens_to_ids(POINT) if self.point_out else -1
        self.think_s_id, self.think_e_id = (tok.convert_tokens_to_ids(t) for t in (THINK_S, THINK_E))   # valid only when cot
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa")
        self.vlm.resize_token_embeddings(len(tok))
        tc = self.vlm.config.get_text_config() if hasattr(self.vlm.config, "get_text_config") else self.vlm.config
        H = getattr(tc, "hidden_size", 3584)
        self.H = H
        self.reg_proj = nn.Linear(2048, H)
        self.reg_ln = nn.LayerNorm(H)        # align injected-register distribution to VLM token stats (Linear alone doesn't normalize)
        self.temporal_head = MLP(2 * H, H, 1)
        self.count_head = MLP(H, H // 2, 1)  # E4 (SEL_HEAD): <TEMP> -> predicted #frames-to-select (learned cutoff vs hard 0.5/top16/cap)
        self.action_head = MLP(H, H // 2, 1)  # E2 (ACTION): <TEMP> -> P(selection insufficient -> refine instead of ground)
        self.reg17 = os.environ.get("REG17") == "1"   # E6: use all 17 segmenter reg tokens/frame (no pooling), per-slot identity
        self.reg_slot_emb = nn.Embedding(17, H)       # E6: learned identity for each of the 17 tokens within a frame
        self.text_proj = MLP(H, H // 2, proj_dim)
        self.inst_proj = MLP(inst_dim, proj_dim, proj_dim)
        self.null_head = MLP(H, H // 2, 1)            # SFRAME: <GROUND> -> null logit (described object ABSENT in this single frame)
        self.inst_in_proj = nn.Linear(inst_dim, H)    # ExpA: inject instance query (qemb 256) -> VLM token space (distribution-aligned like reg_proj/reg_ln)
        self.inst_in_ln = nn.LayerNorm(H)
        self.inst_out_proj = MLP(H, H // 2, proj_dim)  # ExpA: <INST> output hidden -> match space (vs text_proj(ground_h))
        if self.point_out:
            _pg = int(os.environ.get("PATCH_GH", "12")) * int(os.environ.get("PATCH_GW", "16"))   # patch grid size (P), must match cache_patch.py
            self.point_head = PointHead(H, C=2048, P=_pg)   # Arm B: <POINT> hidden cross-attends patch grid -> (x,y,null); reuse seg features not image downsample
        self.gq_proj = MLP(H, H // 2, dec_dim)
        self.mask_embed = MaskEmbedMLP(dec_dim, dec_dim, dec_dim, num_layers=3)   # matches segmenter; init/load from segmenter; freeze unless trainable
        for p in self.mask_embed.parameters(): p.requires_grad = bool(mask_embed_trainable)
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / temp).log())
        if lora:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            self.vlm = get_peft_model(self.vlm, cfg)
        # gradient checkpointing: all-frame register (R1, up to ~2k tokens) blows 80GB without it.
        # NO_GRAD_CKPT=1 disables it (faster: backward skips forward-recompute) when memory allows; default keeps it ON (running experiments unaffected).
        if os.environ.get("NO_GRAD_CKPT") != "1":
            self.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(self.vlm, "enable_input_require_grads"): self.vlm.enable_input_require_grads()
        else:
            print("[agent] NO_GRAD_CKPT=1 -> gradient checkpointing DISABLED (faster backward, more memory)", flush=True)
        # LOWMEM (G5 history multi-turn: ~24 images/forward): run the FROZEN vision tower under no_grad so
        # its activations aren't stored for backward. Mathematically identical for trainable (LoRA) params
        # since the vision tower has no trainable params. Opt-in via env -> G1-4 untouched.
        self._lowmem = os.environ.get("LOWMEM") == "1"
        if self._lowmem:
            base = self.vlm.get_base_model() if hasattr(self.vlm, "get_base_model") else self.vlm
            vis = next((m for n, m in base.named_modules() if n.split(".")[-1] == "visual"), None)
            if vis is not None and not any(p.requires_grad for p in vis.parameters()):
                _vf = vis.forward
                def _no_grad_vis(*a, __f=_vf, **k):
                    with torch.no_grad():
                        return __f(*a, **k)
                vis.forward = _no_grad_vis
                print("[agent] LOWMEM: frozen vision tower -> no_grad (drops vision activations)", flush=True)
            # CRITICAL: output_hidden_states=True retains ALL layer outputs -> defeats gradient
            # checkpointing in backward (verified: bwd peak 57.7GB @3140 tok). Instead grab the final
            # hidden via a pre-hook on lm_head and call with output_hidden_states=False -> grad-ckpt
            # actually frees/recomputes intermediates -> backward drops to ~forward level.
            self._lm_head = base.get_output_embeddings()
            print("[agent] LOWMEM: last-hidden via lm_head hook, output_hidden_states=False (restores grad-ckpt)", flush=True)

    def encode(self, input_ids, attention_mask, register, reg_frame, multi_readout=False, multi_ground=False, return_logits=False, inst_emb=None, inst_valid=None, **mm):
        """register (B,R,2048) segmenter reg/camera tokens to scatter into <REG> positions;
        reg_frame (B,R) frame index of each reg token (for per-frame pooling).
        Returns: temp_h (B,H), ground_h (B,H), reg_h (B,R,H).
        multi_readout=True (G5 history multi-turn): also returns the LAST <GROUND> hidden
        (turn-2 / R2 readout) so the R1 ground (first) and R2 ground (last) are read in ONE
        forward -> tuple becomes (temp_h, ground_first, ground_last, reg_h, reg_frame)."""
        base = self.vlm.get_base_model() if hasattr(self.vlm, "get_base_model") else self.vlm
        emb = base.get_input_embeddings()(input_ids)                       # (B,T,H)
        regm = (input_ids == self.reg_id)                                  # (B,T)
        # register is 1 token/frame (camera+register pooled upstream). segmenter tokens have HUGE magnitude
        # (massive activations) and a Linear alone won't normalize -> inject would wash out text (verified cosine 1.0).
        # (1) unit-normalize segmenter direction; (2) project; (3) LayerNorm to VLM token distribution; (4) match text-emb magnitude.
        reg_in = F.normalize(register.float(), dim=-1).to(self.reg_proj.weight.dtype)  # (B,R,2048)
        rp = self.reg_proj(reg_in)                                         # (B,R,H), in head dtype
        if self.reg17:                                                     # E6: 17 segmenter tokens/frame (unpooled) -> per-slot learned identity
            slot = (torch.arange(register.shape[1], device=rp.device) % 17)   # (R,) position within frame (layout = frame-major 17s)
            rp = rp + self.reg_slot_emb(slot).to(rp.dtype)[None]           # broadcast (1,R,H)
        rp = self.reg_ln(rp)                                               # align distribution (same dtype as head)
        tok_norm = emb.detach().norm(dim=-1).mean().clamp(min=1e-6).to(rp.dtype)
        rp = rp / rp.norm(dim=-1, keepdim=True).clamp(min=1e-6) * tok_norm  # match VLM token magnitude
        rvalid = reg_frame.ge(0)                                            # (B,R) True for real frames (pad = -1)
        emb = emb.clone(); emb[regm] = rp[rvalid].to(emb.dtype)             # scatter register into <REG> slots (BS>1: skip padded)
        if self.inst_append and inst_emb is not None:                       # ExpA: inject instance queries (qemb) into <INST> slots, SAME distribution-alignment as register
            instm = (input_ids == self.inst_id)
            ii = F.normalize(inst_emb.float(), dim=-1).to(self.inst_in_proj.weight.dtype)
            ip = self.inst_in_ln(self.inst_in_proj(ii))
            ip = ip / ip.norm(dim=-1, keepdim=True).clamp(min=1e-6) * tok_norm
            emb[instm] = ip[inst_valid].to(emb.dtype)
        am = attention_mask
        if getattr(self, "mf_bidir", False):                               # MFRAME bidirectional: 4D additive mask that only blocks pad keys (no causal) -> every token attends to all real tokens
            B_, T_ = attention_mask.shape
            key_ok = attention_mask[:, None, None, :].to(emb.dtype)         # (B,1,1,T)
            am = ((1.0 - key_ok) * torch.finfo(emb.dtype).min).expand(B_, 1, T_, T_)  # (B,1,T,T): 0 on real keys, -inf on pad; broadcast over query dim -> non-causal
        if getattr(self, "_lowmem", False):                                # capture last hidden via hook; no output_hidden_states (keeps grad-ckpt effective)
            _cap = {}
            _hk = self._lm_head.register_forward_pre_hook(lambda m, a: _cap.__setitem__("h", a[0]))
            try:
                self.vlm(inputs_embeds=emb, attention_mask=am, use_cache=False, **mm)
            finally:
                _hk.remove()
            h = _cap["h"]                                                  # (B,T,H) final hidden (lm_head input)
            lm_logits = self._lm_head(h) if return_logits else None        # E5 CoT: LM logits for <think> loss
        else:
            out = self.vlm(inputs_embeds=emb, attention_mask=am,
                           output_hidden_states=True, use_cache=False, **mm)
            h = out.hidden_states[-1]                                      # (B,T,H)
            lm_logits = out.logits if return_logits else None              # E5 CoT: LM logits for <think> loss
        B = h.size(0); ar = torch.arange(B, device=h.device)
        gmask = (input_ids == self.ground_id)
        temp_h = h[ar, (input_ids == self.temp_id).float().argmax(1)]
        if multi_ground:                                                  # MFRAME: ALL <GROUND> hiddens (B,G,H), one per frame
            _gc = gmask.sum(1)                                            # (B,) per-row <GROUND> count
            if int(_gc.min()) == int(_gc.max()):
                ground_h = h[gmask].view(B, -1, self.H)                   # uniform (train + every pre-g3 caller): unchanged
            else:
                # RAGGED: P2-g3 lets a <done/> window emit no Turn-2 readout, so rows in one batch carry 8 vs 16
                # <GROUND>. view() silently assumes uniform and blows up here. RIGHT-align the pad so callers
                # reading the LAST nf grounds (`gall[w, -nf:]`) stay correct for both row kinds.
                _G = int(_gc.max()); ground_h = h.new_zeros(B, _G, self.H)
                _flat = h[gmask]; _o = 0
                for _b in range(B):
                    _c = int(_gc[_b]); ground_h[_b, _G - _c:] = _flat[_o:_o + _c]; _o += _c
        else:
            ground_h = h[ar, gmask.float().argmax(1)]                     # FIRST <GROUND> (R1 turn)
        self._point_h = None                                             # dual-output: per-frame <POINT> hiddens (B,G,H), read by point_head; None when POINT_HEAD off
        if self.point_out and multi_ground:
            self._point_h = h[input_ids == self.point_id].view(B, -1, self.H)
        reg_h = h.new_zeros(register.shape[0], register.shape[1], self.H)  # (B,R,H), padded slots = 0
        reg_h[rvalid] = h[regm]                                            # place real <REG> hiddens into valid slots
        self._inst_h = None                                                # ExpA: contextualized <INST> output hiddens (B,M,H) for matching
        if self.inst_append and inst_emb is not None:
            self._inst_h = h.new_zeros(inst_emb.shape[0], inst_emb.shape[1], self.H)
            self._inst_h[inst_valid] = h[input_ids == self.inst_id]
        if multi_readout:                                                 # G5: also LAST <GROUND> (R2 turn)
            last_idx = (gmask.size(1) - 1) - torch.flip(gmask, [1]).float().argmax(1)
            ground_last = h[ar, last_idx]
            return (temp_h, ground_h, ground_last, reg_h, reg_frame) + ((lm_logits,) if return_logits else ())
        return (temp_h, ground_h, reg_h, reg_frame) + ((lm_logits,) if return_logits else ())

    @torch.no_grad()
    def gen_boxes(self, input_ids, attention_mask, register, reg_frame, max_new_tokens=160, eos_id=None, pad_id=None, return_scores=False, do_sample=False, temperature=1.0, **mm):
        """Arm C: inject registers into <REG> slots (SAME as encode) then autoregressively GENERATE the
        per-frame box-list text after the '... Boxes:' prefix. Returns generated token ids (continuation only);
        return_scores=True -> also return per-step prob of the emitted token (B,gen_len) for box confidence."""
        base = self.vlm.get_base_model() if hasattr(self.vlm, "get_base_model") else self.vlm
        emb = base.get_input_embeddings()(input_ids)
        regm = (input_ids == self.reg_id)
        reg_in = F.normalize(register.float(), dim=-1).to(self.reg_proj.weight.dtype)
        rp = self.reg_proj(reg_in)
        if self.reg17:
            slot = (torch.arange(register.shape[1], device=rp.device) % 17)
            rp = rp + self.reg_slot_emb(slot).to(rp.dtype)[None]
        rp = self.reg_ln(rp)
        tok_norm = emb.detach().norm(dim=-1).mean().clamp(min=1e-6).to(rp.dtype)
        rp = rp / rp.norm(dim=-1, keepdim=True).clamp(min=1e-6) * tok_norm
        rvalid = reg_frame.ge(0)
        emb = emb.clone(); emb[regm] = rp[rvalid].to(emb.dtype)               # scatter registers (identical to encode)
        if return_scores:
            out = self.vlm.generate(inputs_embeds=emb, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
                                    do_sample=do_sample, temperature=temperature, eos_token_id=eos_id, pad_token_id=pad_id,
                                    output_scores=True, return_dict_in_generate=True, **mm)
            seq = out.sequences                                              # (B, gen_len) continuation
            tok_prob = torch.stack([torch.softmax(out.scores[t].float(), -1).gather(1, seq[:, t:t+1]).squeeze(1)
                                    for t in range(len(out.scores))], 1)     # (B, gen_len) prob of emitted token
            return seq, tok_prob
        # do_sample/temperature are explicit params (default False/1.0 = the previous hardcoded behaviour, so
        # every existing caller is bit-identical). RL needs to SAMPLE here; passing it through **mm collided.
        return self.vlm.generate(inputs_embeds=emb, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
                                 do_sample=do_sample, temperature=temperature, eos_token_id=eos_id, pad_token_id=pad_id, **mm)

    def temporal_logits(self, temp_h, reg_h, reg_frame, S):
        """per-frame logit (option c): maxpool a frame's <REG> hiddens, concat <TEMP>, MLP."""
        B = reg_h.size(0); wdt = self.temporal_head.net[0].weight.dtype
        logits = reg_h.new_full((B, S), float("-inf"))
        for b in range(B):
            for f in range(S):
                m = reg_frame[b] == f
                if m.any():
                    feat = reg_h[b][m].max(0).values                      # maxpool over that frame's reg tokens
                    logits[b, f] = self.temporal_head(torch.cat([feat, temp_h[b]]).to(wdt))[0]
        return logits                                                     # (B,S)

    def count_pred(self, temp_h):
        """E4: predict #frames-to-select from <TEMP> hidden (softplus -> positive count). (B,)"""
        wdt = self.count_head.net[0].weight.dtype
        return F.softplus(self.count_head(temp_h.to(wdt)).squeeze(-1))

    def action_logit(self, temp_h):
        """E2: logit that the current frame selection is insufficient (-> refine rather than ground). (B,)"""
        wdt = self.action_head.net[0].weight.dtype
        return self.action_head(temp_h.to(wdt)).squeeze(-1)

    def match_logits(self, ground_h, inst_emb, inst_mask):
        """G1/G3: <GROUND> vs segmenter instance query embeddings -> CE over instances."""
        t = F.normalize(self.text_proj(ground_h.to(self.text_proj.net[0].weight.dtype)), dim=-1)
        e = F.normalize((inst_emb.to(t.dtype) if self.no_inst_proj else self.inst_proj(inst_emb.to(t.dtype))), dim=-1)  # ablation: raw teacher qemb (no query-side proj)
        logits = torch.einsum("bp,bmp->bm", t, e) * self.logit_scale.exp()
        return logits.masked_fill(~inst_mask, float("-inf"))

    def inst_match_logits(self, ground_h, inst_h, inst_mask):
        """ExpA: <GROUND> vs the CONTEXTUALIZED <INST> output hiddens (instance queries that went through the VLM) -> CE over instances."""
        t = F.normalize(self.text_proj(ground_h.to(self.text_proj.net[0].weight.dtype)), dim=-1)
        e = F.normalize(self.inst_out_proj(inst_h.to(t.dtype)), dim=-1)
        logits = torch.einsum("bp,bmp->bm", t, e) * self.logit_scale.exp()
        return logits.masked_fill(~inst_mask, float("-inf"))

    def null_logit(self, ground_h):
        """SFRAME: scalar logit that the described object is ABSENT in this single frame. (B,1)"""
        wdt = self.null_head.net[0].weight.dtype
        return self.null_head(ground_h.to(wdt))

    def decode_mask(self, ground_h, mask_feat):
        """G2/G4: <GROUND> -> segmenter query space -> mask_embed -> dot frozen per-frame mask_feat field.
        mask_feat (B,S,256,h,w) -> per-frame mask logits (B,S,h,w)."""
        q = self.gq_proj(ground_h.to(self.gq_proj.net[0].weight.dtype))   # (B,256)
        me = self.mask_embed(q)                                           # (B,256)
        return torch.einsum("bd,bsdhw->bshw", me.to(mask_feat.dtype), mask_feat)
