#!/usr/bin/env python
"""vLLM backend for the MFRAME box path.

WHAT IT REPLACES. Only the autoregressive generate inside _gen_mf_box_batch. Prompt construction and box parsing
stay in grounder_infer.py and are shared verbatim with the torch path, so a difference between the two backends can
only come from generation itself -- not from a prompt that drifted.

WHAT IT CANNOT REPLACE. The <GROUND> readout. vLLM returns logprobs, not arbitrary hidden states, so the qemb
half stays in torch. That half is one forward per window rather than ~60 autoregressive steps, so it is the
cheap side anyway.

THE INJECTION IS COPIED, NOT REIMPLEMENTED. agent_model.encode's exact sequence
    normalize -> reg_proj -> (+reg_slot_emb) -> reg_ln -> rescale to the sequence's mean token norm
runs here on ag's own layers. tok_norm depends on the whole sequence's token embeddings (image placeholders
included), which is why it is computed client-side from an embedding lookup rather than approximated: an eval
that silently injects something slightly different is exactly the failure this exercise exists to avoid.
"""
import os

# vLLM V1 spawns EngineCore as a SEPARATE process, and spawn re-executes the parent script through runpy as
# __main__ -- so the child re-runs grounder_infer.py with a sys.path that lacks VGGT_OMEGA_ROOT and dies on
# `No module named 'eval.eval_instance_seg'`. In-process EngineCore removes the child entirely; we drive one
# GPU per process anyway, so the separate process buys nothing here. Must be set before vllm is imported.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
import torch.nn.functional as F

_LLM = [None]
RPF = 17                                              # <REG> per frame, matching the prompt's runs


def _get_llm():
    if _LLM[0] is None:
        from vllm import LLM, SamplingParams          # noqa: F401  (import cost paid once)
        import vllm_regplugin
        vllm_regplugin.register()
        _LLM[0] = LLM(
            model=os.environ["HF_EXPORT"], dtype="bfloat16",
            gpu_memory_utilization=float(os.environ.get("VLLM_GPU_FRAC", "0.42")),
            max_model_len=int(os.environ.get("VLLM_MAXLEN", "8192")),
            enforce_eager=os.environ.get("VLLM_EAGER") == "1", enable_mm_embeds=True,
            hf_overrides={"architectures": ["Qwen3VLWithRegisters"]},
            limit_mm_per_prompt={"image": int(os.environ.get("MF_NF", "8")), "video": 0,
                                 "register": int(os.environ.get("MF_NF", "8"))},
        )
    return _LLM[0]


@torch.no_grad()
def _inject(ag, input_ids, register, reg_frame):
    """agent_model.encode's register path, verbatim. Returns (R, H) final <REG> embeddings for ONE window."""
    base = ag.vlm.get_base_model() if hasattr(ag.vlm, "get_base_model") else ag.vlm
    emb = base.get_input_embeddings()(input_ids)                        # (1,T,H) -- lookup only, no forward
    reg_in = F.normalize(register.float(), dim=-1).to(ag.reg_proj.weight.dtype)
    rp = ag.reg_proj(reg_in)
    if ag.reg17:
        slot = torch.arange(register.shape[1], device=rp.device) % RPF
        rp = rp + ag.reg_slot_emb(slot).to(rp.dtype)[None]
    rp = ag.reg_ln(rp)
    tok_norm = emb.detach().norm(dim=-1).mean().clamp(min=1e-6).to(rp.dtype)
    rp = rp / rp.norm(dim=-1, keepdim=True).clamp(min=1e-6) * tok_norm
    return rp[reg_frame.ge(0)]                                          # (R,H), padded frames dropped


@torch.no_grad()
def gen_texts(ag, proc, texts, images, enc_ids, register, reg_frame, nf, max_new_tokens=160):
    """One vLLM call for the whole window batch. vLLM's continuous batching is the point: the torch path pads
    every window to the longest and decodes them in lockstep."""
    from vllm import SamplingParams
    llm = _get_llm()
    reqs = []
    for b in range(len(texts)):
        rp = _inject(ag, enc_ids[b:b + 1], register[b:b + 1], reg_frame[b:b + 1])
        reqs.append({"prompt": texts[b],
                     "multi_modal_data": {"image": images[b],
                                          "register": rp.reshape(nf, RPF, rp.shape[-1]).float().cpu()}})
    outs = llm.generate(reqs, SamplingParams(temperature=0.0, max_tokens=max_new_tokens), use_tqdm=False)
    return [o.outputs[0].text for o in outs]
