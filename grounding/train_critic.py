#!/usr/bin/env python
"""Two-path selector, trained on GT. The model emits two independent answers per query and they are strongly
complementary (700-query oracle 48.7 vs 38.4 for the better single path), so a selector is worth more than
anything Phase-2 achieved. GPT-5.6 is measured on the same task as a reference point, but it is NOT the
teacher: both candidates' true IoU is already computed, so the labels are free and the ceiling is the oracle
rather than GPT.

INPUT is g2's real predictions, rendered exactly as they are at inference -- otherwise the critic would meet a
different distribution when deployed.

THREE THINGS THE DATA DOES, all of which matter more than the loss:
  * ORDER IS RANDOMISED PER SAMPLE PER EPOCH and the branch identity never appears. In the cards on disk, A is
    always qemb and B always the box path; trained on that, the model can learn "prefer the first" -- which is
  * BOTH-BAD PAIRS ARE DROPPED. When neither candidate clears the threshold the choice cannot move box@.5, yet
    such a pair can carry a large |dIoU| and would soak up gradient. Dropped, not down-weighted.
  * STRATIFIED 70/30 discriminable vs near-tie. Pure discriminable data would leave the critic uncalibrated on
    the ~60% of real queries where the two paths nearly agree.

Env: CARDS, OUT_DIR, EPOCHS, LR, BS, GRAD_ACC, STRAT, TIE_FRAC
"""
import os, sys, json, glob, random
import _env
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoProcessor, AutoModelForImageTextToText

D = os.environ.get("GROUNDING_DATA_ROOT", "/PATH/TO/data")
CARDS = os.environ.get("CARDS", f"{D}/critic_cards_scanrefer")
CARD_DIRS = [c for c in CARDS.split(":") if c]
# Resolved lazily: the inference entry points import Critic from this module, and must not be
# made to supply training paths just to construct the class.
def model_path():
    return _env.require("MODEL_PATH", "Qwen3-VL-2B-Instruct checkout for the critic")


def out_dir():
    d = os.environ.get("OUT_DIR") or os.path.join(_env.require("AGENT_CKPT_ROOT", "checkpoint root"), "critic_2b")
    os.makedirs(d, exist_ok=True)
    return d


EPOCHS = int(os.environ.get("EPOCHS", "3")); LR = float(os.environ.get("LR", "5e-6"))
BS = int(os.environ.get("BS", "1")); GA = int(os.environ.get("GRAD_ACC", "8"))
TIE_FRAC = float(os.environ.get("TIE_FRAC", "0.30"))    # share of near-ties kept, for calibration
TIE_THR = float(os.environ.get("TIE_THR", "0.10"))
ALPHA = float(os.environ.get("ALPHA", "0.5"))           # extra weight for pairs that straddle 0.5
PROMPT = ('Two candidate objects are highlighted in magenta, one per image; each image shows three views of '
          'the same scene with the original frame beside the highlighted one.\nReferring expression: "{d}"\n'
          'Which candidate does the expression refer to?')

PROMPT_PF = ('Two candidate segmentations of the same object in a video are shown on the SAME {k} frames, '
             'sampled uniformly in time.\n'
             'Images 1-{k}: candidate 1, in temporal order.\n'
             'Images {k1}-{k2}: candidate 2, on exactly the same {k} frames, in the same order.\n'
             'Every image is [the original frame | the same frame with that candidate\'s mask filled in magenta].\n'
             'Referring expression: "{d}"\n'
             'Across the whole video, which candidate better segments the object the expression refers to?')


PROMPT_PF2 = ('You are given {k} frames of the same scene, sampled where the two candidate segmentations '
              'disagree most.\n'
              'Each image is [the original frame | candidate 1 in MAGENTA | candidate 2 in CYAN] on that frame.\n'
              'Referring expression: "{d}"\n'
              'Overall, which candidate segments the referred object better?')

PROMPT_PF3 = ('You are given {k} frames of the same scene, sampled where the two candidate segmentations '
              'disagree most.\n'
              'Each image is [candidate 1\'s mask in MAGENTA | candidate 2\'s mask in CYAN], both drawn on '
              'that same frame as a semi-transparent overlay with a coloured outline.\n'
              'Referring expression: "{d}"\n'
              'Overall, which candidate segments the referred object better?')


def build_pairs():
    ms = []
    for _d in CARD_DIRS:
        for f in glob.glob(f"{_d}/meta_*.jsonl"):
            for l in open(f):
                try:
                    _m = json.loads(l); _m["_dir"] = _d; ms.append(_m)
                except Exception: pass
    SR = f"{D}/scanrefer/annotations/scanrefer/ScanRefer_filtered.json"
    desc = {(x["scene_id"], str(x["object_id"]), str(x.get("ann_id"))): x["description"]
            for x in json.load(open(SR))}
    out = []
    for m in ms:
        if not m.get("desc"):
            m["desc"] = desc.get((m["scene"], str(m["oid"]), str(m["ann"])), "")
        if not m["desc"]: continue
        a, b = m["A_biou"], m["B_biou"]
        if max(a, b) < 0.5: continue                     # both bad -> the choice cannot move box@.5, drop
        out.append(m)
    disc = [m for m in out if abs(m["A_biou"] - m["B_biou"]) >= TIE_THR]
    tie = [m for m in out if abs(m["A_biou"] - m["B_biou"]) < TIE_THR]
    random.Random(0).shuffle(tie)
    keep_tie = tie[:int(len(disc) * TIE_FRAC / max(1e-9, 1 - TIE_FRAC))]
    pairs = disc + keep_tie
    random.Random(1).shuffle(pairs)
    _ho = float(os.environ.get("HOLDOUT", "0") or 0)
    if _ho > 0:
        import hashlib
        def _h(k): return int(hashlib.md5(k.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        test = [m for m in pairs if _h(m["key"]) < _ho]
        pairs = [m for m in pairs if _h(m["key"]) >= _ho]
        with open(f"{out_dir()}/holdout_keys.json", "w") as f:
            json.dump(sorted(m["key"] for m in test), f)
        print(f"[critic] held out {len(test)} pairs (key hash<{_ho}), train {len(pairs)} ", flush=True)
    if os.environ.get("CLS_BAL", "0") == "1":
        from collections import defaultdict
        cnt = defaultdict(lambda: [0, 0])
        for m in pairs:
            k = m.get("_dir", "?")
            cnt[k][0] += int(m["A_biou"] > m["B_biou"]); cnt[k][1] += 1
        for m in pairs:
            a, n = cnt[m.get("_dir", "?")]
            pA = a / max(1, n)
            pA = min(max(pA, 1e-3), 1 - 1e-3)
            m["_wcls"] = (0.5 / pA) if m["A_biou"] > m["B_biou"] else (0.5 / (1 - pA))
        print("[critic] class balancing enabled; per-source P(A better) -> loss weights:", flush=True)
        for k, (a, n) in sorted(cnt.items()):
            pA = a / max(1, n)
            print(f"    {os.path.basename(k):28s} A better {a}/{n} = {100*pA:.1f}%  "
                  f"A-weight x{0.5/max(pA,1e-3):.3f}  B-weight x{0.5/max(1-pA,1e-3):.3f}", flush=True)
    return pairs, len(ms), len(disc), len(keep_tie)


class PairSet(Dataset):
    def __init__(self, pairs): self.p = pairs
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        m = self.p[i]
        # order randomised EVERY time this sample is drawn -> branch identity carries no positional signal
        flip = random.random() < 0.5
        _d = m.get("_dir", CARD_DIRS[0])
        _f = str(m.get("fmt", "card3"))
        k = int(_f[len("perframe"):]) if _f.startswith("perframe") else 1
        if _f.startswith("pairframe") or _f.startswith("pfab"):
            _ab = _f.startswith("pfab")
            kk = int(_f[len("pfab"):] if _ab else _f[len("pairframe"):])
            _pk = int(os.environ.get("PAIRK", "0")) or kk
            _sel = ([int(x) for x in np.linspace(0, kk - 1, _pk).round()] if _pk < kk else list(range(kk)))
            tag = "Q" if flip else "P"
            imgs = [f"{_d}/img/{m['key']}__{tag}{j}.jpg" for j in _sel]
            kk = len(_sel)
            ga, gb = m["A_biou"], m["B_biou"]
            g1, g2 = (gb, ga) if flip else (ga, gb)
            return dict(imgs=[Image.open(p).convert("RGB") for p in imgs], k=(-1000 - kk) if _ab else -kk,
                        desc=m["desc"],
                        y=(1.0 if g1 > g2 else 0.0),
                        w=(abs(g1 - g2) + ALPHA * float((max(g1, g2) >= .5) != (min(g1, g2) >= .5)))
                          * float(m.get("_wcls", 1.0)))
        if k > 1:
            pa = [f"{_d}/img/{m['key']}__A{j}.jpg" for j in range(k)]
            pb = [f"{_d}/img/{m['key']}__B{j}.jpg" for j in range(k)]
        else:
            pa = [f"{_d}/img/{m['key']}__A.jpg"]; pb = [f"{_d}/img/{m['key']}__B.jpg"]
        first, second = (pb, pa) if flip else (pa, pb)
        ga, gb = m["A_biou"], m["B_biou"]
        g1, g2 = (gb, ga) if flip else (ga, gb)
        y = 1.0 if g1 > g2 else 0.0                       # label is about POSITION, never about branch
        w = (abs(g1 - g2) + ALPHA * float((max(g1, g2) >= .5) != (min(g1, g2) >= .5))) * float(m.get("_wcls", 1.0))
        return dict(imgs=[Image.open(p).convert("RGB") for p in (first + second)],
                    k=k, desc=m["desc"], y=y, w=w)


class Critic(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            path, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa")
        if os.environ.get("GRAD_CKPT", "1") == "1":
            try:
                self.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                self.vlm.config.use_cache = False
                print("[critic] gradient checkpointing enabled", flush=True)
            except Exception as e:
                print(f"[critic] gradient checkpointing failed to enable: {type(e).__name__}: {e}", flush=True)
        tc = self.vlm.config.get_text_config() if hasattr(self.vlm.config, "get_text_config") else self.vlm.config
        self.head = nn.Linear(getattr(tc, "hidden_size", 2048), 1)   # single logit: P(first is better)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.head = self.head.to(next(self.vlm.parameters()).dtype)
    def forward(self, **enc):
        base = self.vlm.get_base_model() if hasattr(self.vlm, "get_base_model") else self.vlm
        out = base(**enc, output_hidden_states=True)
        h = out.hidden_states[-1]
        last = enc["attention_mask"].sum(1) - 1
        x = h[torch.arange(h.shape[0], device=h.device), last]
        return self.head(x.to(self.head.weight.dtype)).squeeze(-1).float()


def make_collate(proc):
    def coll(batch):
        texts, images = [], []
        for b in batch:
            k = b.get("k", 1)
            if k <= -1000: txt = PROMPT_PF3.format(k=-1000 - k, d=b["desc"])
            elif k < 0: txt = PROMPT_PF2.format(k=-k, d=b["desc"])
            elif k > 1: txt = PROMPT_PF.format(k=k, k1=k + 1, k2=2 * k, d=b["desc"])
            else:       txt = PROMPT.format(d=b["desc"])
            c = [{"type": "image"} for _ in b["imgs"]] + [{"type": "text", "text": txt}]
            texts.append(proc.apply_chat_template([{"role": "user", "content": c}], tokenize=False,
                                                  add_generation_prompt=True))
            images.append(b["imgs"])
        enc = proc(text=texts, images=images, return_tensors="pt", padding=True)
        return dict(enc), torch.tensor([b["y"] for b in batch]), torch.tensor([b["w"] for b in batch])
    return coll


def main():
    set_seed(0)
    acc = Accelerator(gradient_accumulation_steps=GA)
    pairs, ntot, nd, nt = build_pairs()
    if acc.is_main_process:
        print(f"[critic] cards {ntot} -> train pairs {len(pairs)} (separable {nd} + near-tie {nt}); "
              f"pairs with both arms <0.5 dropped", flush=True)
    if not pairs: print("[critic] no training data"); return
    model = Critic(model_path())
    dl = DataLoader(PairSet(pairs), batch_size=BS, shuffle=True, num_workers=4,
                    collate_fn=make_collate(model.proc))
    _hp = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head.")]
    _bp = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
    HEAD_LR = float(os.environ.get("HEAD_LR", "0") or 0)
    if HEAD_LR > 0:
        opt = torch.optim.AdamW([{"params": _bp, "lr": LR}, {"params": _hp, "lr": HEAD_LR}], lr=LR)
        if acc.is_main_process:
            print(f"[critic] separate head lr {HEAD_LR} ({len(_hp)} tensors), backbone {LR}", flush=True)
    else:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    model, opt, dl = acc.prepare(model, opt, dl)
    step = 0
    for ep in range(EPOCHS):
        for enc, y, w in dl:
            with acc.accumulate(model):
                logit = model(**enc)
                loss = (w.to(logit.device) * F.binary_cross_entropy_with_logits(
                    logit, y.to(logit.device).float(), reduction="none")).mean()
                acc.backward(loss)
                if acc.sync_gradients: acc.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            step += 1
            if acc.is_main_process and step % 20 == 0:
                print(f"[critic] ep{ep} step{step} loss {loss.item():.4f}", flush=True)
        acc.wait_for_everyone()
        sd = acc.get_state_dict(model)
        if acc.is_main_process:
            torch.save(sd, f"{out_dir()}/full_model.pt")
            json.dump({"epoch": ep + 1, "step": step}, open(f"{out_dir()}/meta.json", "w"))
            print(f"[critic] saved @ep{ep+1}", flush=True)
    acc.wait_for_everyone()
    if acc.is_main_process:
        open(f"{out_dir()}/DONE", "w").write("done")
        print("CRITIC_DONE", flush=True)


if __name__ == "__main__": main()
