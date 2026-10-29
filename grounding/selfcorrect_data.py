#!/usr/bin/env python
"""Phase-2 Stage-2 data: 2-pass self-correction. One sample = ONE window trajectory (phase2_sft.jsonl):
  Turn-1  [prompt + 8 frames] <think>cot</think> Frame1: <REG>*17 <GROUND> ... Boxes: {PASS-1 boxes}   <- model's OWN, MASKED
  [REVIEW] F1: box=.. qemb=.. · ...                                                                      <- MASKED
  Turn-2  <done/>  |  <reflect>..</reflect> Boxes(corrected): F1: <box>..</box> ...                      <- LM LOSS here only
Register (s4vy reg of the sel frames) injected into <REG> exactly like training. NO spatial/qemb loss on Turn-1
(it is the model's own tendency = context). Replay (full Arm C GT) is interleaved by train_phase2 to prevent
Pass-1 degradation. Dump keys are LOCAL frame idx 0..MF_NF-1; sel = actual frames (jpg + register indexing)."""
import os, json, numpy as np
import _env
from torch.utils.data import Dataset
from PIL import Image
REG, GROUND, THINK_S, THINK_E = "<REG>", "<GROUND>", "<think>", "</think>"
P2_T2_GROUND = os.environ.get("P2_T2_GROUND", "1") == "1"   # 0 -> Turn-2 emits NO <GROUND>. The per-frame
#   <REG>x17 are still injected, so Turn-2's INPUT is byte-for-byte what g2 saw; only the readout position is
#   gone, so the second turn cannot re-decide the instance and corrects the box list alone.
MF_PROMPT = 'Look at these {n} video frames. For each frame, if the object described below appears in it, ground that object; if it does not appear in that frame, report absent.\nDescription: "{d}"'
D = _env.data_root()
SC2D = os.environ.get("SCANNET200_2D_ROOT", "/PATH/TO/codebase/SegVGGT/data_processing/scannet200/scenes_2d")
REGc = f"{D}/s4vy_register_cache"
R9c = f"{D}/s4vy_train_cache"        # qemb: Turn-2 re-emits the <GROUND> readout, so it needs the instance queries to match against
MF_NF = int(os.environ.get("MF_NF", "8"))

def pass1_box_str(pb, nf=MF_NF):
    """model's Pass-1 boxes as the training boxstr format (local idx keys)."""
    parts = []
    for k in range(nf):
        b = pb.get(str(k))
        if b: parts.append(f"F{k+1}: <box>{int(b[0]*1000)},{int(b[1]*1000)},{int(b[2]*1000)},{int(b[3]*1000)}</box>")
        else: parts.append(f"F{k+1}: absent")
    return " ".join(parts)

class SelfCorrectDataset(Dataset):
    def __init__(self, sft_jsonl):
        self.rows = [json.loads(l) for l in open(sft_jsonl)]
        self._rc = {}                                                     # per-scene register cache (amortize npz)
    def __len__(self): return len(self.rows)
    def _reg(self, sc):
        if sc not in self._rc: self._rc = {sc: np.load(f"{REGc}/{sc}.npz")["register"]}   # keep only current scene (worker-local)
        return self._rc[sc]
    def __getitem__(self, i):
        r = self.rows[i]; sc = r["scene"]; sel = r["sel"]
        ims = []
        for f in sel:
            im = Image.open(f"{SC2D}/{sc}/color/{f}.jpg").convert("RGB")
            if im.width > 512: im = im.resize((512, int(im.height * 512 / im.width)))
            ims.append(im)
        reg = self._reg(sc)[sel].astype(np.float32)                       # (MF_NF,17,2048)
        return {"regime": "p2", "images": ims, "reg": reg, "desc": r["desc"], "cot": r["cot"], "correct": r["correct"],
                "pass1_str": pass1_box_str(r["pass1_box"]), "review": r["review"], "turn2": r["turn2"],
                "turn2_reflect": r.get("turn2_reflect", r["turn2"]), "turn2_boxes": r.get("turn2_boxes", ""),
                "qemb": np.load(f"{R9c}/{sc}.npz", allow_pickle=True)["qemb"].astype(np.float32),
                "turn2_qtargets": r.get("turn2_qtargets", [-1] * MF_NF),
                "has_ground": bool(r.get("turn2_has_ground", True))}

def build_p2_text(b, nf=MF_NF, so_far=True):
    """Return (prefix, turn2): prefix = everything up to & including [REVIEW] (MASKED); turn2 = correction (LOSS).
    so_far=True bundles the vision content placeholders in prefix (for plen re-tokenize with images)."""
    content = [{"type": "image"} for _ in range(nf)] + [{"type": "text", "text": MF_PROMPT.format(n=nf, d=b["desc"])}]
    return content   # tokenization happens in collate (needs the processor)

def make_collate(proc):
    reg_id = proc.tokenizer.convert_tokens_to_ids(REG)
    ground_id = proc.tokenizer.convert_tokens_to_ids(GROUND)
    def collate_phase2(batch):
        import torch
        nf = MF_NF; proc.tokenizer.padding_side = "right"                 # loss span at [plen:] -> right-pad
        texts, images, prefixes = [], [], []
        for b in batch:
            content = [{"type": "image"} for _ in range(nf)] + [{"type": "text", "text": MF_PROMPT.format(n=nf, d=b["desc"])}]
            t = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
            t += THINK_S + b["cot"] + THINK_E                             # Turn-1 CoT (masked)
            for j in range(nf): t += f' Frame {j+1}: ' + REG * 17 + GROUND
            t += " Boxes: " + b["pass1_str"]                             # Turn-1 Pass-1 boxes (model's OWN, masked)
            t += " " + b["review"]                                       # [REVIEW] injection (masked)
            prefix = t                                                   # everything up to here = MASKED
            # Turn-2 has the SAME STRUCTURE as Turn-1 -- reasoning, then the per-frame <GROUND> readout blocks,
            # then the box list -- so the correction covers BOTH heads. Without the second readout the qemb
            # instance choice is frozen at Pass-1 and reflection can only ever fix boxes.
            t += " " + b["turn2_reflect"]
            # P2-g3: the box list is ALWAYS re-emitted (a continuous coordinate always needs refining, and 0 of
            # 2379 done windows had boxes exactly equal to GT), while the <GROUND> readout is emitted ONLY when
            # the gate says the instance choice must change -- otherwise 90% of readout frames were supervised
            # to reproduce Pass-1, i.e. to learn an identity map.
            if b["has_ground"]:
                for j in range(nf): t += f' Frame {j+1}: ' + REG * 17 + (GROUND if P2_T2_GROUND else "")
            if b["turn2_boxes"]: t += " " + b["turn2_boxes"]              # empty under DONE_NO_BOX: <done/> ends the turn
            t += proc.tokenizer.eos_token                                 # EOS in the loss span teaches the model to STOP
            texts.append(t); images.append(b["images"]); prefixes.append(prefix)
        enc = proc(text=texts, images=images, return_tensors="pt", padding=True)
        ids = enc["input_ids"]; lm = torch.full_like(ids, -100)
        for j in range(ids.shape[0]):
            plen = int(proc(text=[prefixes[j]], images=[images[j]], return_tensors="pt")["attention_mask"][0].sum())
            lm[j, plen:] = ids[j, plen:]; lm[j, enc["attention_mask"][j] == 0] = -100   # loss ONLY on Turn-2, mask pad
        lm[(ids == reg_id) | (ids == ground_id)] = -100                  # readout slots are injected/read, never predicted
        # register now covers BOTH turns' <REG> runs: injection scatters into <REG> positions in order, so the
        # same per-frame registers are repeated for Turn-1 and Turn-2.
        nrun = [2 if b["has_ground"] else 1 for b in batch]               # <REG> runs present in this sample's text
        R = max(nrun) * nf * 17
        reg_pad = torch.zeros(len(batch), R, 2048); regfr = torch.full((len(batch), R), -1, dtype=torch.long)
        for j, b in enumerate(batch):
            one = torch.from_numpy(b["reg"]).reshape(nf * 17, 2048)
            k = nrun[j]; reg_pad[j, :k * nf * 17] = torch.cat([one] * k, 0)
            regfr[j, :k * nf * 17] = torch.arange(nf).repeat_interleave(17).repeat(k)
        M = max(b["qemb"].shape[0] for b in batch); P = batch[0]["qemb"].shape[1]
        ie = torch.zeros(len(batch), M, P); im_ = torch.zeros(len(batch), M, dtype=torch.bool)
        tgt = torch.zeros(len(batch), nf, dtype=torch.long)
        for j, b in enumerate(batch):
            n = b["qemb"].shape[0]; ie[j, :n] = torch.from_numpy(b["qemb"]); im_[j, :n] = True
            tgt[j] = torch.tensor([(M if (q < 0 or q >= n) else q) for q in b["turn2_qtargets"]])   # null -> column M
        d = dict(enc=dict(enc), register=reg_pad, reg_frame=regfr, regime=["p2"] * len(batch),
                 nreg_f=[k * nf for k in nrun], lm_labels=lm, inst_emb=ie, inst_mask=im_)
        # no <GROUND> in Turn-2 -> no readout to supervise (the loss reads the LAST nf <GROUND> hiddens)
        if P2_T2_GROUND and all(b["has_ground"] for b in batch): d["p2_qtargets"] = tgt   # BS=1 -> batch homogeneous
        return d
    return collate_phase2
