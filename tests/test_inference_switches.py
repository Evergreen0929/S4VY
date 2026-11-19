"""The two inference switches: --mask-id-enhance (decoder) and --speedup (fast_inference).

Neither is part of the paper; with both off the entry points must reproduce the paper's run, and
--speedup must leave the LoRA adapters (and so the LoRA-off geometry path) untouched.
"""
import unittest

import numpy as np
import torch
from torch import nn

from segomega.engine.decode import (MAX_INPUT_LONG_SIDE, decode_idmap, masks_to_idmap_owned,
                                    masks_to_idmap_paper, model_input_hw)
from segomega.engine.fast_inference import _lora_state, cache_bf16_weights
from segomega.models.lora import LoRALinear, inject_lora, set_lora_enabled


def _preds(obj_p, mask_logits):
    """objectness logits whose softmax[:, 0] equals obj_p, plus (1,N,S,h,w) mask logits."""
    p = torch.tensor(obj_p)
    objectness = torch.stack([p.log(), (1 - p).log()], -1)[None]
    return {"objectness": objectness, "mask_logits": mask_logits[None]}


class DecoderSwitchTests(unittest.TestCase):
    def setUp(self):
        # query 0: confident, small (4x4); query 1: less confident, large (8x8) and covering query 0
        m = torch.full((2, 1, 8, 8), -10.0)
        m[0, 0, 2:6, 2:6] = 10.0
        m[1] = 3.0
        self.preds = _preds([0.9, 0.5], m)

    def test_paper_decoder_larger_mask_wins(self):
        idmap, scores = masks_to_idmap_paper(self.preds["objectness"], self.preds["mask_logits"], (8, 8))
        self.assertEqual(idmap.shape, (1, 8, 8))
        self.assertEqual(set(np.unique(idmap)), {2})                  # the large mask painted last covers all
        np.testing.assert_allclose(scores, [0.9, 0.5], rtol=1e-6)    # ids 1..n in ascending area

    def test_owned_decoder_confident_query_wins(self):
        idmap, scores = masks_to_idmap_owned(self.preds["objectness"], self.preds["mask_logits"], (8, 8))
        self.assertTrue((idmap[0, 2:6, 2:6] == 1).all())            # id = query index + 1
        self.assertEqual(int((idmap == 2).sum()), 64 - 16)

    def test_switch_dispatch(self):
        for flag, fn in ((True, masks_to_idmap_owned), (False, masks_to_idmap_paper)):
            a = decode_idmap(self.preds, (8, 8), mask_id_enhance=flag)
            b = fn(self.preds["objectness"], self.preds["mask_logits"], (8, 8))
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])


class NativeInputTests(unittest.TestCase):
    def test_native_keeps_frames_as_given(self):
        for hw in ((384, 688), (448, 592), (288, 928), (690, 920), (481, 333)):
            self.assertEqual(model_input_hw(*hw, "native"), hw)

    def test_native_caps_oversized_frames(self):
        h, w = model_input_hw(1080, 1920, "native")
        self.assertLessEqual(max(h, w), MAX_INPUT_LONG_SIDE)
        self.assertEqual((h % 16, w % 16), (0, 0))


class _Agg(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
        self.conv = nn.Conv2d(3, 8, 3)

    def forward(self, x):
        return self.blocks(x)


class SpeedupLeavesLoRATests(unittest.TestCase):
    def _model(self):
        torch.manual_seed(0)
        agg = _Agg()
        inject_lora(agg, rank=4, alpha=4.0, target_substrings=("0", "2"), verbose=False)
        for m in agg.modules():
            if isinstance(m, LoRALinear):
                nn.init.normal_(m.lora_b.weight, std=0.1)              # a non-zero adapter
        return agg.eval()

    def test_adapters_are_not_cached_or_merged(self):
        agg = self._model()
        n_lora = sum(isinstance(m, LoRALinear) for m in agg.modules())
        self.assertEqual(n_lora, 2)
        cached = cache_bf16_weights(agg)
        self.assertEqual(cached, 3)                                    # two LoRA base Linears + the conv
        for m in agg.modules():
            if isinstance(m, LoRALinear):
                self.assertFalse(hasattr(m.lora_a, "_bf16_weight"))
                self.assertFalse(hasattr(m.lora_b, "_bf16_weight"))
                self.assertTrue(hasattr(m.base, "_bf16_weight"))
        self.assertEqual(set_lora_enabled(agg, False), n_lora)       # the LoRA switch still finds every adapter

    def test_lora_state_in_graph_key(self):
        agg = self._model()
        on = _lora_state(agg)
        set_lora_enabled(agg, False)
        self.assertNotEqual(_lora_state(agg), on)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_bf16_cache_bitwise_equals_autocast_with_lora_on_and_off(self):
        ref, fast = self._model().cuda(), self._model().cuda()
        cache_bf16_weights(fast)
        x = torch.randn(5, 7, 32, device="cuda")
        for flag in (True, False):
            set_lora_enabled(ref, flag); set_lora_enabled(fast, flag)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                a, b = ref(x), fast(x)
            torch.testing.assert_close(b, a, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
