"""Memory-mode parity, training supervision and graph fallback.

Head integration tests require VGGT_OMEGA_ROOT, as does the model itself.
"""
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from segomega.engine.fast_inference import GraphRunner
from segomega.engine.inference_memory import configure_inference_memory
from segomega.utils.host_tokens import HostTokens


class HostTokenTests(unittest.TestCase):
    def test_values_devices_and_slices(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices += [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
        for device in devices:
            tokens = [torch.randn(1, 2, 7, 16, device=device), None]
            host = HostTokens(tokens)
            self.assertEqual(host._values[0][0].device.type, "cpu")
            self.assertEqual(host[0].device, device)
            torch.testing.assert_close(host[0], tokens[0], rtol=0, atol=0)
            self.assertIsNone(host[-1])
            self.assertEqual(len(host[:]), 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_graph_runner_bypasses_capture_for_offload(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = SimpleNamespace(offload_tokens=True)

            def forward(self, images):
                return {"value": images * 2}

        runner = GraphRunner(Model().eval())
        with patch.object(runner, "_capture", side_effect=AssertionError("CPU offload must not capture")):
            for value in [3., 7.]:
                x = torch.full((1, 2, 3, 4, 4), value, device="cuda")
                torch.testing.assert_close(runner(x)["value"], x * 2, rtol=0, atol=0)
        self.assertFalse(runner.graphs)


@unittest.skipUnless(Path(os.environ.get("VGGT_OMEGA_ROOT", "/missing")).is_dir(), "VGGT_OMEGA_ROOT required")
class HeadMemoryTests(unittest.TestCase):
    def test_final_static_prompt_outputs_and_training_gradients(self):
        from segomega.models.mask_feature_head import MaskFeatureHead
        from segomega.models.query_decoder import QueryInstanceDecoder

        torch.manual_seed(17)
        head = MaskFeatureHead(dim_in=32, features=32, mask_dim=32,
                               out_channels=[32] * 4, intermediate_layer_idx=[0, 1, 2, 3]).eval()
        decoder = QueryInstanceDecoder(token_dim=32, mask_dim=32, dec_dim=32,
                                       num_queries=5, num_layers=2, num_heads=4, memory_layer_idx=3).eval()
        model = nn.Module()
        model.backbone = nn.Identity()
        model.mask_feature_head, model.query_decoder = head, decoder
        model.eval()
        tokens = [torch.randn(1, 2, 21, 32) for _ in range(4)]
        images = torch.rand(1, 2, 3, 64, 64)
        prompt = dict(prompt_q=torch.randn(1, 2, 32), prompt_q_pos=torch.randn(1, 2, 32),
                      prompt_valid=torch.tensor([[True, False]]))

        def forward(values):
            return decoder(values, 5, head(values, images, 5), (4, 4), **prompt)

        with torch.no_grad():
            reference = forward(tokens)
            configure_inference_memory(model, offload_backbone=True)
            actual = forward(HostTokens(tokens))
        for key in ["objectness", "mask_logits", "mask_embed_final", "prompt_objectness", "prompt_mask_logits", "prompt_mask_embed_final"]:
            torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)
        self.assertEqual(len(reference["aux_outputs"]), 2)
        self.assertEqual(actual["aux_outputs"], [])
        self.assertEqual(actual["prompt_aux_outputs"], [])

        # Returning to training must restore deep supervision and gradients.
        model.train()
        train = forward(tokens)
        self.assertEqual(len(train["aux_outputs"]), 2)
        self.assertEqual(len(train["prompt_aux_outputs"]), 2)
        loss = train["mask_logits"].square().mean()
        for auxiliary in train["aux_outputs"] + train["prompt_aux_outputs"]:
            loss = loss + auxiliary["mask_logits"].square().mean()
        loss.backward()
        for parameter in [head.projects[0].weight, decoder.query_feat.weight, decoder.layers[0].cross_attn.in_proj_weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        with self.assertRaises(ValueError):
            configure_inference_memory(model)


if __name__ == "__main__":
    unittest.main()
