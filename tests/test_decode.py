import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch


# Import the decoder directly: the engine package also imports the external VGGT-Omega backbone.
_spec = importlib.util.spec_from_file_location(
    "decode", Path(__file__).resolve().parents[1] / "segomega" / "engine" / "decode.py"
)
decode = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(decode)


class MasksToIdmapTest(unittest.TestCase):
    def setUp(self):
        self.objectness = torch.tensor([[[10.0, -10.0]]])

    def decode_probabilities(self, probabilities, out_width=None, **kwargs):
        probabilities = torch.tensor(probabilities).reshape(1, 1, 1, 1, -1)
        return decode.masks_to_idmap(
            self.objectness,
            torch.logit(probabilities),
            (1, out_width or probabilities.shape[-1]),
            min_pixels=0,
            **kwargs,
        )

    def test_non_default_threshold(self):
        for threshold, expected in ((0.3, [1, 1, 1, 0]), (0.8, [1, 0, 0, 0])):
            with self.subTest(mask_thr=threshold):
                idmap, scores = self.decode_probabilities(
                    [0.9, 0.6, 0.4, 0.2], mask_thr=threshold
                )
                np.testing.assert_array_equal(idmap, [[expected]])
                self.assertEqual(scores.shape, (1,))

    def test_default_threshold(self):
        idmap, _ = self.decode_probabilities([0.9, 0.6, 0.5, 0.2])
        np.testing.assert_array_equal(idmap, [[[1, 1, 0, 0]]])

    def test_threshold_applies_after_upsampling(self):
        for threshold, expected in ((0.3, [1, 1, 1, 0]), (0.8, [1, 0, 0, 0])):
            with self.subTest(mask_thr=threshold):
                idmap, _ = self.decode_probabilities(
                    [0.9, 0.2], out_width=4, mask_thr=threshold
                )
                np.testing.assert_array_equal(idmap, [[expected]])

    def test_no_instances_above_threshold(self):
        idmap, scores = self.decode_probabilities([0.6, 0.2], out_width=4, mask_thr=0.8)
        np.testing.assert_array_equal(idmap, np.zeros((1, 1, 4), dtype=np.int32))
        self.assertEqual(scores.shape, (0,))


if __name__ == "__main__":
    unittest.main()
