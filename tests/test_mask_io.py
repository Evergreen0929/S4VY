import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from segomega.utils.mask_io import overlay_ids, palette, parse_color, read_idmap


class MaskDisplayTests(unittest.TestCase):
    def test_packed_and_uint16_ids_and_color_only_override(self):
        ids = np.array([[0, 147], [169, 1025]], dtype=np.uint16)
        packed = np.stack([ids % 256, ids // 256, np.zeros_like(ids)], -1).astype(np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            for name, pixels in [("uint16", ids), ("packed", packed)]:
                path = Path(folder) / f"{name}.png"
                Image.fromarray(pixels).save(path)
                np.testing.assert_array_equal(read_idmap(path), ids)
                np.testing.assert_array_equal(read_idmap(path, (4, 4)), ids.repeat(2, 0).repeat(2, 1))
        rgb = np.full((2, 2, 3), 100, np.uint8)
        previous_ids, previous_rgb = ids.copy(), rgb.copy()
        colors = palette([parse_color("147=255,145,35"), parse_color("169=55,200,245")])
        output = overlay_ids(rgb, ids, colors, alpha=1)
        np.testing.assert_array_equal(output[0, 1], [255, 145, 35])
        np.testing.assert_array_equal(output[1, 0], [55, 200, 245])
        np.testing.assert_array_equal(output[0, 0], rgb[0, 0])
        np.testing.assert_array_equal(ids, previous_ids)
        np.testing.assert_array_equal(rgb, previous_rgb)


if __name__ == "__main__":
    unittest.main()
