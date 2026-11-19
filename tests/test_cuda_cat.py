"""Run with python -m unittest discover -s tests -v.

Set S4VY_RUN_LARGE_CUDA_TESTS=1 for the real 4-GiB regressions (24 GiB free
VRAM recommended), and S4VY_RUN_COMPILE_TESTS=1 for Inductor coverage.
"""
import os
import unittest
from unittest.mock import patch

import torch

from segomega.utils.cuda_cat import SafeCudaCat, safe_cuda_cat


class CatTests(unittest.TestCase):
    def test_cpu_values_and_errors(self):
        a, b = torch.randn(2, 3), torch.randn(2, 3)
        with SafeCudaCat():
            torch.testing.assert_close(torch.cat([a, b], -1), safe_cuda_cat([a, b], -1), rtol=0, atol=0)
            out = torch.empty(4, 3)
            self.assertIs(torch.cat([a, b], out=out), out)
            torch.testing.assert_close(out, torch.cat([a, b]), rtol=0, atol=0)
        for tensors, dim in [([], 0), ([a, b], 2), ([a, b[:, :1]], 0)]:
            try:
                torch.cat(tensors, dim)
            except (ValueError, RuntimeError, IndexError) as native_error:
                with SafeCudaCat(), self.assertRaises(type(native_error)) as caught:
                    torch.cat(tensors, dim)
                self.assertEqual(str(caught.exception), str(native_error))
            else:
                self.fail("Invalid native cat unexpectedly succeeded")

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_copy_values_gradients_and_scope(self):
        original = torch.cat
        # Force the workaround on small inputs to check gradients cheaply.
        with patch('segomega.utils.cuda_cat._CUDA_CAT_BYTE_LIMIT', 0):
            for dim in [0, 1, -1]:
                a = torch.randn(2, 3, 4, device='cuda', requires_grad=True)
                b = torch.randn_like(a, requires_grad=True)
                expected = torch.cat([a, b], dim)
                with SafeCudaCat():
                    actual = torch.cat(tensors=[a, b], dim=dim, out=None)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                weight = torch.randn_like(actual)
                gradients = torch.autograd.grad((actual * weight).sum(), (a, b))
                reference = torch.autograd.grad((expected * weight).sum(), (a, b))
                for got, want in zip(gradients, reference):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)
            a = a.detach().transpose(1, 2)
            for b in [a * 2, a.double()]:
                expected = torch.cat([a, b], 1)
                with SafeCudaCat():
                    actual = torch.cat([a, b], 1)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertIs(torch.cat, original)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_graph(self):
        a = torch.randn(2, 3, device='cuda')
        b = torch.randn_like(a)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with patch('segomega.utils.cuda_cat._CUDA_CAT_BYTE_LIMIT', 0):
            with torch.cuda.stream(stream), SafeCudaCat():
                for _ in range(3):
                    torch.cat([a, b], -1)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream), SafeCudaCat():
                actual = torch.cat([a, b], -1)
            a.fill_(3); b.fill_(7)
            graph.replay()
            torch.testing.assert_close(actual, torch.cat([a, b], -1), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get('S4VY_RUN_COMPILE_TESTS') == '1', 'opt-in Inductor test')
    def test_compiled_block(self):
        def block(a, b):
            return torch.cat([a + 1, b * 2], -1)
        a, b = torch.randn(2, 3, device='cuda'), torch.randn(2, 3, device='cuda')
        expected = block(a, b)
        with patch('segomega.utils.cuda_cat._CUDA_CAT_BYTE_LIMIT', 0):
            compiled = torch.compile(block, fullgraph=True)
            for _ in range(2):
                with SafeCudaCat():
                    actual = compiled(a, b)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get('S4VY_RUN_LARGE_CUDA_TESTS') == '1', 'opt-in large CUDA regression')
    def test_real_feature_boundaries(self):
        # 225 is just below 4 GiB; 226 and 256 reproduce the cached-feature
        # corruption. 782 reproduces the different register-token layout.
        for frames, layout in [(225, 'cache'), (226, 'cache'), (256, 'cache'), (782, 'register')]:
            with self.subTest(frames=frames, layout=layout), torch.no_grad():
                if layout == 'cache':
                    a = torch.ones((1, frames, 2321, 1024), device='cuda')
                    b = torch.full_like(a, 2)
                    dim = -1
                else:
                    a = torch.ones((1, frames, 17, 1024), device='cuda')
                    b = torch.full((1, frames, 2304, 1024), 2., device='cuda')
                    dim = 2
                with SafeCudaCat():
                    actual = torch.cat([a, b], dim)
                first, second = actual.split([a.shape[dim], b.shape[dim]], dim)
                self.assertEqual(int((first != 1).sum()), 0)
                self.assertEqual(int((second != 2).sum()), 0)
                self.assertTrue(actual[0, -1, -1, -1024:].cpu().eq(2).all())
                del a, b, actual, first, second
                torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
