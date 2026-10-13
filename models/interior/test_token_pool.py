from __future__ import annotations

import unittest

import torch

from angioflow.models.interior.token_pool import GeometryLatentResampler


class GeometryLatentResamplerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.dim = 32
        self.counts = (11, 5, 8)
        self.batch = torch.repeat_interleave(
            torch.arange(len(self.counts)), torch.tensor(self.counts))
        self.tokens = torch.randn(sum(self.counts), self.dim, requires_grad=True)
        self.pos = torch.randn(sum(self.counts), 3)

    def test_fixed_shape_batch_vector_and_gradients(self):
        pool = GeometryLatentResampler(
            self.dim, n_tokens=6, n_heads=4, n_layers=2)
        out, out_batch = pool(self.tokens, self.pos, self.batch)

        self.assertEqual(out.shape, (3 * 6, self.dim))
        self.assertTrue(torch.equal(
            out_batch, torch.arange(3).repeat_interleave(6)))
        out.square().mean().backward()
        self.assertIsNotNone(self.tokens.grad)
        self.assertGreater(float(self.tokens.grad.abs().sum()), 0.0)
        self.assertIsNotNone(pool.latents.grad)

    def test_invariant_to_source_order_within_each_graph(self):
        pool = GeometryLatentResampler(
            self.dim, n_tokens=4, n_heads=4, n_layers=1).eval()
        order = torch.cat([
            torch.arange(lo, lo + count)[torch.randperm(count)]
            for lo, count in zip((0, 11, 16), self.counts)
        ])
        with torch.no_grad():
            expected, expected_batch = pool(self.tokens, self.pos, self.batch)
            actual, actual_batch = pool(
                self.tokens[order], self.pos[order], self.batch[order])
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.equal(actual_batch, expected_batch))

    def test_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            GeometryLatentResampler(30, n_tokens=4, n_heads=4)
        with self.assertRaises(ValueError):
            GeometryLatentResampler(32, n_tokens=0, n_heads=4)


if __name__ == "__main__":
    unittest.main()
