from __future__ import annotations

import unittest

import torch
from torch_geometric.data import Data

from angioflow.data.gradient_verification.verify_interior_gradients import (
    draw_training_subset,
    estimate_local_gradients,
    parse_optional_positive_int,
    select_case_indices,
)


def complete_directed_edges(num_nodes):
    src = torch.arange(num_nodes).repeat_interleave(num_nodes - 1)
    dst = torch.cat([
        torch.cat([torch.arange(i), torch.arange(i + 1, num_nodes)])
        for i in range(num_nodes)
    ])
    return torch.stack([src, dst])


class LocalGradientEstimatorTest(unittest.TestCase):
    def test_recovers_exact_linear_vector_and_scalar_fields(self):
        torch.manual_seed(4)
        num_nodes = 24
        pos = torch.randn(num_nodes, 3, dtype=torch.float64) * 3e-3
        grad_v = torch.tensor([
            [2.0, -3.0, 4.0],
            [5.0, 6.0, -7.0],
            [8.0, -9.0, 10.0],
        ], dtype=torch.float64)
        grad_p = torch.tensor([11.0, -12.0, 13.0], dtype=torch.float64)
        data = Data(
            pos=pos,
            vel=pos @ grad_v.T + torch.tensor([0.2, -0.4, 0.8]),
            press=(pos @ grad_p + 101325.0).unsqueeze(1),
            edge_index=complete_directed_edges(num_nodes),
        )

        est_v, est_p, valid = estimate_local_gradients(data, ridge=0.0)

        self.assertTrue(valid.all())
        expected_v = grad_v.reshape(1, 9).expand(num_nodes, -1)
        expected_p = grad_p.reshape(1, 3).expand(num_nodes, -1)
        torch.testing.assert_close(est_v, expected_v, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(est_p, expected_p, rtol=1e-9, atol=1e-8)

    def test_rejects_coplanar_stencil(self):
        pos = torch.tensor([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
        ])
        data = Data(
            pos=pos,
            vel=torch.randn(5, 3),
            press=torch.randn(5, 1),
            edge_index=complete_directed_edges(5),
        )
        _, _, valid = estimate_local_gradients(data, ridge=0.0)
        self.assertFalse(valid.any())

    def test_training_subset_is_seeded_and_capped(self):
        first = draw_training_subset(1000, ratio=0.2, max_points=75, seed=9)
        second = draw_training_subset(1000, ratio=0.2, max_points=75, seed=9)
        self.assertEqual(len(first), 75)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(len(torch.unique(first)), 75)

    def test_uncapped_subset_and_cli_value(self):
        subset = draw_training_subset(1000, ratio=0.8, max_points=None, seed=2)
        self.assertEqual(len(subset), 800)
        self.assertIsNone(parse_optional_positive_int("none"))
        self.assertEqual(parse_optional_positive_int("12000"), 12000)

    def test_case_selection_is_unique_and_balanced_by_version(self):
        class Dataset:
            _index = (
                [("aneug_flow", f"a{i}", frame) for i in range(70) for frame in (0, 20)]
                + [("angioflow", f"b{i}", frame) for i in range(80) for frame in (0, 20)]
            )

        selected = select_case_indices(Dataset(), max_cases=100, seed=3)
        keys = [Dataset._index[index][:2] for index in selected]
        versions = [version for version, _ in keys]
        self.assertEqual(len(keys), 100)
        self.assertEqual(len(set(keys)), 100)
        self.assertEqual(versions.count("aneug_flow"), 50)
        self.assertEqual(versions.count("angioflow"), 50)


if __name__ == "__main__":
    unittest.main()
