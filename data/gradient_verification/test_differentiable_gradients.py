from __future__ import annotations

import unittest

import torch
from torch.autograd import gradcheck
from torch_geometric.data import Data

from angioflow.data.gradient_verification.differentiable_gradients import (
    apply_gradient_operator,
    build_gradient_operator,
    estimate_velocity_pressure_gradients,
)
from angioflow.data.gradient_verification.test_verify_interior_gradients import (
    complete_directed_edges,
)
from angioflow.data.gradient_verification.verify_interior_gradients import (
    estimate_local_gradients,
)


class DifferentiableGradientTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.num_nodes = 18
        self.pos = torch.randn(self.num_nodes, 3, dtype=torch.float64) * 4e-3
        self.edge_index = complete_directed_edges(self.num_nodes)

    def test_exact_linear_velocity_and_pressure(self):
        grad_v_truth = torch.tensor([
            [2.0, -3.0, 4.0],
            [5.0, 6.0, -7.0],
            [8.0, -9.0, 10.0],
        ], dtype=torch.float64)
        grad_p_truth = torch.tensor([11.0, -12.0, 13.0], dtype=torch.float64)
        velocity = self.pos @ grad_v_truth.T + torch.tensor([0.3, -0.2, 0.7])
        pressure = (self.pos @ grad_p_truth + 101325.0).unsqueeze(1)

        grad_v, grad_p, valid, _ = estimate_velocity_pressure_gradients(
            self.pos, velocity, pressure, self.edge_index, ridge=0.0)

        self.assertTrue(valid.all())
        torch.testing.assert_close(
            grad_v, grad_v_truth.reshape(1, 9).expand(self.num_nodes, -1),
            rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(
            grad_p, grad_p_truth.expand(self.num_nodes, -1), rtol=1e-9, atol=1e-8)

    def test_matches_existing_reference_estimator(self):
        velocity = torch.randn(self.num_nodes, 3, dtype=torch.float64)
        pressure = torch.randn(self.num_nodes, 1, dtype=torch.float64)
        data = Data(pos=self.pos, vel=velocity, press=pressure,
                    edge_index=self.edge_index)

        reference_v, reference_p, reference_valid = estimate_local_gradients(data)
        actual_v, actual_p, actual_valid, _ = estimate_velocity_pressure_gradients(
            self.pos, velocity, pressure, self.edge_index)

        self.assertTrue(torch.equal(actual_valid, reference_valid))
        torch.testing.assert_close(actual_v[actual_valid], reference_v[actual_valid],
                                   rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(actual_p[actual_valid], reference_p[actual_valid],
                                   rtol=1e-10, atol=1e-10)

    def test_autograd_gradcheck_scalar_and_vector(self):
        operator = build_gradient_operator(
            self.pos, self.edge_index, ridge=3e-2, solver_dtype=torch.float64)
        scalar = torch.randn(self.num_nodes, dtype=torch.float64, requires_grad=True)
        vector = torch.randn(self.num_nodes, 3, dtype=torch.float64, requires_grad=True)

        self.assertTrue(gradcheck(
            lambda value: apply_gradient_operator(value, operator),
            (scalar,), eps=1e-6, atol=1e-5, rtol=1e-4))
        self.assertTrue(gradcheck(
            lambda value: apply_gradient_operator(value, operator),
            (vector,), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_physics_loss_backpropagates_to_velocity(self):
        operator = build_gradient_operator(self.pos.float(), self.edge_index)
        velocity = torch.randn(self.num_nodes, 3, requires_grad=True)
        gradient = apply_gradient_operator(velocity, operator)
        divergence = gradient[:, 0, 0] + gradient[:, 1, 1] + gradient[:, 2, 2]
        loss = divergence[operator.valid_mask].square().mean()
        loss.backward()

        self.assertIsNotNone(velocity.grad)
        self.assertTrue(torch.isfinite(velocity.grad).all())
        self.assertGreater(float(velocity.grad.abs().sum()), 0.0)

    def test_invalid_coplanar_stencils_are_masked(self):
        pos = torch.tensor([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
        ])
        edges = complete_directed_edges(len(pos))
        operator = build_gradient_operator(pos, edges, ridge=0.0)
        gradient = apply_gradient_operator(torch.randn(5, 2), operator)
        self.assertFalse(operator.valid_mask.any())
        self.assertEqual(float(gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
