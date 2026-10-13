"""Differentiable spatial gradients on validated interior point graphs.

The geometry-dependent weighted least-squares operator is built once, then
applied to predicted scalar/vector fields using only differentiable tensor
operations. Neighborhood selection and the validity mask are intentionally
treated as fixed: training needs gradients with respect to predicted fields,
not with respect to point coordinates or discrete graph connectivity.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.utils import scatter


@dataclass(frozen=True)
class GradientOperator:
    """Geometry-only coefficients for a local weighted-least-squares gradient."""

    edge_index: torch.Tensor       # [2, E], directed center -> neighbor
    coefficients: torch.Tensor     # [E, 3], maps field differences to xyz derivatives
    valid_mask: torch.Tensor       # [N]
    degree: torch.Tensor           # [N]
    condition_ratio: torch.Tensor  # [N], lambda_min / lambda_max before ridge
    scale: torch.Tensor            # [N], RMS stencil radius in input coordinate units
    num_nodes: int

    def to(self, device=None, dtype=None):
        coefficient_dtype = dtype or self.coefficients.dtype
        return GradientOperator(
            edge_index=self.edge_index.to(device=device),
            coefficients=self.coefficients.to(device=device, dtype=coefficient_dtype),
            valid_mask=self.valid_mask.to(device=device),
            degree=self.degree.to(device=device),
            condition_ratio=self.condition_ratio.to(device=device, dtype=coefficient_dtype),
            scale=self.scale.to(device=device, dtype=coefficient_dtype),
            num_nodes=self.num_nodes,
        )


def build_gradient_operator(pos, edge_index, ridge=3e-2, distance_power=2.0,
                            min_neighbors=3, condition_tol=1e-6,
                            solver_dtype=None, detach=True):
    """Build coefficients for differentiable gradients on a fixed point graph.

    For center ``i`` and neighbors ``j``, the returned edge coefficient ``c_ij``
    gives ``grad(f)_i = sum_j c_ij * (f_j - f_i)``. Coordinates are normalized
    by each stencil's RMS radius, so ``ridge`` is dimensionless.
    
    """
    if pos.ndim != 2 or pos.size(1) != 3:
        raise ValueError(f"pos must have shape [N, 3], got {tuple(pos.shape)}")
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    if not pos.is_floating_point():
        raise TypeError("pos must be floating point")
    if ridge < 0 or distance_power < 0 or condition_tol < 0:
        raise ValueError("ridge, distance_power and condition_tol must be non-negative")
    if min_neighbors < 3:
        raise ValueError("min_neighbors must be at least 3 for a 3D gradient")

    num_nodes = pos.size(0)
    if edge_index.numel() == 0:
        raise ValueError("edge_index contains no edges")
    if edge_index.min() < 0 or edge_index.max() >= num_nodes:
        raise ValueError("edge_index contains a node index outside pos")

    calculation_dtype = solver_dtype or pos.dtype
    position = pos.to(dtype=calculation_dtype)
    edges = edge_index.long()
    center, neighbor = edges
    nonself = center != neighbor
    edges = edges[:, nonself]
    center, neighbor = edges
    if center.numel() == 0:
        raise ValueError("edge_index contains only self edges")

    dx = position[neighbor] - position[center]
    degree = torch.bincount(center, minlength=num_nodes)
    squared_radius = scatter(
        dx.square().sum(dim=1), center, dim=0, dim_size=num_nodes, reduce="sum")
    scale = (squared_radius / degree.clamp_min(1).to(calculation_dtype)).sqrt()
    safe_scale = scale.clamp_min(torch.finfo(calculation_dtype).eps)

    normalized_dx = dx / safe_scale[center, None]
    distance = normalized_dx.norm(dim=1).clamp_min(torch.finfo(calculation_dtype).eps)
    weight = distance.pow(-distance_power)
    weighted_dx = weight[:, None] * normalized_dx
    gram_terms = weighted_dx[:, :, None] * normalized_dx[:, None, :]
    gram = scatter(gram_terms, center, dim=0, dim_size=num_nodes, reduce="sum")

    eigenvalues = torch.linalg.eigvalsh(gram)
    largest = eigenvalues[:, -1]
    condition_ratio = eigenvalues[:, 0] / largest.clamp_min(
        torch.finfo(calculation_dtype).eps)
    valid = (
        (degree >= min_neighbors)
        & torch.isfinite(scale)
        & (scale > 0)
        & torch.isfinite(condition_ratio)
        & (largest > 0)
        & (condition_ratio > condition_tol)
    )

    eye = torch.eye(3, device=pos.device, dtype=calculation_dtype).expand(num_nodes, 3, 3)
    regularized = gram + ridge * eye
    safe_matrix = torch.where(valid[:, None, None], regularized, eye)
    inverse = torch.linalg.solve(safe_matrix, eye)
    coefficients = torch.einsum(
        "eij,ej->ei", inverse[center], weighted_dx) / safe_scale[center, None]
    coefficients = coefficients * valid[center, None]

    operator = GradientOperator(
        edge_index=edges,
        coefficients=coefficients,
        valid_mask=valid,
        degree=degree,
        condition_ratio=condition_ratio,
        scale=scale,
        num_nodes=num_nodes,
    )
    if detach:
        operator = GradientOperator(
            edge_index=operator.edge_index.detach(),
            coefficients=operator.coefficients.detach(),
            valid_mask=operator.valid_mask.detach(),
            degree=operator.degree.detach(),
            condition_ratio=operator.condition_ratio.detach(),
            scale=operator.scale.detach(),
            num_nodes=operator.num_nodes,
        )
    return operator


def apply_gradient_operator(values, operator: GradientOperator):
    """Apply a fixed operator to ``values [N]`` or ``values [N, C]``.

    Returns ``[N, 3]`` for a scalar input and ``[N, C, 3]`` for a multi-channel
    input. Invalid stencils are zero; use ``operator.valid_mask`` to mask losses.
    """
    scalar_input = values.ndim == 1
    if scalar_input:
        values = values[:, None]
    if values.ndim != 2 or values.size(0) != operator.num_nodes:
        raise ValueError(
            f"values must have shape [{operator.num_nodes}] or "
            f"[{operator.num_nodes}, C], got {tuple(values.shape)}")
    if values.device != operator.coefficients.device:
        raise ValueError("values and operator must be on the same device")

    center, neighbor = operator.edge_index
    difference = values[neighbor] - values[center]
    coefficients = operator.coefficients.to(dtype=values.dtype)
    contribution = coefficients[:, :, None] * difference[:, None, :]
    coordinate_first = scatter(
        contribution, center, dim=0, dim_size=operator.num_nodes, reduce="sum")
    gradient = coordinate_first.transpose(1, 2)
    return gradient[:, 0] if scalar_input else gradient


def estimate_velocity_pressure_gradients(pos, vel, press, edge_index, operator=None,
                                         **operator_kwargs):
    """Convenience wrapper returning dataset-compatible flattened gradients."""
    if operator is None:
        operator = build_gradient_operator(pos, edge_index, **operator_kwargs)
    grad_v = apply_gradient_operator(vel, operator).reshape(pos.size(0), 9)
    pressure = press[:, 0] if press.ndim == 2 and press.size(1) == 1 else press
    grad_p = apply_gradient_operator(pressure, operator)
    return grad_v, grad_p, operator.valid_mask, operator
