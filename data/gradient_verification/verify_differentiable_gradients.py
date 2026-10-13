"""Validate the differentiable gradient operator on cached AneuG-Flow/AngioFlow CFD cases.

This is deliberately separate from ``verify_interior_gradients.py``. It compares
the vectorized operator with the established CPU reference, compares both with
stored CFD gradients, and proves that a gradient/divergence loss backpropagates
to velocity and pressure inputs.

Example:
  conda activate new
  python -m angioflow.data.gradient_verification.verify_differentiable_gradients \
    --save_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_full \
    --interior_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_interior \
    --versions aneug_flow,angioflow --frames 0 --ratio 0.4 --max_cases 2 \
    --max_points none --device cpu --solver_dtype float64
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from angioflow.config.defaults import add_data_args
from angioflow.data.datasets import load_excluded_keys
from angioflow.data.gradient_verification.differentiable_gradients import (
    estimate_velocity_pressure_gradients,
)
from angioflow.data.gradient_verification.verify_interior_gradients import (
    build_dataset,
    estimate_local_gradients,
    induced_sample,
    parse_optional_positive_int,
    select_case_indices,
    summarize_components,
    summarize_graph,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Check accuracy and autograd of the vectorized gradient operator.")
    add_data_args(parser)
    parser.add_argument("--interior_dir", default=None)
    parser.add_argument("--data_pool_ratio", type=float, default=0.16)
    parser.add_argument("--freeze_dataset", action="store_true", default=True)
    parser.add_argument("--no_freeze_dataset", dest="freeze_dataset", action="store_false")
    parser.add_argument("--ratio", type=float, default=0.4)
    parser.add_argument("--max_points", type=parse_optional_positive_int, default=None)
    parser.add_argument("--max_cases", type=int, default=2,
                        help="unique cases balanced across requested versions")
    parser.add_argument("--ridge", type=float, default=3e-2)
    parser.add_argument("--distance_power", type=float, default=2.0)
    parser.add_argument("--min_neighbors", type=int, default=3)
    parser.add_argument("--condition_tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--solver_dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--skip_reference", action="store_true",
                        help="skip the slower Python/CPU reference comparison")
    parser.add_argument("--output_path", default=None)
    return parser.parse_args()


def _mean_relative_mse(estimate, truth, valid):
    return float(np.mean(summarize_components(estimate, truth, valid)[2]))


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    args = get_args()
    if args.ratio <= 0 or args.ratio > 1:
        raise ValueError("--ratio must lie in (0, 1]")
    device = torch.device(args.device)
    dtype = torch.float64 if args.solver_dtype == "float64" else torch.float32

    exclude_keys = load_excluded_keys(args.save_dir) if args.exclude_anomalies else set()
    dataset = build_dataset(args, exclude_keys)
    case_indices = select_case_indices(dataset, args.max_cases, args.seed)
    selected = [dataset._index[index] for index in case_indices]
    print(f"[setup] ratio={args.ratio:g} max_points={args.max_points} device={device} "
          f"dtype={args.solver_dtype} cases={selected}", flush=True)

    results = []
    for case_number, dataset_index in enumerate(case_indices, start=1):
        version, case_id, frame = dataset._index[dataset_index]
        print(f"[case] {case_number}/{len(case_indices)} {version}/{case_id} frame={frame}",
              flush=True)
        wall, interior = dataset[dataset_index]
        _, sampled = induced_sample(
            wall, interior, args.ratio, args.max_points, args.seed + dataset_index)
        mean_degree, edge_count = summarize_graph(sampled.edge_index, sampled.num_nodes)

        reference_v = reference_p = reference_valid = None
        reference_seconds = None
        if not args.skip_reference:
            start = time.perf_counter()
            reference_v, reference_p, reference_valid = estimate_local_gradients(
                sampled, ridge=args.ridge, min_neighbors=args.min_neighbors,
                distance_power=args.distance_power, condition_tol=args.condition_tol)
            reference_seconds = time.perf_counter() - start

        pos = sampled.pos.to(device=device, dtype=dtype)
        edge_index = sampled.edge_index.to(device)
        velocity = sampled.vel.to(device=device, dtype=dtype).detach().requires_grad_(True)
        pressure = sampled.press.to(device=device, dtype=dtype).detach().requires_grad_(True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        start = time.perf_counter()
        grad_v, grad_p, valid, operator = estimate_velocity_pressure_gradients(
            pos, velocity, pressure, edge_index, ridge=args.ridge,
            distance_power=args.distance_power, min_neighbors=args.min_neighbors,
            condition_tol=args.condition_tol, solver_dtype=dtype)
        _sync(device)
        forward_seconds = time.perf_counter() - start

        divergence = grad_v.view(-1, 3, 3).diagonal(dim1=1, dim2=2).sum(dim=1)
        physics_loss = (
            grad_v[valid].square().mean()
            + grad_p[valid].square().mean()
            + divergence[valid].square().mean()
        )
        _sync(device)
        start = time.perf_counter()
        physics_loss.backward()
        _sync(device)
        backward_seconds = time.perf_counter() - start

        actual_v = grad_v.detach().cpu().double()
        actual_p = grad_p.detach().cpu().double()
        actual_valid = valid.cpu()
        truth_v = sampled.grad_v.double().cpu()
        truth_p = sampled.grad_p.double().cpu()
        vel_rel_mse = _mean_relative_mse(actual_v, truth_v, actual_valid)
        press_rel_mse = _mean_relative_mse(actual_p, truth_p, actual_valid)

        result = {
            "version": version,
            "case_id": case_id,
            "frame": frame,
            "pool_count": interior.num_nodes,
            "sample_count": sampled.num_nodes,
            "edge_count": edge_count,
            "mean_degree": mean_degree,
            "valid_fraction": float(actual_valid.double().mean()),
            "velocity_relative_mse_vs_cfd": vel_rel_mse,
            "pressure_relative_mse_vs_cfd": press_rel_mse,
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "reference_seconds": reference_seconds,
            "velocity_input_grad_norm": float(velocity.grad.norm().detach().cpu()),
            "pressure_input_grad_norm": float(pressure.grad.norm().detach().cpu()),
            "peak_gpu_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None),
        }
        if reference_v is not None:
            common_valid = actual_valid & reference_valid
            result.update({
                "valid_mask_matches_reference": bool(torch.equal(actual_valid, reference_valid)),
                "velocity_max_abs_difference_vs_reference": float(
                    (actual_v[common_valid] - reference_v[common_valid]).abs().max()),
                "pressure_max_abs_difference_vs_reference": float(
                    (actual_p[common_valid] - reference_p[common_valid]).abs().max()),
            })
        results.append(result)
        print(
            f"[result] points={sampled.num_nodes} edges={edge_count} degree={mean_degree:.2f} "
            f"valid={result['valid_fraction']:.3f} vel_rel_mse={vel_rel_mse:.4g} "
            f"press_rel_mse={press_rel_mse:.4g} forward={forward_seconds:.2f}s "
            f"backward={backward_seconds:.2f}s vel_grad_norm="
            f"{result['velocity_input_grad_norm']:.3g}", flush=True)
        if reference_v is not None:
            print(
                f"[reference] time={reference_seconds:.2f}s mask_match="
                f"{result['valid_mask_matches_reference']} vel_max_diff="
                f"{result['velocity_max_abs_difference_vs_reference']:.3g} "
                f"press_max_diff={result['pressure_max_abs_difference_vs_reference']:.3g}",
                flush=True)

    interior_root = args.interior_dir or os.path.join(args.save_dir, "interior")
    default_root = os.path.join(
        os.path.dirname(os.path.abspath(interior_root)), "angioflow_gradient_verification")
    output_path = args.output_path or os.path.join(
        default_root, f"differentiable_gradient_check_ratio_{args.ratio:g}.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump({"config": vars(args), "results": results}, file, indent=2)
    print(f"[done] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
