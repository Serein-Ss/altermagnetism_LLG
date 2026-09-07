"""Gate AFM/altermagnet path data by physical scale and numerical convergence."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "core"))

from unified_llg import CommonSOT, GomonayModelAdapter, UnifiedLLGSolver, collinear_state  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=(8, 16, 32, 48, 64))
    p.add_argument("--paths", type=int, default=100)
    p.add_argument("--temperature", type=float, default=5.0)
    p.add_argument("--drive", type=float, default=0.8)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--dt", type=float, default=1e-16)
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--equilibration-steps", type=int, default=2_000)
    p.add_argument("--pulse-duration", type=float, default=0.5e-12)
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--afm-control", action="store_true")
    p.add_argument("--skip-dt-scan", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "data" / "size_convergence" / "spatial_convergence.json")
    return p


def wilson(successes: int, count: int, z: float = 1.95996398454) -> tuple[float, float]:
    if count == 0:
        return 0.0, 1.0
    p = successes / count
    denominator = 1.0 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def regularized_rate_per_cell_ps(events: int, paths: int, size: int, duration_s: float) -> float:
    """Estimate Poisson event intensity using a Jeffreys-style half-count."""
    probability = (events + 0.5) / (paths + 1.0)
    return -math.log1p(-probability) / (size * size * duration_s / 1e-12)


@torch.no_grad()
def simulate(args, size: int, dt: float, steps: int, seed_offset: int) -> dict:
    model = GomonayModelAdapter(alternating_exchange=not args.afm_control)
    solver = UnifiedLLGSolver(
        model,
        alpha=args.alpha,
        sot=CommonSOT(damping_like=args.drive, start=0.0, end=args.pulse_duration),
    )
    preparation = UnifiedLLGSolver(model, alpha=args.alpha)
    spins = collinear_state(args.paths, 2, size, size, antiferromagnetic=True, device=args.device, dtype=torch.float64)
    generator_device = args.device if args.device.startswith("cuda") else "cpu"
    generators = [
        torch.Generator(device=generator_device).manual_seed(args.seed + seed_offset + i)
        for i in range(args.paths)
    ]
    equilibration_steps = int(round(args.equilibration_steps * args.dt / dt))
    for step in range(equilibration_steps):
        spins = preparation.stochastic_heun_step(spins, step * dt, dt, args.temperature, generators)

    save_every = min(args.save_every, steps)
    saved_steps = list(range(0, steps + 1, save_every))
    if saved_steps[-1] != steps:
        saved_steps.append(steps)
    order = []
    spatial_std = []
    wall_density = []
    save_index = 0
    for step in range(steps + 1):
        if save_index < len(saved_steps) and step == saved_steps[save_index]:
            local = 0.5 * (spins[:, 0, ..., 2] - spins[:, 1, ..., 2])
            order.append(local.mean(dim=(1, 2)).cpu().numpy())
            spatial_std.append(local.std(dim=(1, 2), correction=0).cpu().numpy())
            sign = local < 0
            walls = 0.5 * (
                (sign != torch.roll(sign, -1, 1)).to(torch.float64).mean(dim=(1, 2))
                + (sign != torch.roll(sign, -1, 2)).to(torch.float64).mean(dim=(1, 2))
            )
            wall_density.append(walls.cpu().numpy())
            save_index += 1
        if step < steps:
            spins = solver.stochastic_heun_step(spins, step * dt, dt, args.temperature, generators)

    order = np.stack(order, axis=1)
    spatial_std = np.stack(spatial_std, axis=1)
    wall_density = np.stack(wall_density, axis=1)
    time = np.asarray(saved_steps) * dt
    crossed = (order < 0).any(axis=1)
    switched = order[:, -1] < 0
    transition_std = np.where(np.abs(order) < 0.5, spatial_std, 0.0).max(axis=1)
    nonuniform = switched & (transition_std >= 0.25)
    first_passage = np.full(args.paths, np.nan)
    for i in range(args.paths):
        hits = np.flatnonzero(order[i] < 0)
        if len(hits):
            first_passage[i] = time[hits[0]]
    fine_labels, fine_fpt_index = classify_paths(order, spatial_std)
    cadence = []
    for stride in (2, 4):
        indices = np.arange(0, order.shape[1], stride)
        if indices[-1] != order.shape[1] - 1:
            indices = np.append(indices, order.shape[1] - 1)
        coarse_labels, coarse_fpt_index = classify_paths(order[:, indices], spatial_std[:, indices])
        coarse_fpt_original_index = np.full_like(coarse_fpt_index, np.nan)
        finite_coarse = np.isfinite(coarse_fpt_index)
        coarse_fpt_original_index[finite_coarse] = indices[coarse_fpt_index[finite_coarse].astype(int)]
        mismatch = np.isfinite(fine_fpt_index) != np.isfinite(coarse_fpt_original_index)
        both = np.isfinite(fine_fpt_index) & np.isfinite(coarse_fpt_original_index)
        if mismatch.any():
            max_error_intervals = None
        elif both.any():
            max_error_intervals = float(np.max(np.abs(fine_fpt_index[both] - coarse_fpt_original_index[both])) / stride)
        else:
            max_error_intervals = 0.0
        cadence.append(
            {
                "subsample_stride": stride,
                "saved_dt_s": stride * save_every * dt,
                "mechanism_label_agreement": float((fine_labels == coarse_labels).mean()),
                "max_first_passage_error_in_coarse_intervals": max_error_intervals,
            }
        )
    low, high = wilson(int(nonuniform.sum()), args.paths)
    duration_s = steps * dt
    return {
        "size": size,
        "linear_nm": size * model.parameters.lattice_nm,
        "dt_s": dt,
        "steps": steps,
        "saved_dt_s": save_every * dt,
        "final_neel_z_mean": float(order[:, -1].mean()),
        "final_neel_z_std": float(order[:, -1].std(ddof=1)) if args.paths > 1 else 0.0,
        "crossing_fraction": float(crossed.mean()),
        "switching_fraction": float(switched.mean()),
        "nonuniform_switch_count": int(nonuniform.sum()),
        "nonuniform_switch_fraction": float(nonuniform.mean()),
        "nonuniform_fraction_wilson95": [low, high],
        "switching_rate_per_cell_per_ps_regularized": regularized_rate_per_cell_ps(
            int(switched.sum()), args.paths, size, duration_s
        ),
        "nonuniform_rate_per_cell_per_ps_regularized": regularized_rate_per_cell_ps(
            int(nonuniform.sum()), args.paths, size, duration_s
        ),
        "mean_peak_spatial_std": float(spatial_std.max(axis=1).mean()),
        "mean_peak_wall_density": float(wall_density.max(axis=1).mean()),
        "median_first_passage_s": float(np.nanmedian(first_passage)) if np.isfinite(first_passage).any() else None,
        "save_cadence_subsampling": cadence,
        "time_s": time.tolist(),
        "ensemble_mean_order": order.mean(axis=0).tolist(),
    }


def relative_change(a: float, b: float, floor: float = 1e-8) -> float:
    return abs(a - b) / max(abs(a), abs(b), floor)


def classify_paths(order: np.ndarray, spatial_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    crossed = (order < 0).any(axis=1)
    switched = order[:, -1] < 0
    transition_std = np.where(np.abs(order) < 0.5, spatial_std, 0.0).max(axis=1)
    labels = np.full(len(order), 0, dtype=np.int8)
    labels[crossed & ~switched] = 1
    labels[switched & (transition_std < 0.25)] = 2
    labels[switched & (transition_std >= 0.25)] = 3
    first_passage = np.full(len(order), np.nan)
    for i, path in enumerate(order):
        hits = np.flatnonzero(path < 0)
        if len(hits):
            first_passage[i] = hits[0]
    return labels, first_passage


def main() -> None:
    args = parser().parse_args()
    if len(args.sizes) < 1 or min(args.sizes) < 3 or args.paths < 1:
        raise ValueError("provide at least one size >=3 and paths >=1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = GomonayModelAdapter(alternating_exchange=not args.afm_control)
    wall_width_nm = 4.0
    wall_width_cells = wall_width_nm / model.parameters.lattice_nm
    no_overlap_cells = math.ceil(4.0 * wall_width_cells)

    size_results = []
    for index, size in enumerate(args.sizes):
        result = simulate(args, size, args.dt, args.steps, index * 10_000_000)
        size_results.append(result)
        print(f"size={size} done: nonuniform={result['nonuniform_switch_count']}/{args.paths}")

    last = size_results[-1]
    size_changes = None
    if len(size_results) >= 2:
        previous = size_results[-2]
        size_changes = {
            "final_neel_z_mean": relative_change(last["final_neel_z_mean"], previous["final_neel_z_mean"]),
            "switching_fraction_absolute": abs(last["switching_fraction"] - previous["switching_fraction"]),
            "mean_peak_spatial_std": relative_change(last["mean_peak_spatial_std"], previous["mean_peak_spatial_std"]),
            "mean_peak_wall_density": relative_change(last["mean_peak_wall_density"], previous["mean_peak_wall_density"]),
            "nonuniform_rate_per_cell_per_ps_regularized": relative_change(
                last["nonuniform_rate_per_cell_per_ps_regularized"],
                previous["nonuniform_rate_per_cell_per_ps_regularized"],
            ),
        }
    dt_results = []
    if not args.skip_dt_scan:
        for index, factor in enumerate((2.0, 1.0, 0.5)):
            dt = args.dt * factor
            duration = args.dt * args.steps
            steps = int(round(duration / dt))
            if factor == 1.0:
                dt_results.append(last)
            else:
                dt_results.append(simulate(args, args.sizes[-1], dt, steps, 800_000_000 + index * 10_000_000))
            print(f"dt={dt:.3e} done")
    dt_change = None
    if len(dt_results) >= 2:
        fine, nominal = dt_results[-1], dt_results[-2]
        dt_change = {
            "final_neel_z_mean": relative_change(fine["final_neel_z_mean"], nominal["final_neel_z_mean"]),
            "switching_fraction_absolute": abs(fine["switching_fraction"] - nominal["switching_fraction"]),
            "mean_peak_spatial_std": relative_change(fine["mean_peak_spatial_std"], nominal["mean_peak_spatial_std"]),
        }

    checks = {
        "largest_size_at_least_4_wall_widths": args.sizes[-1] >= no_overlap_cells,
        "ensemble_at_least_100_paths": args.paths >= 100,
        "nonuniform_paths_observed": last["nonuniform_switch_count"] >= 3,
        "last_two_sizes_peak_spatial_std_rel_change_le_0p10": size_changes is not None and size_changes["mean_peak_spatial_std"] <= 0.10,
        "last_two_sizes_peak_wall_density_rel_change_le_0p10": size_changes is not None and size_changes["mean_peak_wall_density"] <= 0.10,
        "last_two_sizes_nonuniform_rate_rel_change_le_0p25": size_changes is not None and size_changes["nonuniform_rate_per_cell_per_ps_regularized"] <= 0.25,
        "nominal_vs_half_dt_switch_fraction_abs_change_le_0p05": dt_change is not None and dt_change["switching_fraction_absolute"] <= 0.05,
        "nominal_vs_half_dt_peak_spatial_std_rel_change_le_0p10": dt_change is not None and dt_change["mean_peak_spatial_std"] <= 0.10,
        "fourfold_coarser_save_mechanism_labels_identical": last["save_cadence_subsampling"][-1]["mechanism_label_agreement"] == 1.0,
        "fourfold_coarser_save_fpt_error_le_one_interval": (
            last["save_cadence_subsampling"][-1]["max_first_passage_error_in_coarse_intervals"] is not None
            and last["save_cadence_subsampling"][-1]["max_first_passage_error_in_coarse_intervals"] <= 1.0
        ),
    }
    report = {
        "status": "certified_for_nonuniform_path_training" if all(checks.values()) else "not_certified",
        "scope": "weak ensemble convergence gate; production claims also require independent reruns and threshold sensitivity",
        "system": "afm_control" if args.afm_control else "d_wave_altermagnet",
        "published_scale_anchor": {
            "domain_wall_width_nm": wall_width_nm,
            "lattice_nm": model.parameters.lattice_nm,
            "domain_wall_width_cells": wall_width_cells,
            "minimum_periodic_pair_no_overlap_cells": no_overlap_cells,
            "recommended_candidate_minimum": 48,
            "recommended_production_anchor": 64,
        },
        "size_scaling_convention": (
            "Global at-least-one-event probabilities are extensive and need not become "
            "size independent. Certification compares per-cell event intensity and local "
            "wall descriptors; global probabilities remain reported observables."
        ),
        "protocol": vars(args) | {"output": str(args.output)},
        "size_results": size_results,
        "last_two_size_changes": size_changes,
        "dt_results": dt_results,
        "nominal_vs_half_dt_changes": dt_change,
        "save_cadence_rule": "subsample identical fine-cadence paths; require mechanism labels unchanged and FPT error <= one coarse interval",
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
