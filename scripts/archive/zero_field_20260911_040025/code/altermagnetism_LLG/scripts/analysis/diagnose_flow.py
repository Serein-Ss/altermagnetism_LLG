"""Run pre-training diagnostics for the current conditional flow checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from altermagnetism_LLG.scripts.analysis.evaluate import load_model  # noqa: E402
from altermagnetism_LLG.scripts.analysis.evaluation import (  # noqa: E402
    calibrate_basins,
    generate_flow,
    neel_observables,
    path_descriptors,
    summarize,
)

from altermagnetism_LLG.scripts.core.project_paths import generated_path
from altermagnetism_LLG.scripts.visualization.model_plots import _plot


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--calibration-input", type=Path, nargs="+", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--system", default="d_wave_altermagnet")
    p.add_argument(
        "--integration-steps", type=int, nargs="+",
        default=(16, 32, 64, 128),
    )
    p.add_argument("--integration-seeds", type=int, default=4)
    p.add_argument("--diversity-seeds", type=int, default=16)
    p.add_argument("--condition-id", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p


def _pair_metrics(left: np.ndarray, right: np.ndarray) -> dict:
    return {
        "endpoint_mean_absolute_difference": float(
            np.abs(left[:, -1] - right[:, -1]).mean()
        ),
        "neel_z_path_rmse": float(
            np.sqrt(np.mean((left - right) ** 2))
        ),
        "maximum_neel_z_absolute_difference": float(
            np.abs(left - right).max()
        ),
    }


def _path_spread(paths: np.ndarray) -> float:
    if len(paths) < 2:
        return 0.0
    differences = []
    for left in range(len(paths)):
        for right in range(left + 1, len(paths)):
            differences.append(
                np.sqrt(np.mean((paths[left] - paths[right]) ** 2))
            )
    return float(np.mean(differences))




@torch.no_grad()
def main() -> None:
    args = parser().parse_args()
    if (
        min(args.integration_steps) < 1
        or args.integration_seeds < 1
        or args.diversity_seeds < 16
    ):
        raise ValueError(
            "integration steps/seeds must be positive; diversity requires >=16 seeds"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; use a Slurm GPU allocation"
        )
    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.overwrite
    ):
        raise FileExistsError(
            f"refusing to overwrite nonempty {args.output_dir}; pass --overwrite"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_target = generated_path(args.output_dir / "diagnostic_trajectories.h5")
    if trajectory_target.exists() and not args.overwrite:
        raise FileExistsError(trajectory_target)
    model, checkpoint = load_model(args.checkpoint, "flow", args.device)
    calibration = calibrate_basins(
        args.calibration_input, args.system, seed=args.seed
    )

    with h5py.File(args.input, "r", swmr=True) as source:
        group = source[args.system]
        time = source["time"][:].astype(np.float32)
        condition_ids = group["condition_id"][:]
        test = group["split"][:] == 2
        condition_indices = {}
        for condition_id in np.unique(condition_ids):
            indices = np.flatnonzero(
                (condition_ids == condition_id) & test
            )
            if not len(indices):
                raise ValueError(
                    f"condition {condition_id} has no test path"
                )
            condition_indices[int(condition_id)] = int(indices[0])
        if args.condition_id not in condition_indices:
            raise ValueError(
                f"condition {args.condition_id} is unavailable"
            )
        fixed_index = condition_indices[args.condition_id]
        initial = torch.from_numpy(
            group["initial_spins"][fixed_index]
        ).unsqueeze(0).to(args.device)
        vector = torch.from_numpy(
            group["vector_condition"][fixed_index]
        ).unsqueeze(0).to(args.device)
        scalar_by_condition = {
            condition_id: torch.from_numpy(
                group["model_condition"][index]
            ).unsqueeze(0).to(args.device)
            for condition_id, index in condition_indices.items()
        }
        drive_by_condition = {
            condition_id: float(
                group["physical_condition"][index, 2]
            )
            for condition_id, index in condition_indices.items()
        }
        reference_spins = group["spins"][fixed_index]
        reference_row = path_descriptors(
            reference_spins, time, calibration
        )

    physical_time = torch.from_numpy(time).unsqueeze(0).to(args.device)
    output_h5 = generated_path(args.output_dir / "diagnostic_trajectories.h5")
    if output_h5.exists():
        output_h5.unlink()
    integration_paths = {}
    integration_orders = {}
    scalar = scalar_by_condition[args.condition_id]
    with h5py.File(output_h5, "w") as output:
        output.create_dataset("time", data=time)
        integration_group = output.create_group("integration_convergence")
        for steps in args.integration_steps:
            samples = []
            orders = []
            for seed_offset in range(args.integration_seeds):
                sample = generate_flow(
                    model, initial, scalar, vector, physical_time,
                    steps, args.seed + seed_offset,
                )[0].cpu().numpy().astype(np.float32)
                samples.append(sample)
                orders.append(neel_observables(sample)[0])
            integration_paths[steps] = np.stack(samples)
            integration_orders[steps] = np.stack(orders)
            integration_group.create_dataset(
                f"steps_{steps}", data=integration_paths[steps],
                compression="gzip", compression_opts=4, shuffle=True,
            )
            print(f"integration steps={steps}", flush=True)

        sensitivity_group = output.create_group("condition_sensitivity")
        sensitivity_samples = []
        sensitivity_orders = []
        ordered_conditions = sorted(
            condition_indices, key=drive_by_condition.get
        )
        for condition_id in ordered_conditions:
            sample = generate_flow(
                model, initial, scalar_by_condition[condition_id],
                vector, physical_time, max(args.integration_steps), args.seed,
            )[0].cpu().numpy().astype(np.float32)
            sensitivity_samples.append(sample)
            sensitivity_orders.append(neel_observables(sample)[0])
        sensitivity_samples = np.stack(sensitivity_samples)
        sensitivity_orders = np.stack(sensitivity_orders)
        sensitivity_group.create_dataset(
            "spins", data=sensitivity_samples,
            compression="gzip", compression_opts=4, shuffle=True,
        )
        sensitivity_group.create_dataset(
            "drives_T",
            data=np.asarray(
                [drive_by_condition[item] for item in ordered_conditions]
            ),
        )
        print("condition sensitivity complete", flush=True)

        diversity_group = output.create_group("same_initial_diversity")
        diversity_samples = []
        diversity_rows = []
        for seed_offset in range(args.diversity_seeds):
            sample = generate_flow(
                model, initial, scalar, vector, physical_time,
                max(args.integration_steps),
                args.seed + 10_000 + seed_offset,
            )[0].cpu().numpy().astype(np.float32)
            diversity_samples.append(sample)
            diversity_rows.append(
                path_descriptors(sample, time, calibration)
            )
        diversity_samples = np.stack(diversity_samples)
        diversity_orders = np.stack(
            [row["order"] for row in diversity_rows]
        )
        diversity_group.create_dataset(
            "spins", data=diversity_samples,
            compression="gzip", compression_opts=4, shuffle=True,
        )
        diversity_group.create_dataset(
            "latent_seeds",
            data=np.arange(
                args.seed + 10_000,
                args.seed + 10_000 + args.diversity_seeds,
            ),
        )
        print("same-initial diversity complete", flush=True)

    reference_steps = max(args.integration_steps)
    convergence = {
        str(steps): _pair_metrics(
            integration_orders[steps],
            integration_orders[reference_steps],
        )
        for steps in args.integration_steps
        if steps != reference_steps
    }
    step32 = convergence.get("32")
    integration_converged = bool(
        step32 is not None
        and step32["endpoint_mean_absolute_difference"] <= 0.02
        and step32["neel_z_path_rmse"] <= 0.02
    )
    sensitivity_endpoint_span = float(
        np.ptp(sensitivity_orders[:, -1])
    )
    sensitivity_path_spread = _path_spread(sensitivity_orders)
    condition_sensitive = bool(
        sensitivity_endpoint_span >= 0.05
        or sensitivity_path_spread >= 0.05
    )
    diversity_endpoint_std = float(
        diversity_orders[:, -1].std()
    )
    diversity_path_spread = _path_spread(diversity_orders)
    mode_collapse = bool(
        diversity_endpoint_std < 0.01
        and diversity_path_spread < 0.01
    )
    report = {
        "status": "pretraining_diagnostics_complete",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "fixed_source_index": fixed_index,
        "fixed_condition_id": args.condition_id,
        "time_ps": (time * 1e12).tolist(),
        "basin_calibration": calibration.to_dict(),
        "reference_fixed_initial_path": {
            key: value
            for key, value in reference_row.items()
            if key not in ("order", "spatial_std")
        },
        "integration_convergence": {
            "reference_steps": reference_steps,
            "thresholds": {
                "endpoint_mean_absolute_difference": 0.02,
                "neel_z_path_rmse": 0.02,
            },
            "comparisons_to_reference": convergence,
            "steps_32_converged": integration_converged,
            "neel_z_paths": {
                str(key): value.tolist()
                for key, value in integration_orders.items()
            },
        },
        "condition_sensitivity": {
            "drives_T": [
                drive_by_condition[item]
                for item in ordered_conditions
            ],
            "fixed_latent_seed": args.seed,
            "endpoint_span": sensitivity_endpoint_span,
            "mean_pairwise_path_rmse": sensitivity_path_spread,
            "sensitivity_floor": 0.05,
            "condition_sensitive": condition_sensitive,
            "neel_z_paths": sensitivity_orders.tolist(),
        },
        "same_initial_diversity": {
            "condition_drive_T": drive_by_condition[args.condition_id],
            "latent_seed_count": args.diversity_seeds,
            "endpoint_std": diversity_endpoint_std,
            "mean_pairwise_path_rmse": diversity_path_spread,
            "collapse_floor": 0.01,
            "mode_collapse_detected": mode_collapse,
            "summary": summarize(diversity_rows),
            "neel_z_paths": diversity_orders.tolist(),
        },
        "diagnosis": {
            "sampling_integration_error": not integration_converged,
            "condition_failure": not condition_sensitive,
            "mode_collapse": mode_collapse,
            "rule": (
                "32-step error is compared with paired 128-step paths; "
                "condition and diversity floors detect near-invariance, "
                "not full physical calibration"
            ),
        },
        "trajectory_file": str(output_h5.resolve()),
    }
    report_path = args.output_dir / "diagnostics.json"
    report_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _plot(report, args.output_dir / "diagnostics.png")
    print(json.dumps({
        "status": report["status"],
        "diagnosis": report["diagnosis"],
        "output": str(args.output_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
