"""Evaluate deterministic and flow checkpoints on the held-out L96 ensemble."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from altermagnetism_LLG.model.baseline import DeterministicPathModel  # noqa: E402
from altermagnetism_LLG.model.network import PeriodicEquivariantFlowNet  # noqa: E402
from altermagnetism_LLG.model.sphere import sample_reference_path, sphere_exp  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--flow-checkpoint", type=Path, required=True)
    p.add_argument("--baseline-checkpoint", type=Path, required=True)
    p.add_argument("--system", default="d_wave_altermagnet")
    p.add_argument("--integration-steps", type=int, default=32)
    p.add_argument("--max-paths-per-condition", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "production_v1" / "l96_evaluation.json",
    )
    return p


def load_model(path: Path, expected_type: str, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model_type = checkpoint["model_type"]
    if model_type != expected_type:
        raise ValueError(f"{path} contains {model_type}, expected {expected_type}")
    if model_type == "flow":
        model = PeriodicEquivariantFlowNet(**checkpoint["model_config"])
    else:
        model = DeterministicPathModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def descriptors(spins: np.ndarray, time: np.ndarray) -> dict:
    local = 0.5 * (spins[:, 0, ..., 2] - spins[:, 1, ..., 2])
    order = local.mean(axis=(1, 2))
    spatial_std = local.std(axis=(1, 2))
    hits = np.flatnonzero(order < 0.0)
    transition_std = np.where(np.abs(order) < 0.5, spatial_std, 0.0).max()
    switched = bool(order[-1] < 0.0)
    return {
        "order": order,
        "endpoint": float(order[-1]),
        "crossed": bool(len(hits)),
        "switched": switched,
        "nonuniform": bool(switched and transition_std >= 0.25),
        "first_passage_ps": float(time[hits[0]] * 1e12) if len(hits) else None,
        "peak_spatial_std": float(spatial_std.max()),
        "max_spin_norm_error": float(
            np.abs(np.linalg.norm(spins, axis=-1) - 1.0).max()
        ),
    }


def summarize(rows: list[dict]) -> dict:
    orders = np.stack([row["order"] for row in rows])
    endpoints = np.asarray([row["endpoint"] for row in rows])
    passages = [
        row["first_passage_ps"]
        for row in rows
        if row["first_passage_ps"] is not None
    ]
    return {
        "paths": len(rows),
        "crossing_fraction": float(
            np.mean([row["crossed"] for row in rows])
        ),
        "switching_fraction": float(
            np.mean([row["switched"] for row in rows])
        ),
        "nonuniform_fraction": float(
            np.mean([row["nonuniform"] for row in rows])
        ),
        "endpoint_mean": float(endpoints.mean()),
        "endpoint_std": float(endpoints.std()),
        "median_first_passage_ps": (
            float(np.median(passages)) if passages else None
        ),
        "mean_peak_spatial_std": float(
            np.mean([row["peak_spatial_std"] for row in rows])
        ),
        "max_spin_norm_error": float(
            max(row["max_spin_norm_error"] for row in rows)
        ),
        "mean_neel_z_path": orders.mean(axis=0).tolist(),
        "endpoints": endpoints.tolist(),
    }


def compare(reference: dict, generated: dict) -> dict:
    reference_endpoints = np.sort(np.asarray(reference["endpoints"]))
    generated_endpoints = np.sort(np.asarray(generated["endpoints"]))
    return {
        "switching_fraction_absolute_error": abs(
            generated["switching_fraction"] - reference["switching_fraction"]
        ),
        "nonuniform_fraction_absolute_error": abs(
            generated["nonuniform_fraction"] - reference["nonuniform_fraction"]
        ),
        "endpoint_wasserstein_equal_count": float(
            np.abs(reference_endpoints - generated_endpoints).mean()
        ),
        "mean_neel_z_path_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        np.asarray(reference["mean_neel_z_path"])
                        - np.asarray(generated["mean_neel_z_path"])
                    )
                    ** 2
                )
            )
        ),
    }


@torch.no_grad()
def generate_flow(
    model,
    initial: torch.Tensor,
    scalar: torch.Tensor,
    vector: torch.Tensor,
    physical_time: torch.Tensor,
    integration_steps: int,
    seed: int,
) -> torch.Tensor:
    generator_device = (
        str(initial.device) if initial.device.type == "cuda" else "cpu"
    )
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    state = sample_reference_path(
        initial, physical_time.shape[1], generator=generator
    )
    step = 1.0 / integration_steps
    for index in range(integration_steps):
        tau = torch.full(
            (initial.shape[0],),
            index * step,
            device=initial.device,
            dtype=initial.dtype,
        )
        velocity = model(
            state, tau, initial, scalar, vector, physical_time
        )
        state = sphere_exp(state, step * velocity)
        state[:, 0] = initial
    return state


@torch.no_grad()
def main() -> None:
    args = parser().parse_args()
    if args.integration_steps < 1 or args.max_paths_per_condition < 0:
        raise ValueError("integration-steps must be positive and path limit nonnegative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; submit through a GPU allocation")

    flow, flow_checkpoint = load_model(
        args.flow_checkpoint, "flow", args.device
    )
    baseline, baseline_checkpoint = load_model(
        args.baseline_checkpoint, "deterministic", args.device
    )
    report = {
        "status": "held_out_l96_distributional_evaluation",
        "input": str(args.input.resolve()),
        "lattice_size": [96, 96],
        "integration_steps": args.integration_steps,
        "flow_best_epoch": flow_checkpoint["epoch"],
        "baseline_best_epoch": baseline_checkpoint["epoch"],
        "conditions": [],
    }

    with h5py.File(args.input, "r", swmr=True) as h5:
        group = h5[args.system]
        nx, ny = group["spins"].shape[-3:-1]
        if (nx, ny) != (96, 96):
            raise ValueError(f"expected L96 input, received {nx}x{ny}")
        time = h5["time"][:].astype(np.float32)
        physical_time = torch.from_numpy(time).unsqueeze(0).to(args.device)
        condition_ids = group["condition_id"][:]
        for condition_id in np.unique(condition_ids):
            indices = np.flatnonzero(condition_ids == condition_id)
            if args.max_paths_per_condition:
                indices = indices[: args.max_paths_per_condition]
            reference_rows = []
            flow_rows = []
            baseline_rows = []
            for local_index, trajectory in enumerate(indices):
                reference_spins = group["spins"][trajectory]
                reference_rows.append(descriptors(reference_spins, time))
                initial = torch.from_numpy(
                    group["initial_spins"][trajectory]
                ).unsqueeze(0).to(args.device)
                scalar = torch.from_numpy(
                    group["model_condition"][trajectory]
                ).unsqueeze(0).to(args.device)
                vector = torch.from_numpy(
                    group["vector_condition"][trajectory]
                ).unsqueeze(0).to(args.device)
                flow_spins = generate_flow(
                    flow,
                    initial,
                    scalar,
                    vector,
                    physical_time,
                    args.integration_steps,
                    args.seed + int(trajectory),
                )[0].cpu().numpy()
                baseline_spins = baseline(
                    initial,
                    len(time),
                    scalar,
                    vector,
                    physical_time,
                )[0].cpu().numpy()
                flow_rows.append(descriptors(flow_spins, time))
                baseline_rows.append(descriptors(baseline_spins, time))
                print(
                    f"condition={int(condition_id)} "
                    f"path={local_index + 1}/{len(indices)}",
                    flush=True,
                )

            reference_summary = summarize(reference_rows)
            flow_summary = summarize(flow_rows)
            baseline_summary = summarize(baseline_rows)
            drive = float(group["physical_condition"][indices[0], 2])
            report["conditions"].append(
                {
                    "condition_id": int(condition_id),
                    "drive_T": drive,
                    "reference": reference_summary,
                    "flow": flow_summary,
                    "deterministic": baseline_summary,
                    "flow_errors": compare(reference_summary, flow_summary),
                    "deterministic_errors": compare(
                        reference_summary, baseline_summary
                    ),
                }
            )

    for model_name in ("flow", "deterministic"):
        comparisons = [
            condition[
                "flow_errors" if model_name == "flow" else "deterministic_errors"
            ]
            for condition in report["conditions"]
        ]
        report[f"{model_name}_aggregate"] = {
            key: float(np.mean([item[key] for item in comparisons]))
            for key in comparisons[0]
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
