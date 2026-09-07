"""Evaluate full test trajectories at any lattice size with LLG-fitted basins."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time as walltime

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from altermagnetism_LLG.model.baseline import DeterministicPathModel  # noqa: E402
from altermagnetism_LLG.model.evaluation import (  # noqa: E402
    BasinCalibration,
    calibrate_basins,
    compare,
    generate_flow,
    neel_observables,
    path_descriptors,
    summarize,
)
from altermagnetism_LLG.model.network import PeriodicEquivariantFlowNet  # noqa: E402


SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--calibration-input", type=Path, nargs="+", required=True)
    p.add_argument("--flow-checkpoint", type=Path, required=True)
    p.add_argument("--baseline-checkpoint", type=Path, default=None)
    p.add_argument("--system", default="d_wave_altermagnet")
    p.add_argument("--split", choices=tuple(SPLIT_CODES), default="test")
    p.add_argument("--integration-steps", type=int, default=32)
    p.add_argument("--max-paths-per-condition", type=int, default=0)
    p.add_argument("--residence-ps", type=float, default=0.10)
    p.add_argument("--unresolved-limit", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--skip-animations", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def load_model(path: Path, expected_type: str, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["model_type"] != expected_type:
        raise ValueError(
            f"{path} contains {checkpoint['model_type']}, expected {expected_type}"
        )
    if expected_type == "flow":
        model = PeriodicEquivariantFlowNet(**checkpoint["model_config"])
    else:
        model = DeterministicPathModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def _json_row(row: dict) -> dict:
    return {
        key: value
        for key, value in row.items()
        if key not in ("order", "spatial_std")
    }


def _plot_histograms(
    condition_rows: list[dict],
    output: Path,
    calibration: BasinCalibration | None = None,
) -> None:
    figure, axes = plt.subplots(
        1, len(condition_rows),
        figsize=(5 * len(condition_rows), 4),
        squeeze=False,
    )
    bins = np.linspace(-1.0, 1.0, 31)
    styles = (
        ("reference", "black", "-"),
        ("flow", "tab:blue", "--"),
        ("deterministic", "tab:orange", ":"),
    )
    for axis, condition in zip(axes[0], condition_rows):
        for name, color, linestyle in styles:
            if name in condition:
                axis.hist(
                    condition[name]["endpoints"],
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=1.8,
                    label=name,
                    color=color,
                    linestyle=linestyle,
                )
        if calibration is not None:
            axis.axvline(
                calibration.negative_threshold, color="0.4",
                linestyle="-.", linewidth=1.0, label="basin thresholds",
            )
            axis.axvline(
                calibration.positive_threshold, color="0.4",
                linestyle="-.", linewidth=1.0,
            )
        axis.set_title(
            f"L={condition['lattice_size']}, H={condition['drive_T']:.2f} T; "
            f"n={condition['reference']['paths']} trajectories/model"
        )
        axis.set_xlabel(r"endpoint $n_z$")
        axis.set_ylabel("density")
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_paths(
    condition_rows: list[dict],
    time: np.ndarray,
    output: Path,
    key: str,
    ylabel: str,
    calibration: BasinCalibration | None = None,
) -> None:
    figure, axes = plt.subplots(
        len(condition_rows), 1,
        figsize=(8, 3 * len(condition_rows)),
        squeeze=False,
    )
    time_ps = time * 1e12
    styles = (
        ("reference", "black", "-"),
        ("flow", "tab:blue", "--"),
        ("deterministic", "tab:orange", ":"),
    )
    for axis, condition in zip(axes[:, 0], condition_rows):
        for name, color, linestyle in styles:
            if name in condition:
                paths_key = (
                    "neel_z_paths"
                    if key == "mean_neel_z_path"
                    else "spatial_std_paths"
                )
                paths = np.asarray(condition[name][paths_key])
                axis.plot(
                    time_ps, paths.T, color=color,
                    linewidth=0.6, alpha=0.16,
                )
                axis.plot(
                    time_ps, condition[name][key],
                    color=color, linestyle=linestyle,
                    linewidth=2.2, label=f"{name} mean",
                )
        if calibration is not None and key == "mean_neel_z_path":
            axis.axhline(
                calibration.negative_threshold, color="0.4",
                linestyle="-.", linewidth=1.0,
            )
            axis.axhline(
                calibration.positive_threshold, color="0.4",
                linestyle="-.", linewidth=1.0,
            )
        axis.set_title(
            f"L={condition['lattice_size']}, H={condition['drive_T']:.2f} T; "
            f"n={condition['reference']['paths']} trajectories/model"
        )
        axis.set_xlabel("time (ps)")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _animate_pair(
    reference: np.ndarray,
    generated: np.ndarray,
    time: np.ndarray,
    output: Path,
) -> None:
    ref_order, _ = neel_observables(reference)
    gen_order, _ = neel_observables(generated)
    ref_local = 0.5 * (
        reference[:, 0, ..., 2] - reference[:, 1, ..., 2]
    )
    gen_local = 0.5 * (
        generated[:, 0, ..., 2] - generated[:, 1, ..., 2]
    )
    frames = np.unique(
        np.linspace(0, len(time) - 1, min(34, len(time))).astype(int)
    )
    figure, axes = plt.subplots(2, 2, figsize=(9, 7))
    ref_image = axes[0, 0].imshow(
        ref_local[0], vmin=-1, vmax=1, cmap="coolwarm"
    )
    gen_image = axes[0, 1].imshow(
        gen_local[0], vmin=-1, vmax=1, cmap="coolwarm"
    )
    axes[0, 0].set_title("real LLG local Neel-z")
    axes[0, 1].set_title("flow local Neel-z")
    for axis in axes[0]:
        axis.set_xticks([])
        axis.set_yticks([])
    axes[1, 0].plot(time * 1e12, ref_order, color="black")
    axes[1, 1].plot(time * 1e12, gen_order, color="tab:blue")
    ref_marker = axes[1, 0].axvline(0.0, color="tab:red")
    gen_marker = axes[1, 1].axvline(0.0, color="tab:red")
    for axis in axes[1]:
        axis.set_ylim(-1.05, 1.05)
        axis.set_xlabel("time (ps)")
        axis.set_ylabel(r"$n_z$")
    figure.colorbar(
        ref_image, ax=axes[0].tolist(), label="local Neel-z"
    )

    def update(frame: int):
        ref_image.set_data(ref_local[frame])
        gen_image.set_data(gen_local[frame])
        position = time[frame] * 1e12
        ref_marker.set_xdata([position, position])
        gen_marker.set_xdata([position, position])
        figure.suptitle(f"t={position:.3f} ps")
        return ref_image, gen_image, ref_marker, gen_marker

    animation = FuncAnimation(
        figure, update, frames=frames, interval=100, blit=False
    )
    animation.save(output, writer=PillowWriter(fps=10), dpi=100)
    plt.close(figure)


@torch.no_grad()
def main() -> None:
    args = parser().parse_args()
    if args.integration_steps < 1 or args.max_paths_per_condition < 0:
        raise ValueError(
            "integration steps must be positive and path limit nonnegative"
        )
    if args.residence_ps <= 0 or not 0 <= args.unresolved_limit <= 1:
        raise ValueError(
            "residence-ps must be positive and unresolved-limit in [0,1]"
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

    calibration = calibrate_basins(
        args.calibration_input,
        args.system,
        residence_ps=args.residence_ps,
        seed=args.seed,
    )
    flow, flow_checkpoint = load_model(
        args.flow_checkpoint, "flow", args.device
    )
    baseline = baseline_checkpoint = None
    if args.baseline_checkpoint is not None:
        baseline, baseline_checkpoint = load_model(
            args.baseline_checkpoint, "deterministic", args.device
        )

    with h5py.File(args.input, "r", swmr=True) as source:
        group = source[args.system]
        time = source["time"][:].astype(np.float32)
        nx, ny = group["spins"].shape[-3:-1]
        selected = np.flatnonzero(
            group["split"][:] == SPLIT_CODES[args.split]
        )
        if not len(selected):
            raise ValueError(
                f"input contains no {args.split} trajectories"
            )
        condition_ids = group["condition_id"][:]
        selected_by_condition = []
        for condition_id in np.unique(condition_ids[selected]):
            indices = selected[condition_ids[selected] == condition_id]
            if args.max_paths_per_condition:
                indices = indices[: args.max_paths_per_condition]
            selected_by_condition.append((int(condition_id), indices))
        flat_indices = np.concatenate(
            [indices for _, indices in selected_by_condition]
        )
        count = len(flat_indices)
        output_h5 = args.output_dir / "generated_trajectories.h5"
        if output_h5.exists():
            output_h5.unlink()
        condition_reports = []
        animation_inputs = {}
        with h5py.File(output_h5, "w") as generated_h5:
            generated_h5.create_dataset("time", data=time)
            generated_h5.create_dataset("source_index", data=flat_indices)
            generated_h5.create_dataset(
                "condition_id", data=condition_ids[flat_indices]
            )
            generated_h5.attrs["source"] = str(args.input.resolve())
            generated_h5.attrs["basin_calibration_json"] = json.dumps(
                calibration.to_dict()
            )
            shape = (count,) + group["spins"].shape[1:]
            chunks = (
                1, min(8, shape[1]), shape[2],
                min(16, nx), min(16, ny), 3,
            )
            flow_data = generated_h5.create_group("flow").create_dataset(
                "spins", shape=shape, dtype="f4", chunks=chunks,
                compression="gzip", compression_opts=4, shuffle=True,
            )
            baseline_data = None
            if baseline is not None:
                baseline_data = generated_h5.create_group(
                    "deterministic"
                ).create_dataset(
                    "spins", shape=shape, dtype="f4", chunks=chunks,
                    compression="gzip", compression_opts=4, shuffle=True,
                )
            output_index = 0
            physical_time = torch.from_numpy(time).unsqueeze(0).to(args.device)
            for condition_id, indices in selected_by_condition:
                reference_rows = []
                flow_rows = []
                baseline_rows = []
                path_reports = []
                flow_seconds = 0.0
                baseline_seconds = 0.0
                for local_index, trajectory in enumerate(indices):
                    reference_spins = group["spins"][trajectory]
                    initial = torch.from_numpy(
                        group["initial_spins"][trajectory]
                    ).unsqueeze(0).to(args.device)
                    scalar = torch.from_numpy(
                        group["model_condition"][trajectory]
                    ).unsqueeze(0).to(args.device)
                    vector = torch.from_numpy(
                        group["vector_condition"][trajectory]
                    ).unsqueeze(0).to(args.device)
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    started = walltime.perf_counter()
                    flow_spins = generate_flow(
                        flow, initial, scalar, vector, physical_time,
                        args.integration_steps,
                        args.seed + int(trajectory),
                    )[0].cpu().numpy().astype(np.float32)
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    flow_seconds += walltime.perf_counter() - started
                    flow_data[output_index] = flow_spins
                    reference_row = path_descriptors(
                        reference_spins, time, calibration
                    )
                    flow_row = path_descriptors(
                        flow_spins, time, calibration
                    )
                    reference_rows.append(reference_row)
                    flow_rows.append(flow_row)
                    item = {
                        "source_index": int(trajectory),
                        "trajectory_id": int(
                            group["trajectory_id"][trajectory]
                        ),
                        "latent_seed": args.seed + int(trajectory),
                        "reference": _json_row(reference_row),
                        "flow": _json_row(flow_row),
                    }
                    if baseline is not None:
                        if args.device.startswith("cuda"):
                            torch.cuda.synchronize()
                        started = walltime.perf_counter()
                        baseline_spins = baseline(
                            initial, len(time), scalar, vector, physical_time
                        )[0].cpu().numpy().astype(np.float32)
                        if args.device.startswith("cuda"):
                            torch.cuda.synchronize()
                        baseline_seconds += walltime.perf_counter() - started
                        baseline_data[output_index] = baseline_spins
                        baseline_row = path_descriptors(
                            baseline_spins, time, calibration
                        )
                        baseline_rows.append(baseline_row)
                        item["deterministic"] = _json_row(
                            baseline_row
                        )
                    if condition_id not in animation_inputs:
                        animation_inputs[condition_id] = (
                            reference_spins.copy(), flow_spins.copy()
                        )
                    path_reports.append(item)
                    output_index += 1
                    print(
                        f"L={nx} condition={condition_id} "
                        f"path={local_index + 1}/{len(indices)}",
                        flush=True,
                    )
                reference_summary = summarize(reference_rows)
                flow_summary = summarize(flow_rows)
                drive = float(
                    group["physical_condition"][indices[0], 2]
                )
                condition_report = {
                    "condition_id": condition_id,
                    "drive_T": drive,
                    "lattice_size": nx,
                    "reference": reference_summary,
                    "flow": flow_summary,
                    "flow_errors": compare(
                        reference_summary, flow_summary
                    ),
                    "generation_runtime_seconds": {
                        "flow_total": flow_seconds,
                        "flow_per_path": flow_seconds / len(indices),
                        "deterministic_total": baseline_seconds,
                        "deterministic_per_path": (
                            baseline_seconds / len(indices)
                            if baseline is not None else None
                        ),
                    },
                    "paths": path_reports,
                }
                if baseline_rows:
                    baseline_summary = summarize(baseline_rows)
                    condition_report["deterministic"] = baseline_summary
                    condition_report["deterministic_errors"] = compare(
                        reference_summary, baseline_summary
                    )
                condition_reports.append(condition_report)

    insufficient = [
        row["condition_id"]
        for row in condition_reports
        if row["reference"]["unresolved_transition_fraction"]
        > args.unresolved_limit
    ]
    report = {
        "status": (
            "observation_window_insufficient"
            if insufficient
            else "evaluation_complete"
        ),
        "input": str(args.input.resolve()),
        "split": args.split,
        "lattice_size": [nx, ny],
        "integration_steps": args.integration_steps,
        "generation_seed_base": args.seed,
        "flow_best_epoch": flow_checkpoint["epoch"],
        "baseline_best_epoch": (
            baseline_checkpoint["epoch"]
            if baseline_checkpoint else None
        ),
        "basin_calibration": calibration.to_dict(),
        "observation_window_assessment": {
            "unresolved_fraction_limit": args.unresolved_limit,
            "insufficient_condition_ids": insufficient,
            "action": (
                "extend real LLG observation before assigning completion probabilities"
                if insufficient
                else "current observation window passes the unresolved-fraction gate"
            ),
        },
        "generated_trajectories": str(output_h5.resolve()),
        "conditions": condition_reports,
    }
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _plot_histograms(
        condition_reports, args.output_dir / "endpoint_histograms.png",
        calibration,
    )
    _plot_paths(
        condition_reports, time,
        args.output_dir / "neel_z_paths.png",
        "mean_neel_z_path", r"$n_z$",
        calibration,
    )
    _plot_paths(
        condition_reports, time,
        args.output_dir / "spatial_std_paths.png",
        "mean_spatial_std_path", r"spatial std of local $n_z$",
    )
    if not args.skip_animations:
        for condition_id, pair in animation_inputs.items():
            _animate_pair(
                pair[0], pair[1], time,
                args.output_dir
                / f"condition_{condition_id}_reference_vs_flow.gif",
            )
    print(json.dumps({
        "status": report["status"],
        "lattice_size": report["lattice_size"],
        "output": str(args.output_dir.resolve()),
        "insufficient_condition_ids": insufficient,
    }, indent=2))


if __name__ == "__main__":
    main()
