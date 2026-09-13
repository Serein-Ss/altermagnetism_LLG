"""Model-evaluation plots and animations, separated from metrics and sampling."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from ..core.project_paths import asset_path
from ..analysis.evaluation import BasinCalibration, neel_observables

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
    figure.savefig(asset_path(output), dpi=180)
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
    figure.savefig(asset_path(output), dpi=180)
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
    animation.save(asset_path(output), writer=PillowWriter(fps=10), dpi=100)
    plt.close(figure)


def _plot(report: dict, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    time = np.asarray(report["time_ps"])
    for steps, paths in report["integration_convergence"]["neel_z_paths"].items():
        axes[0].plot(
            time, np.mean(paths, axis=0), label=f"{steps} steps"
        )
    axes[0].set_title("flow integration convergence")
    axes[0].set_xlabel("time (ps)")
    axes[0].set_ylabel(r"$n_z$")
    axes[0].legend(fontsize=8)

    sensitivity = report["condition_sensitivity"]
    for drive, path in zip(sensitivity["drives_T"], sensitivity["neel_z_paths"]):
        axes[1].plot(time, path, label=f"{drive:.2f} T")
    axes[1].set_title("fixed initial + latent seed")
    axes[1].set_xlabel("time (ps)")
    axes[1].set_ylabel(r"$n_z$")
    axes[1].legend(fontsize=8)

    diversity = np.asarray(
        report["same_initial_diversity"]["neel_z_paths"]
    )
    axes[2].plot(time, diversity.T, alpha=0.35, linewidth=0.8)
    axes[2].plot(time, diversity.mean(axis=0), color="black", linewidth=2)
    axes[2].set_title("fixed initial + condition, 16 latents")
    axes[2].set_xlabel("time (ps)")
    axes[2].set_ylabel(r"$n_z$")
    figure.tight_layout()
    figure.savefig(asset_path(output), dpi=180)
    plt.close(figure)


