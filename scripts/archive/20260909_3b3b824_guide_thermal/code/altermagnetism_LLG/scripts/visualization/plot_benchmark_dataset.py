"""Create QA and physics-summary PNGs for the unified LLG benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
COLORS = {"ordinary": "#4c78a8", "afm": "#f28e2b", "am": "#7a5195"}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot the unified LLG benchmark")
    p.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "training_benchmark" / "llg_spatial_paths.h5",
    )
    p.add_argument(
        "--metrics",
        type=Path,
        default=ROOT / "data" / "training_benchmark" / "benchmark_certification.json",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "assets" / "training_benchmark",
    )
    return p


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
        }
    )


def plot_certification(metrics: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)
    systems = list(metrics["systems"])
    short = ["AFM control", "d-wave AM", "ordinary moments"]
    totals = []
    split_matrix = []
    for system in systems:
        per_condition = metrics["systems"][system]["conditions"]
        counts = np.sum(
            [c["split_counts_train_validation_test"] for c in per_condition], axis=0
        )
        split_matrix.append(counts)
        totals.append(sum(counts))
    split_matrix = np.asarray(split_matrix)
    bottom = np.zeros(len(systems))
    for j, (label, color) in enumerate(
        zip(("train", "validation", "test"), ("#4c78a8", "#9ecae9", "#f28e2b"))
    ):
        axes[0].bar(short, split_matrix[:, j], bottom=bottom, color=color, label=label)
        bottom += split_matrix[:, j]
    axes[0].set_ylabel("whole trajectories")
    axes[0].set_title("a  exact 8:1:1 split in every condition")
    axes[0].legend(ncol=3, fontsize=8, loc="upper center")
    axes[0].tick_params(axis="x", rotation=18)

    ordinary = metrics["systems"]["ordinary_free_moments"]["conditions"]
    temperature = np.array([c["temperature"] for c in ordinary])
    measured = np.array([c["measured_mean_mz"] for c in ordinary])
    sem = np.array([c["trajectory_sem"] for c in ordinary])
    dense_t = np.linspace(0.55, 5.5, 300)
    x = 2.0 / dense_t
    exact_dense = 1.0 / np.tanh(x) - 1.0 / x
    axes[1].plot(dense_t, exact_dense, color="black", lw=1.4, label="exact Langevin")
    axes[1].errorbar(
        temperature,
        measured,
        yerr=1.96 * sem,
        fmt="o",
        color=COLORS["ordinary"],
        capsize=3,
        label="sLLG, 95% trajectory CI",
    )
    axes[1].set_xlabel("temperature (paper units)")
    axes[1].set_ylabel(r"$\langle M_z\rangle$")
    axes[1].set_title("b  ordinary-system equilibrium certificate")
    axes[1].legend(fontsize=8)

    labels = ["finite", "unit norm", "8:1:1", "Langevin", "spin wave"]
    checks = metrics["checks"]
    passed = [
        checks["all_spins_finite"],
        checks["max_norm_error_below_1e-6"],
        checks["split_is_exact_8_1_1_per_condition"],
        checks["nishino_langevin_max_abs_error_below_0p03"],
        checks["gomonay_spinwave_existing_validation_pass"],
    ]
    axes[2].barh(labels, np.ones(len(labels)), color=["#59a14f" if x else "#e15759" for x in passed])
    for i, ok in enumerate(passed):
        axes[2].text(0.5, i, "PASS" if ok else "FAIL", ha="center", va="center", color="white", weight="bold")
    axes[2].set_xlim(0, 1)
    axes[2].set_xticks([])
    axes[2].set_title("c  benchmark gates")
    fig.suptitle("Unified LLG benchmark: data integrity and two literature anchors", weight="bold")
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_path_ensembles(h5: h5py.File, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.0), sharex=True, sharey=True, constrained_layout=True)
    rows = (("conventional_afm_control", "conventional AFM control"), ("d_wave_altermagnet", "d-wave altermagnet"))
    for row, (name, row_label) in enumerate(rows):
        group = h5[name]
        time_ps = group["time"][:] * 1e12
        for condition_id in range(3):
            ax = axes[row, condition_id]
            mask = group["condition_id"][:] == condition_id
            paths = group["neel"][:][mask, :, 2]
            drive = group["condition"][:][mask][0, 2]
            for path in paths:
                color = "#d95f5f" if path[-1] < 0 else "#8c96a3"
                ax.plot(time_ps, path, color=color, lw=0.65, alpha=0.45)
            ax.plot(time_ps, paths.mean(axis=0), color="black", lw=1.6, label="ensemble mean")
            ax.axhline(0, color="0.75", lw=0.7)
            ax.axvline(0.5, color="#4c78a8", ls="--", lw=0.9)
            ax.set_title(f"{row_label}\n$H_{{DL}}$ = {drive:.1f} T")
            if row == 1:
                ax.set_xlabel("time (ps)")
            if condition_id == 0:
                ax.set_ylabel(r"spatial mean $\bar n_z$")
    axes[0, 2].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Every finite-temperature trajectory is retained | red: negative final basin | dashed: pulse end",
        weight="bold",
    )
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def incoherence(spins: np.ndarray) -> np.ndarray:
    n = 0.5 * (spins[:, :, 0] - spins[:, :, 1])
    return 1.0 - np.linalg.norm(n.mean(axis=(2, 3)), axis=-1) / np.clip(
        np.linalg.norm(n, axis=-1).mean(axis=(2, 3)), 1e-12, None
    )


def plot_endpoints(h5: h5py.File, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), constrained_layout=True)
    rng = np.random.default_rng(7)
    for name, label, color, offset in (
        ("conventional_afm_control", "AFM control", COLORS["afm"], -0.025),
        ("d_wave_altermagnet", "d-wave AM", COLORS["am"], 0.025),
    ):
        group = h5[name]
        drives, means, errors = [], [], []
        for condition_id in range(3):
            mask = group["condition_id"][:] == condition_id
            drive = float(group["condition"][:][mask][0, 2])
            endpoint = group["neel"][:][mask, -1, 2]
            jitter = rng.normal(0, 0.006, len(endpoint)) + drive + offset
            axes[0].scatter(jitter, endpoint, s=14, color=color, alpha=0.58)
            spin = group["spins"][:][mask]
            peak = incoherence(spin).max(axis=1)
            drives.append(drive + offset)
            means.append(peak.mean())
            errors.append(1.96 * peak.std(ddof=1) / np.sqrt(len(peak)))
        axes[1].errorbar(drives, means, yerr=errors, marker="o", capsize=3, color=color, label=label)
    axes[0].axhline(0, color="0.75", lw=0.7)
    axes[0].set_xlabel(r"$H_{DL}$ (T)")
    axes[0].set_ylabel(r"final $\bar n_z$")
    axes[0].set_title("a  endpoint distribution, not only its mean")
    axes[1].set_xlabel(r"$H_{DL}$ (T)")
    axes[1].set_ylabel("peak spatial incoherence")
    axes[1].set_title("b  spatial nonuniformity remains small at 8 x 8")
    axes[1].legend(fontsize=8)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_examples(h5: h5py.File, output: Path) -> None:
    rows = (
        ("ordinary_free_moments", r"ordinary: $S_z$", 1),
        ("conventional_afm_control", r"AFM control: local $n_z$", 1),
        ("d_wave_altermagnet", r"d-wave AM: local $n_z$", 1),
    )
    fig, axes = plt.subplots(3, 3, figsize=(8.2, 7.5), constrained_layout=True)
    image = None
    for row, (name, label, condition_id) in enumerate(rows):
        group = h5[name]
        mask_ids = np.flatnonzero(group["condition_id"][:] == condition_id)
        if name == "ordinary_free_moments":
            endpoint = group["magnetization"][mask_ids, -1, 2]
        else:
            endpoint = group["neel"][mask_ids, -1, 2]
        chosen = mask_ids[np.argmin(np.abs(endpoint - np.median(endpoint)))]
        spins = group["spins"][chosen]
        if name == "ordinary_free_moments":
            field = spins[:, 0, ..., 2]
        else:
            field = 0.5 * (spins[:, 0, ..., 2] - spins[:, 1, ..., 2])
        for col, (frame, when) in enumerate(((0, "initial"), (len(field) // 2, "middle"), (-1, "final"))):
            image = axes[row, col].imshow(
                field[frame].T,
                origin="lower",
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
                interpolation="nearest",
            )
            axes[row, col].set_title(when)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(label)
    fig.colorbar(image, ax=axes, shrink=0.78, label="local longitudinal component")
    fig.suptitle(
        "Full spatial states | deterministic representative = trajectory nearest median endpoint",
        weight="bold",
    )
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    plot_certification(metrics, args.output_dir / "benchmark_certification.png")
    with h5py.File(args.input, "r") as h5:
        plot_path_ensembles(h5, args.output_dir / "switching_path_ensembles.png")
        plot_endpoints(h5, args.output_dir / "endpoint_spatial_statistics.png")
        plot_spatial_examples(h5, args.output_dir / "spatial_trajectory_examples.png")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
