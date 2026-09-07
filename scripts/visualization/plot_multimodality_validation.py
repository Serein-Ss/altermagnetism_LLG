"""Visualize path bifurcation and timestep replication. PNG output only."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PRIMARY = ROOT / "data" / "multimodality_validation"
REPLICATION = ROOT / "data" / "multimodality_validation_dt0p05"
AFM_CONTROL = ROOT / "data" / "multimodality_afm_control"
OUTPUT = ROOT / "assets" / "multimodality_validation" / "path_multimodality.png"

COLORS = ["#3978A8", "#E58A33", "#7851A9"]
LABELS = ["No crossing", "Cross and return", "Negative at 1 ps"]


def load(folder: Path):
    data = np.load(folder / "multimodality_ensemble.npz")
    metrics = json.loads((folder / "multimodality_metrics.json").read_text())
    return data, metrics


def main() -> None:
    data, metrics = load(PRIMARY)
    replica, replica_metrics = load(REPLICATION)
    control, control_metrics = load(AFM_CONTROL)
    time = data["time_ps"]
    nz = data["neel_mean"][..., 2]
    classes = data["physical_class"]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), constrained_layout=True)

    ax = axes[0, 0]
    for cls, color in enumerate(COLORS):
        subset = nz[classes == cls]
        ax.plot(time, subset.T, color=color, alpha=0.075, linewidth=0.65)
    ax.plot(time, nz.mean(axis=0), color="black", linewidth=2.0,
            label="Ensemble mean")
    ax.axvspan(0, metrics["condition"]["pulse_ps"], color="#BBBBBB",
               alpha=0.16, label="SOT on")
    ax.axhline(0, color="#555555", linestyle=":", linewidth=0.8)
    ax.set(title="a  Equal-weight trajectories at one condition",
           xlabel="Time (ps)", ylabel=r"Mean Neel component $n_z$")
    ax.legend(loc="lower left")

    ax = axes[0, 1]
    for cls, (color, label) in enumerate(zip(COLORS, LABELS)):
        subset = nz[classes == cls]
        lo, mean, hi = np.quantile(subset, 0.1, axis=0), subset.mean(0), np.quantile(subset, 0.9, axis=0)
        ax.fill_between(time, lo, hi, color=color, alpha=0.18)
        ax.plot(time, mean, color=color, linewidth=1.8,
                label=f"{label} (n={len(subset)})")
    ax.plot(time, nz.mean(axis=0), color="black", linestyle="--", linewidth=1.6,
            label="All-path mean")
    ax.axhline(0, color="#555555", linestyle=":", linewidth=0.8)
    ax.set(title="b  Physically distinct path families",
           xlabel="Time (ps)", ylabel=r"Mean $n_z$")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    bins = np.linspace(-1, 1, 30)
    for cls, (color, label) in enumerate(zip(COLORS, LABELS)):
        ax.hist(nz[classes == cls, -1], bins=bins, color=color, alpha=0.58,
                label=label)
    ax.axvline(nz[:, -1].mean(), color="black", linestyle="--", linewidth=1.5,
               label=f"Mean = {nz[:, -1].mean():.2f}")
    ax.set(title="c  Endpoint distribution hidden by its mean",
           xlabel=r"Final $n_z$ at 1 ps", ylabel="Trajectory count")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    count_keys = ["no_zero_crossing", "cross_and_return_positive", "negative_at_final_time"]
    values = []
    for result in (metrics, replica_metrics, control_metrics):
        counts = result["physical_path_classes"]
        total = result["condition"]["trajectories"]
        values.append([counts[key]/total for key in count_keys])
    values = np.asarray(values)
    x = np.arange(3)
    width = 0.25
    ax.bar(x-width, values[0], width, color="#517CA3", label="AM: dt = 0.10 fs, n=256")
    ax.bar(x, values[1], width, color="#A7BED3", edgecolor="#517CA3",
           label="AM: dt = 0.05 fs, n=128")
    ax.bar(x+width, values[2], width, color="#B9B9B9", edgecolor="#555555",
           label=r"AFM control: $\widetilde{J}=0$, n=128")
    ax.set_xticks(x, ["No crossing", "Return", "Negative final"], rotation=12)
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("Fraction of all paths")
    ax.set_title("d  Numerical replication and AFM control")
    ax.legend(fontsize=7)
    ax.text(0.98, 0.96,
            f"Path-PCA GMM ΔBIC: {metrics['bic_gain_over_one_component']:.0f}\n"
            f"Replication ΔBIC: {replica_metrics['bic_gain_over_one_component']:.0f}\n"
            f"AFM control ΔBIC: {control_metrics['bic_gain_over_one_component']:.0f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7)

    for ax in axes.flat:
        ax.grid(alpha=0.16)
    fig.suptitle(
        "Finite-temperature path multimodality in the literature-anchored altermagnet",
        fontsize=12, fontweight="bold"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=240, facecolor="white")
    plt.close(fig)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
