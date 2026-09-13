"""Plot stable-helix and literature-chain path validation. PNG only."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from altermagnetism_LLG.scripts.core.project_paths import asset_path, resolve_recorded_path


ROOT = Path(__file__).resolve().parents[2]
HELIX_DIR = ROOT / "data" / "literature" / "noncollinear_validation" / "crnb3s6"
CHAIN_FILE = ROOT / "data" / "literature" / "path_literature_validation" / "bauer2011_accelerated_pilot_paths.h5"
MECHANISM_FILE = CHAIN_FILE.with_name(CHAIN_FILE.stem + "_mechanisms.npz")
OUTPUT = ROOT / "assets/literature_reproduction/cross_paper_legacy/legacy_manual/figures/literature_path_validation.png"


def representative(labels: np.ndarray, name: str) -> int:
    indices = np.flatnonzero(labels == name)
    if not len(indices):
        raise RuntimeError(f"no {name} path is available for plotting")
    return int(indices[0])


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    helix = np.load(resolve_recorded_path(HELIX_DIR / "helix_trajectory.npz"))
    helix_metrics = json.loads(resolve_recorded_path(HELIX_DIR / "validation.json").read_text())
    final_spin = helix["spins"][-1, 0, :, 0]
    position = np.arange(len(final_spin)) * 0.6

    mechanisms = np.load(resolve_recorded_path(MECHANISM_FILE))
    labels = mechanisms["bauer_chain_outcome"]
    success = representative(labels, "negative_endpoint")
    returned = representative(labels, "crossing_return")
    with h5py.File(resolve_recorded_path(CHAIN_FILE), "r") as h5:
        chain_time = h5["time"][:]
        local_z = h5["spins"][:, :, 0, :, 0, 2]

    fig = plt.figure(figsize=(11.0, 6.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(0.9, 1.1))

    ax = fig.add_subplot(grid[0, :2])
    colors = ("#3b75af", "#d65f5f", "#6a6a6a")
    for component, label, color in zip(final_spin.T, ("m_x", "m_y", "m_z"), colors):
        ax.plot(position, component, lw=1.4, label=label, color=color)
    ax.set(
        xlabel="Position along chiral axis (nm)",
        ylabel="Spin component",
        title="a  Published CrNb$_3$S$_6$ parameters stabilize a 48 nm helix",
        xlim=(position[0], position[-1]),
        ylim=(-1.12, 1.12),
    )
    ax.legend(ncol=3, loc="upper right")
    ax.text(
        0.01,
        0.04,
        "parameter prediction: 48.46 nm; max torque: 6.8e-14 T",
        transform=ax.transAxes,
        fontsize=7.5,
    )

    ax = fig.add_subplot(grid[0, 2])
    energy = helix_metrics["winding_sector_energies_J"]
    windings = np.array([0, 1, 2])
    values = np.array([energy[str(i)] for i in windings]) / 1e-22
    ax.bar(windings, values, color=("#a9b2bd", "#4c956c", "#a9b2bd"), width=0.7)
    ax.axhline(0, color="black", lw=0.7)
    ax.set(
        xlabel="Winding number",
        ylabel="Energy ($10^{-22}$ J)",
        title="b  One-turn sector is favored",
        xticks=windings,
    )

    def heatmap(axis, trajectory, title):
        image = axis.imshow(
            local_z[trajectory].T,
            aspect="auto",
            origin="lower",
            extent=(chain_time[0], chain_time[-1], 0, local_z.shape[2] - 1),
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        axis.set(xlabel=r"Time ($\hbar/J$)", ylabel="Chain site", title=title)
        return image

    ax_success = fig.add_subplot(grid[1, 0])
    image = heatmap(ax_success, success, "c  Negative-endpoint path")
    ax_return = fig.add_subplot(grid[1, 1])
    heatmap(ax_return, returned, "d  Crossing-and-return path")
    colorbar = fig.colorbar(image, ax=(ax_success, ax_return), location="bottom", shrink=0.75, pad=0.13)
    colorbar.set_label("Local $m_z$")

    ax = fig.add_subplot(grid[1, 2])
    names = ("no_crossing", "crossing_return", "negative_endpoint")
    display = ("No crossing", "Cross + return", "Negative endpoint")
    counts = np.array([(labels == name).sum() for name in names])
    bars = ax.bar(
        np.arange(3), counts / len(labels), color=("#a9b2bd", "#e0a458", "#4c956c")
    )
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, str(count), ha="center")
    ax.set(
        ylabel="Fraction of all paths",
        title="e  Same-condition outcomes (n=40)",
        xticks=np.arange(3),
        xticklabels=display,
        ylim=(0, 0.85),
    )
    ax.tick_params(axis="x", rotation=18)
    ax.text(
        0.02,
        0.98,
        "Accelerated pilot: L=40, kBT/J=0.20\nAll paths retained; not Fig. 2 replication",
        transform=ax.transAxes,
        va="top",
        fontsize=7.2,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(asset_path(OUTPUT), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
