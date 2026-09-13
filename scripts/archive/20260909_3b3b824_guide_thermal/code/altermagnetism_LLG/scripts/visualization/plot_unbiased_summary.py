"""Plot a compact diagnostic for an unbiased stochastic-LLG ensemble."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=ROOT / "data" / "unbiased")
    p.add_argument("--output", type=Path,
                   default=ROOT / "assets" / "unbiased" / "ensemble_summary.png")
    args = p.parse_args()
    paths = sorted(args.input_dir.glob("trajectory_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no trajectory files in {args.input_dir}")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    final = []
    for path in paths:
        data = np.load(path, allow_pickle=False)
        t = data["time_s"] * 1e12
        nz = data["neel_mean"][:, 2]
        axes[0].plot(t, nz, alpha=0.45, linewidth=0.8)
        final.append(data["neel_mean"][-1])
    final = np.asarray(final)
    axes[0].set(xlabel="Time (ps)", ylabel=r"Mean $n_z$",
                title="All paths (no outcome selection)")
    axes[1].scatter(final[:, 0], final[:, 1], c=final[:, 2], cmap="coolwarm",
                    vmin=-1, vmax=1, s=24, edgecolor="none")
    axes[1].set(xlabel=r"Final $n_x$", ylabel=r"Final $n_y$",
                title=r"Final states; color is $n_z$")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.18)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, facecolor="white")
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

