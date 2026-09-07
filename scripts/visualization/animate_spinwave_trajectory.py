"""Animate the published-mode LLG validation trajectory without rerunning LLG."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Animate the spatial LLG spin wave")
    p.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "literature_validation" / "spinwave_trajectory.npz",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "assets" / "literature_validation" / "trajectory_animation",
    )
    p.add_argument("--fps", type=int, default=20)
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


def save_initial_final(snapshots: np.ndarray, output: Path) -> None:
    transverse = 1e3 * snapshots[:, 0, ..., :2]
    limit = float(np.max(np.abs(transverse)))
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.5), constrained_layout=True)
    for col, (frame, label) in enumerate(((0, "initial, 0 ps"), (-1, "final, 4 ps"))):
        for row, (component, name) in enumerate(((0, r"$10^3 S_{1x}$"), (1, r"$10^3 S_{1y}$"))):
            im = axes[row, col].imshow(
                transverse[frame, ..., component].T,
                origin="lower",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
            )
            axes[row, col].set_title(f"{label}: {name}")
            axes[row, col].set_xlabel("lattice x")
            axes[row, col].set_ylabel("lattice y")
            fig.colorbar(im, ax=axes[row, col], shrink=0.82)
    fig.suptitle("LLG spin-wave state: the final field is a later phase, not a relaxed endpoint")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parser().parse_args()
    if args.fps < 1:
        raise ValueError("fps must be positive")
    data = np.load(args.input)
    snapshots = data["snapshots"]
    steps = data["snapshot_steps"]
    time_ps = 1e12 * data["time_s"]
    snapshot_time_ps = time_ps[steps]
    mode = data["mode_real"] + 1j * data["mode_imag"]
    mode_scale = max(float(np.max(np.abs(mode.real))), float(np.max(np.abs(mode.imag))))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    save_initial_final(snapshots, args.output_dir / "spinwave_initial_final.png")

    transverse = 1e3 * snapshots[:, 0, ..., :2]
    field_limit = float(np.max(np.abs(transverse)))
    fig = plt.figure(figsize=(10.5, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.25))
    ax_x = fig.add_subplot(grid[0, 0])
    ax_q = fig.add_subplot(grid[1, 0])
    ax_t = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])

    image = ax_x.imshow(
        transverse[0, ..., 0].T,
        origin="lower",
        cmap="RdBu_r",
        vmin=-field_limit,
        vmax=field_limit,
        interpolation="nearest",
    )
    fig.colorbar(image, ax=ax_x, label=r"$10^3 S_{1x}$", shrink=0.86)
    ax_x.set_xlabel("lattice x")
    ax_x.set_ylabel("lattice y")
    ax_x.set_title("a  layer-1 transverse field")

    x, y = np.meshgrid(np.arange(snapshots.shape[2]), np.arange(snapshots.shape[3]), indexing="ij")
    stride = 2
    quiver = ax_q.quiver(
        x[::stride, ::stride],
        y[::stride, ::stride],
        transverse[0, ::stride, ::stride, 0],
        transverse[0, ::stride, ::stride, 1],
        angles="xy",
        scale_units="xy",
        scale=1.4,
        color="#305f8d",
        width=0.006,
        pivot="mid",
    )
    ax_q.set_xlim(-0.7, snapshots.shape[2] - 0.3)
    ax_q.set_ylim(-0.7, snapshots.shape[3] - 0.3)
    ax_q.set_aspect("equal")
    ax_q.set_xlabel("lattice x")
    ax_q.set_ylabel("lattice y")
    ax_q.set_title(r"b  transverse vectors $(10^3 S_{1x},10^3 S_{1y})$")

    ax_t.plot(time_ps, 1e3 * mode.real, color="#d55e00", lw=1.0, label="real")
    ax_t.plot(time_ps, 1e3 * mode.imag, color="#0072b2", lw=1.0, label="imag")
    cursor = ax_t.axvline(0, color="black", lw=1.0)
    ax_t.set_xlim(time_ps[0], time_ps[-1])
    ax_t.set_ylim(-1.08 * 1e3 * mode_scale, 1.08 * 1e3 * mode_scale)
    ax_t.set_xlabel("time (ps)")
    ax_t.set_ylabel(r"Fourier amplitude ($10^{-3}$)")
    ax_t.set_title("c  complex amplitude of the selected spatial mode")
    ax_t.legend(ncol=2, loc="upper right")

    complex_path, = ax_c.plot([], [], color="#7a5195", lw=1.1)
    complex_point, = ax_c.plot([], [], "o", color="#ef5675", ms=5)
    ax_c.axhline(0, color="0.85", lw=0.7)
    ax_c.axvline(0, color="0.85", lw=0.7)
    ax_c.set_xlim(-1.08 * 1e3 * mode_scale, 1.08 * 1e3 * mode_scale)
    ax_c.set_ylim(-1.08 * 1e3 * mode_scale, 1.08 * 1e3 * mode_scale)
    ax_c.set_aspect("equal", adjustable="box")
    ax_c.set_xlabel(r"Re mode ($10^{-3}$)")
    ax_c.set_ylabel(r"Im mode ($10^{-3}$)")
    ax_c.set_title("d  precession in complex-mode plane")

    title = fig.suptitle("")

    def update(frame: int):
        image.set_data(transverse[frame, ..., 0].T)
        quiver.set_UVC(
            transverse[frame, ::stride, ::stride, 0],
            transverse[frame, ::stride, ::stride, 1],
        )
        step = int(steps[frame])
        cursor.set_xdata([snapshot_time_ps[frame], snapshot_time_ps[frame]])
        complex_path.set_data(1e3 * mode.real[: step + 1], 1e3 * mode.imag[: step + 1])
        complex_point.set_data([1e3 * mode.real[step]], [1e3 * mode.imag[step]])
        title.set_text(
            f"Deterministic LLG spin wave | t = {snapshot_time_ps[frame]:.2f} ps | "
            "transverse components magnified 1000x"
        )
        return image, quiver, cursor, complex_path, complex_point, title

    animation = FuncAnimation(fig, update, frames=len(snapshots), interval=1000 / args.fps)
    animation.save(
        args.output_dir / "spinwave_trajectory.gif",
        writer=PillowWriter(fps=args.fps),
        dpi=110,
    )
    plt.close(fig)
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
