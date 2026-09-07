"""Select auditable representative paths and visualize their spatial dynamics."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FILES = {
    16: ROOT / "data" / "standard_v2" / "train" / "d_wave_altermagnet_L16.h5",
    32: ROOT / "data" / "standard_v2" / "train" / "d_wave_altermagnet_L32.h5",
    64: ROOT / "data" / "standard_v2" / "train" / "d_wave_altermagnet_L64.h5",
    96: ROOT / "data" / "standard_v2" / "test_large" / "d_wave_altermagnet_L96.h5",
}
OUTPUT = ROOT / "assets" / "standard_v2" / "trajectory_examples"
CONDITION_ID = 1  # 5 K, 0.78 T: all three outcomes coexist at large size.
THRESHOLD = 0.25
LABELS = ("no_crossing", "coherent_like", "spatially_nonuniform")
DISPLAY = {
    "no_crossing": "No crossing",
    "coherent_like": "Coherent-like switch",
    "spatially_nonuniform": "Spatially nonuniform switch",
}
COLORS = {
    "no_crossing": "#3B6FB6",
    "coherent_like": "#E28E2C",
    "spatially_nonuniform": "#2A9D63",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def path_descriptors(spins: np.ndarray, time: np.ndarray) -> dict:
    local = 0.5 * (spins[:, 0, ..., 2] - spins[:, 1, ..., 2])
    order = local.mean(axis=(1, 2))
    spatial_std = local.std(axis=(1, 2))
    sign = local < 0
    wall_density = 0.5 * (
        (sign != np.roll(sign, -1, axis=1)).mean(axis=(1, 2))
        + (sign != np.roll(sign, -1, axis=2)).mean(axis=(1, 2))
    )
    crossings = np.flatnonzero(order < 0)
    switched = bool(order[-1] < 0)
    transition = np.flatnonzero(np.abs(order) < 0.5)
    transition_std = float(spatial_std[transition].max()) if len(transition) else 0.0
    if not len(crossings):
        label = "no_crossing"
        representative_frame = int(np.argmin(order))
        first_passage_ps = None
    elif switched and transition_std < THRESHOLD:
        label = "coherent_like"
        representative_frame = int(transition[np.argmax(spatial_std[transition])])
        first_passage_ps = float(time[crossings[0]] * 1e12)
    elif switched:
        label = "spatially_nonuniform"
        representative_frame = int(transition[np.argmax(spatial_std[transition])])
        first_passage_ps = float(time[crossings[0]] * 1e12)
    else:
        label = "crossing_return"
        representative_frame = int(np.argmax(spatial_std))
        first_passage_ps = float(time[crossings[0]] * 1e12)
    return {
        "label": label,
        "endpoint_order": float(order[-1]),
        "minimum_order": float(order.min()),
        "first_passage_ps": first_passage_ps,
        "transition_spatial_std": transition_std,
        "peak_wall_density": float(wall_density.max()),
        "representative_frame": representative_frame,
        "representative_time_ps": float(time[representative_frame] * 1e12),
        "order": order,
        "spatial_std": spatial_std,
        "wall_density": wall_density,
        "local": local,
    }


def choose_medoid(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    features = np.asarray(
        [
            [
                row["endpoint_order"],
                row["minimum_order"],
                row["first_passage_ps"] if row["first_passage_ps"] is not None else 1.1,
                row["transition_spatial_std"],
                row["peak_wall_density"],
            ]
            for row in rows
        ],
        dtype=float,
    )
    centre = np.median(features, axis=0)
    scale = np.median(np.abs(features - centre), axis=0)
    scale[scale < 1e-8] = 1.0
    index = int(np.argmin(np.sum(((features - centre) / scale) ** 2, axis=1)))
    return rows[index]


def analyze_size(path: Path) -> tuple[np.ndarray, dict[str, dict | None], dict]:
    with h5py.File(path, "r") as h5:
        time = h5["time"][:]
        group = h5["d_wave_altermagnet"]
        condition_indices = np.flatnonzero(group["condition_id"][:] == CONDITION_ID)
        buckets = {label: [] for label in LABELS}
        for index in condition_indices:
            descriptors = path_descriptors(group["spins"][index], time)
            if descriptors["label"] not in buckets:
                continue
            descriptors.update(
                {
                    "index": int(index),
                    "trajectory_id": int(group["trajectory_id"][index]),
                    "trajectory_seed": int(group["trajectory_seed"][index]),
                }
            )
            buckets[descriptors["label"]].append(descriptors)
        selected = {label: choose_medoid(buckets[label]) for label in LABELS}
        for row in selected.values():
            if row is not None:
                row["spins"] = group["spins"][row["index"]]
        physical = group["physical_condition"][condition_indices[0]].tolist()
        counts = {label: len(buckets[label]) for label in LABELS}
    return time, selected, {"physical_condition": physical, "counts": counts}


def add_heatmap(ax, field: np.ndarray, title: str = ""):
    image = ax.imshow(field, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#777777")
        spine.set_linewidth(0.5)
    return image


def plot_size_examples(size: int, time: np.ndarray, selected: dict[str, dict | None]) -> None:
    time_ps = time * 1e12
    fig = plt.figure(figsize=(14.0, 7.7), constrained_layout=True)
    grid = fig.add_gridspec(3, 5, width_ratios=[1.65, 1, 1, 1, 1])
    last_image = None
    for row_index, label in enumerate(LABELS):
        row = selected[label]
        if row is None:
            ax = fig.add_subplot(grid[row_index, :])
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"{DISPLAY[label]}: not observed among the 30 paths at this size and condition",
                ha="center",
                va="center",
                color="#666666",
                fontsize=10,
            )
            continue
        ax = fig.add_subplot(grid[row_index, 0])
        ax.plot(time_ps, row["order"], color=COLORS[label], lw=2.0, label=r"global $n_z$")
        ax.plot(time_ps, row["spatial_std"], color="#555555", lw=1.2, ls="--", label=r"spatial $\sigma(n_z)$")
        ax.axhline(0, color="#999999", lw=0.8)
        ax.axvspan(0, 0.5, color="#D9E6F2", alpha=0.45, lw=0)
        ax.axvline(row["representative_time_ps"], color=COLORS[label], lw=1.0, ls=":")
        ax.set_xlim(time_ps[0], time_ps[-1])
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("order / spatial spread")
        ax.set_title(
            f"{DISPLAY[label]}\nID {row['trajectory_id']}, seed {row['trajectory_seed']}",
            loc="left",
            color=COLORS[label],
            fontweight="bold",
        )
        if row_index == 0:
            ax.legend(loc="lower left", fontsize=7)
        frame_indices = [0, int(np.argmin(np.abs(time_ps - 0.5))), row["representative_frame"], len(time) - 1]
        frame_titles = ["initial", "pulse end", r"diagnostic $t^*$", "final"]
        for column, (frame, frame_title) in enumerate(zip(frame_indices, frame_titles), start=1):
            map_ax = fig.add_subplot(grid[row_index, column])
            last_image = add_heatmap(
                map_ax,
                row["local"][frame],
                f"{frame_title}\n{time_ps[frame]:.2f} ps",
            )
    if last_image is not None:
        fig.colorbar(last_image, ax=fig.axes, shrink=0.68, pad=0.01, label=r"local Néel component $n_z(x,y)$")
    fig.suptitle(
        f"Representative L={size} trajectories at the same condition: 5 K, 0.78 T",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(OUTPUT / f"L{size}_representative_paths.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_size_matrix(all_selected: dict[int, dict[str, dict | None]]) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(8.2, 10.2), constrained_layout=True)
    last_image = None
    for row_index, size in enumerate(FILES):
        for column, label in enumerate(LABELS):
            ax = axes[row_index, column]
            row = all_selected[size][label]
            if row is None:
                ax.set_facecolor("#F1F1F1")
                ax.text(0.5, 0.5, "not observed", ha="center", va="center", color="#777777")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                frame = row["representative_frame"]
                last_image = add_heatmap(
                    ax,
                    row["local"][frame],
                    rf"$t^*$={row['representative_time_ps']:.2f} ps, $\sigma$={row['spatial_std'][frame]:.2f}",
                )
            if row_index == 0:
                ax.text(
                    0.5,
                    1.22,
                    DISPLAY[label],
                    transform=ax.transAxes,
                    ha="center",
                    color=COLORS[label],
                    fontsize=10,
                    fontweight="bold",
                )
            if column == 0:
                ax.text(-0.18, 0.5, f"L={size}", transform=ax.transAxes, ha="right", va="center", fontsize=11, fontweight="bold")
    if last_image is not None:
        fig.colorbar(last_image, ax=axes, shrink=0.65, pad=0.025, label=r"local $n_z(x,y)$")
    fig.suptitle("What the three path classes look like as system size grows\n(all examples: 5 K, 0.78 T)", fontsize=14, fontweight="bold")
    fig.savefig(OUTPUT / "all_sizes_path_class_matrix.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_mechanism_explainer(time: np.ndarray, selected: dict[str, dict | None]) -> None:
    time_ps = time * 1e12
    fig = plt.figure(figsize=(11.5, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.15, 0.75])
    last_image = None
    for column, label in enumerate(LABELS):
        row = selected[label]
        if row is None:
            continue
        top = fig.add_subplot(grid[0, column])
        top.plot(time_ps, row["order"], color=COLORS[label], lw=2.2, label=r"global $n_z$")
        top.plot(time_ps, row["spatial_std"], color="#444444", lw=1.3, ls="--", label=r"spatial $\sigma$")
        top.axhline(0, color="#999999", lw=0.8)
        top.axvline(row["representative_time_ps"], color=COLORS[label], ls=":", lw=1.0)
        top.set_xlim(0, 1)
        top.set_ylim(-1.05, 1.05)
        top.set_xlabel("time (ps)")
        if column == 0:
            top.set_ylabel("global order / spread")
            top.legend(fontsize=7, loc="lower left")
        top.set_title(DISPLAY[label], color=COLORS[label], fontweight="bold", fontsize=11)
        middle = fig.add_subplot(grid[1, column])
        frame = row["representative_frame"]
        last_image = add_heatmap(middle, row["local"][frame], rf"diagnostic frame $t^*$={row['representative_time_ps']:.2f} ps")
        bottom = fig.add_subplot(grid[2, column])
        bottom.hist(row["local"][frame].ravel(), bins=36, range=(-1, 1), color=COLORS[label], alpha=0.82)
        bottom.axvline(0, color="#777777", lw=0.8)
        bottom.set_xlim(-1, 1)
        bottom.set_xlabel(r"local $n_z$")
        if column == 0:
            bottom.set_ylabel("cell count")
    if last_image is not None:
        fig.colorbar(last_image, ax=fig.axes, shrink=0.55, pad=0.015, label=r"local $n_z(x,y)$")
    fig.suptitle(
        "Same size and same physical condition, but different finite-temperature paths\nL=96, T=5 K, drive=0.78 T",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(OUTPUT / "mechanism_definitions_same_condition.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def animate_size(size: int, time: np.ndarray, selected: dict[str, dict | None]) -> None:
    available = [(label, selected[label]) for label in LABELS if selected[label] is not None]
    ncols = len(available)
    time_ps = time * 1e12
    fig, axes = plt.subplots(2, ncols, figsize=(3.7 * ncols, 5.0), gridspec_kw={"height_ratios": [0.8, 1.25]}, squeeze=False)
    images = []
    cursors = []
    for column, (label, row) in enumerate(available):
        curve_ax = axes[0, column]
        curve_ax.plot(time_ps, row["order"], color=COLORS[label], lw=2.0)
        curve_ax.plot(time_ps, row["spatial_std"], color="#555555", lw=1.0, ls="--")
        curve_ax.axhline(0, color="#AAAAAA", lw=0.7)
        cursor = curve_ax.axvline(time_ps[0], color="black", lw=1.0)
        cursors.append(cursor)
        curve_ax.set_xlim(0, 1)
        curve_ax.set_ylim(-1.05, 1.05)
        curve_ax.set_title(DISPLAY[label], color=COLORS[label], fontweight="bold")
        curve_ax.set_xlabel("time (ps)")
        if column == 0:
            curve_ax.set_ylabel(r"global $n_z$ / spatial $\sigma$")
        map_ax = axes[1, column]
        image = map_ax.imshow(row["local"][0], origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest", animated=True)
        images.append(image)
        map_ax.set_xticks([])
        map_ax.set_yticks([])
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.08, top=0.88, wspace=0.28, hspace=0.28)
    colorbar_axis = fig.add_axes([0.92, 0.12, 0.018, 0.36])
    colorbar = fig.colorbar(images[-1], cax=colorbar_axis)
    colorbar.set_label(r"local Néel component $n_z(x,y)$")
    title = fig.suptitle(f"L={size}, 5 K, 0.78 T | t = 0.00 ps", fontsize=13, fontweight="bold")

    frame_indices = list(range(0, len(time), 2))
    if frame_indices[-1] != len(time) - 1:
        frame_indices.append(len(time) - 1)

    def update(frame_index):
        for image, (_, row) in zip(images, available):
            image.set_data(row["local"][frame_index])
        for cursor in cursors:
            cursor.set_xdata([time_ps[frame_index], time_ps[frame_index]])
        title.set_text(f"L={size}, 5 K, 0.78 T | t = {time_ps[frame_index]:.2f} ps")
        return [*images, *cursors, title]

    animation = FuncAnimation(fig, update, frames=frame_indices, interval=85, blit=False)
    animation.save(OUTPUT / f"L{size}_representative_paths.gif", writer=PillowWriter(fps=12), dpi=105)
    plt.close(fig)


def serializable(row: dict | None) -> dict | None:
    if row is None:
        return None
    return {
        key: value
        for key, value in row.items()
        if key not in {"order", "spatial_std", "wall_density", "local", "spins"}
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_times = {}
    all_selected = {}
    report = {
        "selection_condition": {"temperature_K": 5.0, "drive_damping_like_T": 0.78},
        "mechanism_threshold": THRESHOLD,
        "selection_rule": "class medoid in endpoint, minimum order, first-passage time, transition spatial spread and peak periodic-wall density",
        "sizes": {},
    }
    for size, path in FILES.items():
        time, selected, metadata = analyze_size(path)
        all_times[size] = time
        all_selected[size] = selected
        report["sizes"][str(size)] = {
            **metadata,
            "selected": {label: serializable(selected[label]) for label in LABELS},
        }
        plot_size_examples(size, time, selected)
        animate_size(size, time, selected)
    plot_size_matrix(all_selected)
    plot_mechanism_explainer(all_times[96], all_selected[96])
    (OUTPUT / "representative_examples.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "files": sorted(path.name for path in OUTPUT.iterdir())}, indent=2))


if __name__ == "__main__":
    main()
