"""Create a complete visual-QA bundle for all schema-v2 LLG trajectories.

Static trajectory diagnostics are PNG-only, following the project convention.
One compact GIF per size/drive condition shows every trajectory in that block.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
FILES = {
    16: ROOT / "data" / "standard_v2" / "train" / "d_wave_altermagnet_L16.h5",
    32: ROOT / "data" / "standard_v2" / "train" / "d_wave_altermagnet_L32.h5",
    64: ROOT / "data" / "standard_v2" / "train" / "d_wave_altermagnet_L64.h5",
    96: ROOT / "data" / "standard_v2" / "test_large" / "d_wave_altermagnet_L96.h5",
}
DEFAULT_OUTPUT = ROOT / "assets" / "standard_v2" / "all_trajectories"
THRESHOLD = 0.25
PULSE_END_PS = 0.5
LABELS = ("no_crossing", "crossing_return", "coherent_like", "spatially_nonuniform")
DISPLAY = {
    "no_crossing": "No crossing",
    "crossing_return": "Crossing and return",
    "coherent_like": "Coherent-like switch",
    "spatially_nonuniform": "Spatially nonuniform switch",
}
SHORT_LABEL = {
    "no_crossing": "NC",
    "crossing_return": "CR",
    "coherent_like": "COH",
    "spatially_nonuniform": "NU",
}
COLORS = {
    "no_crossing": "#3B6FB6",
    "crossing_return": "#9A6BB5",
    "coherent_like": "#E28E2C",
    "spatially_nonuniform": "#2A9D63",
}
SPLITS = {0: "train", 1: "validation", 2: "test"}
SHORT_SPLIT = {"train": "TR", "validation": "VA", "test": "TE"}


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
        "savefig.facecolor": "white",
    }
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--video-stride", type=int, default=2, help="Use every Nth saved frame in GIFs")
    return p


def descriptors(spins: np.ndarray, time_s: np.ndarray) -> dict:
    local = 0.5 * (spins[:, 0, ..., 2] - spins[:, 1, ..., 2])
    order = local.mean(axis=(1, 2))
    spatial_std = local.std(axis=(1, 2))
    sign = local < 0
    wall_density = 0.5 * (
        (sign != np.roll(sign, -1, axis=1)).mean(axis=(1, 2))
        + (sign != np.roll(sign, -1, axis=2)).mean(axis=(1, 2))
    )
    crossings = np.flatnonzero(order < 0)
    transition = np.flatnonzero(np.abs(order) < 0.5)
    transition_std = float(spatial_std[transition].max()) if len(transition) else 0.0
    if not len(crossings):
        label = "no_crossing"
        diagnostic_frame = int(np.argmin(order))
        first_passage_ps = None
    elif order[-1] >= 0:
        label = "crossing_return"
        diagnostic_frame = int(np.argmax(spatial_std))
        first_passage_ps = float(time_s[crossings[0]] * 1e12)
    elif transition_std < THRESHOLD:
        label = "coherent_like"
        diagnostic_frame = int(transition[np.argmax(spatial_std[transition])])
        first_passage_ps = float(time_s[crossings[0]] * 1e12)
    else:
        label = "spatially_nonuniform"
        diagnostic_frame = int(transition[np.argmax(spatial_std[transition])])
        first_passage_ps = float(time_s[crossings[0]] * 1e12)
    return {
        "label": label,
        "endpoint_order": float(order[-1]),
        "minimum_order": float(order.min()),
        "first_passage_ps": first_passage_ps,
        "transition_spatial_std": transition_std,
        "peak_spatial_std": float(spatial_std.max()),
        "peak_wall_density": float(wall_density.max()),
        "diagnostic_frame": diagnostic_frame,
        "diagnostic_time_ps": float(time_s[diagnostic_frame] * 1e12),
        "order": order,
        "spatial_std": spatial_std,
        "wall_density": wall_density,
        "local": local,
    }


def file_token(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def add_map(ax, field: np.ndarray, title: str):
    image = ax.imshow(
        field,
        origin="lower",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#777777")
        spine.set_linewidth(0.5)
    return image


def plot_trajectory(
    output: Path,
    size: int,
    drive_t: float,
    path_index: int,
    trajectory_id: int,
    seed: int,
    split: str,
    time_s: np.ndarray,
    row: dict,
) -> None:
    time_ps = time_s * 1e12
    pulse_frame = int(np.argmin(np.abs(time_ps - PULSE_END_PS)))
    frames = [0, pulse_frame, row["diagnostic_frame"], len(time_s) - 1]
    titles = [
        f"initial\n{time_ps[0]:.2f} ps",
        f"pulse end\n{time_ps[pulse_frame]:.2f} ps",
        f"diagnostic t*\n{time_ps[row['diagnostic_frame']]:.2f} ps",
        f"final\n{time_ps[-1]:.2f} ps",
    ]

    fig = plt.figure(figsize=(12.0, 5.7), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=[1.55, 1.55, 1.0, 1.0])
    curve_ax = fig.add_subplot(grid[:, :2])
    curve_ax.plot(time_ps, row["order"], color=COLORS[row["label"]], lw=2.0, label=r"global $n_z$")
    curve_ax.plot(time_ps, row["spatial_std"], color="#555555", lw=1.25, ls="--", label=r"spatial $\sigma(n_z)$")
    curve_ax.plot(time_ps, row["wall_density"], color="#9A6BB5", lw=1.1, ls=":", label="periodic wall density")
    curve_ax.axhline(0, color="#999999", lw=0.8)
    curve_ax.axvspan(0, PULSE_END_PS, color="#D9E6F2", alpha=0.45, lw=0, label="SOT pulse")
    curve_ax.axvline(row["diagnostic_time_ps"], color=COLORS[row["label"]], lw=1.0, ls="-.")
    curve_ax.set_xlim(time_ps[0], time_ps[-1])
    curve_ax.set_ylim(-1.05, 1.05)
    curve_ax.set_xlabel("time (ps)")
    curve_ax.set_ylabel("order / spatial descriptor")
    curve_ax.legend(loc="lower left", fontsize=7, ncol=2)
    curve_ax.set_title(
        f"{DISPLAY[row['label']]}  |  endpoint={row['endpoint_order']:.3f}, "
        f"peak spread={row['peak_spatial_std']:.3f}, peak wall density={row['peak_wall_density']:.3f}",
        loc="left",
        color=COLORS[row["label"]],
        fontweight="bold",
    )

    last_image = None
    for map_index, (frame, title) in enumerate(zip(frames, titles)):
        map_ax = fig.add_subplot(grid[map_index // 2, 2 + map_index % 2])
        last_image = add_map(map_ax, row["local"][frame], title)
    fig.colorbar(last_image, ax=fig.axes, shrink=0.72, pad=0.01, label=r"local Néel component $n_z(x,y)$")
    fpt = "none" if row["first_passage_ps"] is None else f"{row['first_passage_ps']:.3f} ps"
    fig.suptitle(
        f"L={size}, T=5 K, damping-like drive={drive_t:.2f} T  |  "
        f"file index={path_index}, trajectory ID={trajectory_id}, split={split}\n"
        f"seed={seed}, first passage={fpt}",
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def condition_overview(output: Path, size: int, drive_t: float, time_s: np.ndarray, rows: list[dict]) -> None:
    time_ps = time_s * 1e12
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    for row in rows:
        color = COLORS[row["label"]]
        axes[0].plot(time_ps, row["order"], color=color, lw=0.9, alpha=0.65)
        axes[1].plot(time_ps, row["spatial_std"], color=color, lw=0.9, alpha=0.65)
    for ax in axes:
        ax.axvspan(0, PULSE_END_PS, color="#D9E6F2", alpha=0.35, lw=0)
        ax.set_xlim(time_ps[0], time_ps[-1])
        ax.set_xlabel("time (ps)")
    axes[0].axhline(0, color="#999999", lw=0.8)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_ylabel(r"global $n_z$")
    axes[0].set_title("All global paths")
    axes[1].axhline(THRESHOLD, color="#777777", lw=0.8, ls="--")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel(r"spatial $\sigma(n_z)$")
    axes[1].set_title("All spatial-spread paths")
    counts = {label: sum(row["label"] == label for row in rows) for label in LABELS}
    count_text = ", ".join(f"{DISPLAY[k]}={v}" for k, v in counts.items() if v)
    fig.suptitle(f"L={size}, T=5 K, drive={drive_t:.2f} T, n={len(rows)}\n{count_text}", fontsize=11, fontweight="bold")
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def load_font(size: int):
    path = Path(mpl.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    return ImageFont.truetype(str(path), size=size)


def create_condition_gif(
    output: Path,
    size: int,
    drive_t: float,
    time_s: np.ndarray,
    rows: list[dict],
    stride: int,
) -> None:
    ncols = 5
    nrows = math.ceil(len(rows) / ncols)
    tile = 112
    label_h = 24
    header_h = 48
    footer_h = 24
    canvas_w = ncols * tile
    canvas_h = header_h + nrows * (tile + label_h) + footer_h
    title_font = load_font(18)
    small_font = load_font(11)
    cmap = mpl.colormaps["RdBu_r"]
    frame_indices = list(range(0, len(time_s), stride))
    if frame_indices[-1] != len(time_s) - 1:
        frame_indices.append(len(time_s) - 1)
    frames = []
    for frame_index in frame_indices:
        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 7),
            f"L={size} | T=5 K | drive={drive_t:.2f} T | all {len(rows)} paths | t={time_s[frame_index]*1e12:.2f} ps",
            fill="black",
            font=title_font,
        )
        for index, row in enumerate(rows):
            grid_x = index % ncols
            grid_y = index // ncols
            x0 = grid_x * tile
            y0 = header_h + grid_y * (tile + label_h)
            rgb = (cmap(np.clip((row["local"][frame_index] + 1.0) / 2.0, 0, 1))[..., :3] * 255).astype(np.uint8)
            image = Image.fromarray(rgb).resize((tile, tile), resample=Image.Resampling.NEAREST)
            canvas.paste(image, (x0, y0))
            color = tuple(int(COLORS[row["label"]][i : i + 2], 16) for i in (1, 3, 5))
            draw.rectangle((x0, y0, x0 + tile - 1, y0 + tile - 1), outline=color, width=3)
            draw.text(
                (x0 + 3, y0 + tile + 3),
                f"ID {row['trajectory_id']} | {SHORT_SPLIT[row['split']]} | {SHORT_LABEL[row['label']]}",
                fill=color,
                font=small_font,
            )
        draw.text(
            (8, canvas_h - footer_h + 4),
            "NC=no crossing; CR=crossing-return; COH=coherent-like; NU=nonuniform. Blue=-1, red=+1.",
            fill="#333333",
            font=small_font,
        )
        frames.append(canvas)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=False,
        disposal=2,
    )


def plot_dataset_overview(output: Path, grouped: dict[tuple[int, float], list[dict]], times: dict[int, np.ndarray]) -> None:
    drives = (0.70, 0.78, 0.80)
    fig, axes = plt.subplots(4, 3, figsize=(12.5, 11.5), sharex=True, sharey=True, constrained_layout=True)
    for row_index, size in enumerate(FILES):
        time_ps = times[size] * 1e12
        for column, drive in enumerate(drives):
            ax = axes[row_index, column]
            rows = grouped[(size, drive)]
            for item in rows:
                ax.plot(time_ps, item["order"], color=COLORS[item["label"]], lw=0.7, alpha=0.55)
            mean = np.mean([item["order"] for item in rows], axis=0)
            ax.plot(time_ps, mean, color="black", lw=1.8)
            ax.axhline(0, color="#AAAAAA", lw=0.6)
            ax.axvspan(0, PULSE_END_PS, color="#D9E6F2", alpha=0.30, lw=0)
            if row_index == 0:
                ax.set_title(f"drive={drive:.2f} T", fontweight="bold")
            if column == 0:
                ax.set_ylabel(f"L={size}\nglobal $n_z$")
            counts = {label: sum(item["label"] == label for item in rows) for label in LABELS}
            ax.text(
                0.02,
                0.03,
                " / ".join(str(counts[label]) for label in ("no_crossing", "coherent_like", "spatially_nonuniform")),
                transform=ax.transAxes,
                fontsize=7,
                color="#333333",
            )
    for ax in axes[-1]:
        ax.set_xlabel("time (ps)")
    axes[0, 0].text(0.02, 0.91, "counts: no / coherent / nonuniform", transform=axes[0, 0].transAxes, fontsize=7)
    fig.suptitle("All 330 finite-temperature LLG paths\nblack: ensemble mean; colored: individual trajectories", fontsize=14, fontweight="bold")
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_index(output: Path, records: list[dict], condition_files: dict[str, dict]) -> None:
    csv_fields = [
        "size",
        "drive_T",
        "file_index",
        "trajectory_id",
        "trajectory_seed",
        "split",
        "label",
        "endpoint_order",
        "minimum_order",
        "first_passage_ps",
        "transition_spatial_std",
        "peak_spatial_std",
        "peak_wall_density",
        "diagnostic_time_ps",
        "summary_png",
    ]
    with (output / "trajectory_index.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in csv_fields} for row in records)

    groups = []
    for key, files in condition_files.items():
        size, drive = key.split("_")
        subset = [row for row in records if row["size"] == int(size[1:]) and file_token(row["drive_T"]) == drive[1:]]
        cards = "\n".join(
            f'<a class="card" href="{row["summary_png"]}"><img loading="lazy" src="{row["summary_png"]}"><span>'
            f'ID {row["trajectory_id"]} | {row["split"]} | {row["label"]}</span></a>'
            for row in subset
        )
        groups.append(
            f'<section><h2>L={size[1:]}, drive={drive[1:].replace("p", ".")} T</h2>'
            f'<p><a href="{files["overview"]}">condition overview PNG</a> · '
            f'<a href="{files["gif"]}">all-path animation GIF</a></p><div class="grid">{cards}</div></section>'
        )
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>All 330 LLG trajectories</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;color:#222}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}}.card{{border:1px solid #ccc;padding:6px;text-decoration:none;color:#222;background:#fff}}.card img{{width:100%;height:auto;display:block}}.card span{{display:block;padding:5px 2px;font-size:13px}}section{{margin:30px 0}}h1,h2{{margin-bottom:8px}}</style></head>
<body><h1>Complete visual QA: all 330 LLG trajectories</h1>
<p>Every card links to one full-resolution diagnostic PNG. Each condition also has an overlay summary and a GIF containing every path in that condition.</p>
<p><a href="all_330_paths_overview.png">Open dataset-wide overview</a> · <a href="trajectory_index.csv">Open trajectory index</a></p>
{''.join(groups)}</body></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parser().parse_args()
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    records = []
    grouped: dict[tuple[int, float], list[dict]] = {}
    times = {}
    condition_files = {}
    for size, data_path in FILES.items():
        print(f"Reading L={size}: {data_path}", flush=True)
        with h5py.File(data_path, "r") as h5:
            time_s = h5["time"][:]
            times[size] = time_s
            group = h5["d_wave_altermagnet"]
            condition_ids = group["condition_id"][:]
            physical = group["physical_condition"][:]
            for condition_id in np.unique(condition_ids):
                indices = np.flatnonzero(condition_ids == condition_id)
                drive_t = float(physical[indices[0], 2])
                condition_dir = output / f"L{size}" / f"drive_{file_token(drive_t)}T"
                condition_dir.mkdir(parents=True)
                rows = []
                for count, path_index in enumerate(indices, start=1):
                    spins = group["spins"][path_index]
                    row = descriptors(spins, time_s)
                    row.update(
                        {
                            "file_index": int(path_index),
                            "trajectory_id": int(group["trajectory_id"][path_index]),
                            "trajectory_seed": int(group["trajectory_seed"][path_index]),
                            "split": SPLITS[int(group["split"][path_index])],
                        }
                    )
                    png_name = f"trajectory_{int(path_index):04d}.png"
                    plot_trajectory(
                        condition_dir / png_name,
                        size,
                        drive_t,
                        int(path_index),
                        row["trajectory_id"],
                        row["trajectory_seed"],
                        row["split"],
                        time_s,
                        row,
                    )
                    relative_png = (condition_dir / png_name).relative_to(output).as_posix()
                    records.append(
                        {
                            "size": size,
                            "drive_T": drive_t,
                            "file_index": int(path_index),
                            "trajectory_id": row["trajectory_id"],
                            "trajectory_seed": row["trajectory_seed"],
                            "split": row["split"],
                            "label": row["label"],
                            "endpoint_order": row["endpoint_order"],
                            "minimum_order": row["minimum_order"],
                            "first_passage_ps": row["first_passage_ps"],
                            "transition_spatial_std": row["transition_spatial_std"],
                            "peak_spatial_std": row["peak_spatial_std"],
                            "peak_wall_density": row["peak_wall_density"],
                            "diagnostic_time_ps": row["diagnostic_time_ps"],
                            "summary_png": relative_png,
                        }
                    )
                    rows.append(row)
                    if count % 10 == 0 or count == len(indices):
                        print(f"  L={size}, drive={drive_t:.2f} T: {count}/{len(indices)} static summaries", flush=True)
                grouped[(size, drive_t)] = rows
                overview = condition_dir / "condition_overview.png"
                animation = condition_dir / "all_paths.gif"
                condition_overview(overview, size, drive_t, time_s, rows)
                print(f"  Rendering condition GIF: L={size}, drive={drive_t:.2f} T", flush=True)
                create_condition_gif(animation, size, drive_t, time_s, rows, args.video_stride)
                key = f"L{size}_D{file_token(drive_t)}"
                condition_files[key] = {
                    "overview": overview.relative_to(output).as_posix(),
                    "gif": animation.relative_to(output).as_posix(),
                }

    plot_dataset_overview(output / "all_330_paths_overview.png", grouped, times)
    write_index(output, records, condition_files)
    coverage = {
        "status": "complete" if len(records) == 330 else "incomplete",
        "source_trajectory_count": 330,
        "visualized_trajectory_count": len(records),
        "per_trajectory_png_count": len(list(output.glob("L*/drive_*T/trajectory_*.png"))),
        "condition_overview_png_count": len(list(output.glob("L*/drive_*T/condition_overview.png"))),
        "condition_gif_count": len(list(output.glob("L*/drive_*T/all_paths.gif"))),
        "all_source_trajectories_visualized_once": len(records) == 330
        and len({(row["size"], row["file_index"]) for row in records}) == 330,
        "mechanism_threshold": THRESHOLD,
        "note": "Mechanism labels are geometric QA descriptors, not supervised targets or definitive microscopic mechanisms.",
    }
    (output / "coverage_manifest.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    print(json.dumps(coverage, indent=2), flush=True)


if __name__ == "__main__":
    main()
