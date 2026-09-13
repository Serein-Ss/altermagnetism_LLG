"""Visual summary of the certified phase-1 standard dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from altermagnetism_LLG.scripts.core.project_paths import asset_path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "datasets" / "standard_v2" / "suite_certification.json")
    parser.add_argument("--output", type=Path, default=ROOT / "assets" / "research" / "standard_v2" / "dataset_overview.png")
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    sizes = [item["size"][0] for item in report["files"]]
    drives = sorted({row["drive_T"] for item in report["files"] for row in item["conditions"]})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(drives)))
    for drive, color in zip(drives, colors):
        rows = [next(row for row in item["conditions"] if row["drive_T"] == drive) for item in report["files"]]
        endpoint = [
            row.get("negative_endpoint_fraction", row.get("switching_fraction"))
            for row in rows
        ]
        axes[0].plot(sizes, endpoint, "o-", color=color, label=f"{drive:.2f} T")
        axes[1].plot(sizes, [row["nonuniform_fraction_all_paths"] for row in rows], "s-", color=color, label=f"{drive:.2f} T")
    axes[0].set(xlabel="linear size (cells)", ylabel="negative endpoint fraction", ylim=(-0.03, 1.03))
    axes[1].set(xlabel="linear size (cells)", ylabel="nonuniform fraction", ylim=(-0.03, 1.03))
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    labels = [f"L={size}" for size in sizes]
    coherent = [sum(row.get("coherent_like_negative_endpoint", row.get("coherent_like_switch")) for row in item["conditions"]) for item in report["files"]]
    nonuniform = [sum(row.get("spatially_nonuniform_negative_endpoint", row.get("spatially_nonuniform_switch")) for row in item["conditions"]) for item in report["files"]]
    no_crossing = [sum(row["no_crossing"] for row in item["conditions"]) for item in report["files"]]
    axes[2].bar(labels, no_crossing, label="no crossing")
    axes[2].bar(labels, coherent, bottom=no_crossing, label="coherent negative endpoint")
    axes[2].bar(labels, nonuniform, bottom=np.asarray(no_crossing) + np.asarray(coherent), label="nonuniform negative endpoint")
    axes[2].set(ylabel="trajectory count", title="Unbiased retained paths")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"Standard LLG suite: {report['status'].replace('_', ' ')}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(asset_path(args.output), dpi=220)
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
