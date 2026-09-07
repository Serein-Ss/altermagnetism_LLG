"""Plot a convergence report without importing the LLG/PyTorch runtime."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "size_convergence" / "spatial_convergence.json")
    parser.add_argument("--output", type=Path, default=ROOT / "assets" / "size_convergence" / "spatial_convergence.png")
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    results = report["size_results"]
    sizes = np.asarray([item["size"] for item in results])
    limit = report["published_scale_anchor"]["minimum_periodic_pair_no_overlap_cells"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    endpoint = [
        item.get("negative_endpoint_fraction", item.get("switching_fraction"))
        for item in results
    ]
    axes[0].plot(sizes, endpoint, "o-", label="negative endpoint")
    axes[0].plot(sizes, [item.get("nonuniform_negative_endpoint_fraction", item.get("nonuniform_switch_fraction")) for item in results], "s-", label="nonuniform")
    axes[0].axvline(limit, color="0.4", linestyle="--", label="4 wall widths")
    axes[0].set(xlabel="linear size (cells)", ylabel="path fraction", ylim=(-0.03, 1.03))
    axes[0].legend(fontsize=8)
    axes[1].plot(sizes, [item["mean_peak_spatial_std"] for item in results], "o-")
    axes[1].axhline(0.25, color="0.4", linestyle="--")
    axes[1].set(xlabel="linear size (cells)", ylabel="mean peak spatial std")
    for item in results:
        axes[2].plot(np.asarray(item["time_s"]) * 1e12, item["ensemble_mean_order"], label=f"{item['size']}x{item['size']}")
    axes[2].set(xlabel="time (ps)", ylabel="ensemble mean Neel z")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"Convergence gate: {report['status'].replace('_', ' ')}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
