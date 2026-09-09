"""Refresh plots and summaries from saved trajectories without resampling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from altermagnetism_LLG.model.evaluate import (  # noqa: E402
    _json_row,
    _plot_histograms,
    _plot_paths,
)
from altermagnetism_LLG.model.evaluation import (  # noqa: E402
    BasinCalibration,
    compare,
    path_descriptors,
    summarize,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-dir", type=Path, nargs="+", required=True)
    p.add_argument("--system", default="d_wave_altermagnet")
    return p


def refresh(directory: Path, system: str) -> None:
    report_path = directory / "evaluation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    calibration = BasinCalibration(**report["basin_calibration"])
    generated_path = directory / "generated_trajectories.h5"

    with h5py.File(report["input"], "r", swmr=True) as source, h5py.File(
        generated_path, "r", swmr=True
    ) as generated:
        source_group = source[system]
        time = source["time"][:].astype(np.float32)
        positions = {
            int(index): position
            for position, index in enumerate(generated["source_index"][:])
        }
        for condition in report["conditions"]:
            reference_rows = []
            flow_rows = []
            deterministic_rows = []
            for path in condition["paths"]:
                source_index = int(path["source_index"])
                position = positions[source_index]
                reference_row = path_descriptors(
                    source_group["spins"][source_index], time, calibration
                )
                flow_row = path_descriptors(
                    generated["flow/spins"][position], time, calibration
                )
                reference_rows.append(reference_row)
                flow_rows.append(flow_row)
                path["reference"] = _json_row(reference_row)
                path["flow"] = _json_row(flow_row)
                if "deterministic" in generated:
                    deterministic_row = path_descriptors(
                        generated["deterministic/spins"][position],
                        time,
                        calibration,
                    )
                    deterministic_rows.append(deterministic_row)
                    path["deterministic"] = _json_row(deterministic_row)

            condition["reference"] = summarize(reference_rows)
            condition["flow"] = summarize(flow_rows)
            condition["flow_errors"] = compare(
                condition["reference"], condition["flow"]
            )
            if deterministic_rows:
                condition["deterministic"] = summarize(deterministic_rows)
                condition["deterministic_errors"] = compare(
                    condition["reference"], condition["deterministic"]
                )

    report_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _plot_histograms(
        report["conditions"], directory / "endpoint_histograms.png",
        calibration,
    )
    _plot_paths(
        report["conditions"], time,
        directory / "neel_z_paths.png",
        "mean_neel_z_path", r"$n_z$",
        calibration,
    )
    _plot_paths(
        report["conditions"], time,
        directory / "spatial_std_paths.png",
        "mean_spatial_std_path", r"spatial std of local $n_z$",
    )
    print(directory.resolve())


def main() -> None:
    args = parser().parse_args()
    for directory in args.evaluation_dir:
        refresh(directory, args.system)


if __name__ == "__main__":
    main()
