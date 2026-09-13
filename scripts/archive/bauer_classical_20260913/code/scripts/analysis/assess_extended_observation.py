"""Assess whether extended real LLG paths reach data-calibrated basins."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from altermagnetism_LLG.scripts.analysis.evaluation import (  # noqa: E402
    calibrate_basins,
    path_descriptors,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from altermagnetism_LLG.scripts.core.project_paths import resolve_recorded_path


CLASSIFICATION_KEYS = (
    "crossed_zero",
    "negative_endpoint",
    "committed_switch",
    "crossing_return",
    "unresolved_transition",
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration-files", type=Path, nargs="+", required=True)
    p.add_argument("--evaluation-files", type=Path, nargs="+", required=True)
    p.add_argument("--system", default="d_wave_altermagnet")
    p.add_argument("--residence-ps", type=float, default=0.10)
    p.add_argument("--max-unresolved-fraction", type=float, default=0.20)
    p.add_argument("--output", type=Path, required=True)
    return p


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_prefix(path: Path, system: str) -> dict:
    with h5py.File(path, "r", swmr=True) as extended:
        source_path = resolve_recorded_path(str(extended.attrs["continued_from"]))
        with h5py.File(source_path, "r", swmr=True) as source:
            old_frames = len(source["time"])
            checks = {
                "time": np.array_equal(
                    extended["time"][:old_frames], source["time"][:]
                )
            }
            for name in ("spins", "magnetization", "neel", "energy"):
                checks[name] = all(
                    np.array_equal(
                        extended[system][name][index, :old_frames],
                        source[system][name][index],
                    )
                    for index in range(len(source[system][name]))
                )
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hash_matches = sha256(path) == manifest["output_sha256"]
    return {
        "source": str(source_path),
        "datasets": checks,
        "manifest": str(manifest_path),
        "output_sha256_matches_manifest": hash_matches,
        "exact": bool(all(checks.values()) and hash_matches),
    }


def summarize_rows(rows: list[dict]) -> dict:
    result = {"paths": len(rows)}
    for key in CLASSIFICATION_KEYS:
        result[f"{key}_fraction"] = float(
            np.mean([row[key] for row in rows])
        )
    endpoints = np.asarray([row["endpoint"] for row in rows])
    result["endpoint_mean"] = float(endpoints.mean())
    result["endpoint_std"] = float(endpoints.std())
    result["mean_peak_spatial_std"] = float(
        np.mean([row["peak_spatial_std"] for row in rows])
    )
    result["max_spin_norm_error"] = float(
        max(row["max_spin_norm_error"] for row in rows)
    )
    return result


def main() -> None:
    args = parser().parse_args()
    calibration = calibrate_basins(
        args.calibration_files,
        args.system,
        residence_ps=args.residence_ps,
    )
    grouped = defaultdict(list)
    prefix = {}

    for path in args.evaluation_files:
        prefix[str(path)] = verify_prefix(path, args.system)
        with h5py.File(path, "r", swmr=True) as h5:
            group = h5[args.system]
            time = h5["time"][:]
            lattice_size = int(group["spins"].shape[3])
            splits = group["split"][:]
            condition_ids = group["condition_id"][:]
            fields = group["physical_condition"][:, 2]
            for index in range(len(group["spins"])):
                descriptor = path_descriptors(
                    group["spins"][index], time, calibration
                )
                row = {
                    key: descriptor[key]
                    for key in CLASSIFICATION_KEYS
                }
                row.update(
                    endpoint=descriptor["endpoint"],
                    peak_spatial_std=descriptor["peak_spatial_std"],
                    max_spin_norm_error=descriptor["max_spin_norm_error"],
                )
                condition = (
                    lattice_size,
                    int(condition_ids[index]),
                    float(fields[index]),
                )
                grouped[condition].append(row)

    conditions = {}
    must_extend = False
    for (size, condition_id, field), rows in sorted(grouped.items()):
        summary = summarize_rows(rows)
        summary.update(
            lattice_size=size,
            condition_id=condition_id,
            field_tesla=field,
        )
        if (
            summary["unresolved_transition_fraction"]
            > args.max_unresolved_fraction
        ):
            must_extend = True
        conditions[f"L{size}/condition_{condition_id}"] = summary

    report = {
        "status": (
            "extend_observation"
            if must_extend
            else "observation_window_accepted"
        ),
        "decision_rule": {
            "max_unresolved_fraction": args.max_unresolved_fraction,
            "scope": "all real trajectories in each size-condition group",
        },
        "calibration": calibration.to_dict(),
        "prefix_integrity": prefix,
        "conditions": conditions,
    }
    if not all(item["exact"] for item in prefix.values()):
        report["status"] = "prefix_integrity_failure"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "observation_window_accepted":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
