"""Cross-certify the phase-1 standard small-to-large trajectory suite."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILES = (
    ROOT / "data" / "standard_v2" / "train" / "d_wave_altermagnet_L16.h5",
    ROOT / "data" / "standard_v2" / "train" / "d_wave_altermagnet_L32.h5",
    ROOT / "data" / "standard_v2" / "train" / "d_wave_altermagnet_L64.h5",
    ROOT / "data" / "standard_v2" / "test_large" / "d_wave_altermagnet_L96.h5",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()



def outcome_metrics(spins: np.ndarray, time: np.ndarray) -> dict:
    local = 0.5 * (spins[:, :, 0, ..., 2] - spins[:, :, 1, ..., 2])
    order = local.mean(axis=(2, 3))
    spatial_std = local.std(axis=(2, 3))
    crossed = (order < 0).any(axis=1)
    negative_endpoint = order[:, -1] < 0
    transition_std = np.where(np.abs(order) < 0.5, spatial_std, 0.0).max(axis=1)
    nonuniform = negative_endpoint & (transition_std >= 0.25)
    first_passage = []
    for path in order:
        hits = np.flatnonzero(path < 0)
        if len(hits):
            first_passage.append(time[hits[0]])
    sign = local < 0
    wall_density = 0.5 * (
        (sign != np.roll(sign, -1, axis=2)).mean(axis=(2, 3))
        + (sign != np.roll(sign, -1, axis=3)).mean(axis=(2, 3))
    )
    return {
        "paths": int(len(order)),
        "no_crossing": int((~crossed).sum()),
        "crossing_return": int((crossed & ~negative_endpoint).sum()),
        "negative_endpoint": int(negative_endpoint.sum()),
        "coherent_like_negative_endpoint": int((negative_endpoint & ~nonuniform).sum()),
        "spatially_nonuniform_negative_endpoint": int(nonuniform.sum()),
        "negative_endpoint_fraction": float(negative_endpoint.mean()),
        "nonuniform_fraction_all_paths": float(nonuniform.mean()),
        "mean_peak_spatial_std": float(spatial_std.max(axis=1).mean()),
        "mean_peak_wall_density": float(wall_density.max(axis=1).mean()),
        "median_first_passage_ps": float(np.median(first_passage) * 1e12) if first_passage else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", default=DEFAULT_FILES)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "standard_v2" / "suite_certification.json")
    args = parser.parse_args()
    files = []
    all_seeds = []
    protocols = []
    train_count = test_count = 0
    train_sizes = []
    test_sizes = []
    integrity_pass = True
    for path in args.inputs:
        certification_path = path.with_suffix(".certification.json")
        certification = json.loads(certification_path.read_text(encoding="utf-8"))
        manifest_path = path.with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_sha256 = sha256(path)
        integrity_pass &= (
            certification["integrity_status"] == "pass"
            and manifest.get("sha256") == actual_sha256
        )
        with h5py.File(path, "r") as h5:
            group = h5["d_wave_altermagnet"]
            nx, ny = group["spins"].shape[-3:-1]
            split = group["split"][:]
            seeds = group["trajectory_seed"][:]
            all_seeds.extend(int(seed) for seed in seeds)
            time = h5["time"][:]
            split_counts = np.bincount(split, minlength=3).astype(int).tolist()
            is_large_test = group.attrs["split_policy"] == "all_test"
            if is_large_test:
                test_count += len(split)
                test_sizes.append(int(nx))
            else:
                train_count += int((split == 0).sum())
                test_count += int((split == 2).sum())
                train_sizes.append(int(nx))
            condition_rows = []
            for condition_id in np.unique(group["condition_id"][:]):
                indices = np.flatnonzero(group["condition_id"][:] == condition_id)
                physical = group["physical_condition"][indices[0]].tolist()
                protocols.append(tuple(physical))
                metrics = outcome_metrics(group["spins"][indices], time)
                metrics.update(
                    {
                        "condition_id": int(condition_id),
                        "temperature_K": physical[0],
                        "drive_T": physical[2],
                        "split_train_validation_test": np.bincount(split[indices], minlength=3).astype(int).tolist(),
                    }
                )
                condition_rows.append(metrics)
            files.append(
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": actual_sha256,
                    "size": [int(nx), int(ny)],
                    "frames": int(len(time)),
                    "split_policy": group.attrs["split_policy"],
                    "split_train_validation_test": split_counts,
                    "max_spin_norm_error": certification["groups"]["d_wave_altermagnet"]["max_spin_norm_error"],
                    "conditions": condition_rows,
                }
            )

    reference = np.asarray(protocols[0])
    same_base_protocol = all(
        np.allclose(np.delete(np.asarray(row), 2), np.delete(reference, 2), rtol=0.0, atol=0.0)
        for row in protocols
    )
    l64 = next(item for item in files if item["size"][0] == 64)
    l96 = next(item for item in files if item["size"][0] == 96)
    checks = {
        "all_file_integrity_certificates_pass": bool(integrity_pass),
        "all_trajectory_seeds_unique_across_suite": len(all_seeds) == len(set(all_seeds)),
        "same_protocol_except_drive_and_size": bool(same_base_protocol),
        "expected_train_sizes_16_32_64": sorted(train_sizes) == [16, 32, 64],
        "large_size_96_is_all_test": test_sizes == [96] and l96["split_train_validation_test"] == [0, 0, 60],
        "time_grid_101_frames_10fs_to_1ps": all(
            item["frames"] == 101 for item in files
        ) and reference[4] == 5e-17 and reference[5] == 1e-14 and reference[6] == 1e-12,
        "training_contains_nonuniform_paths": sum(c["spatially_nonuniform_negative_endpoint"] for c in l64["conditions"]) >= 3,
        "large_test_contains_nonuniform_paths": sum(c["spatially_nonuniform_negative_endpoint"] for c in l96["conditions"]) >= 3,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    report = {
        "schema_version": 2,
        "status": "ready_for_phase1_model_training" if all(checks.values()) else "not_ready",
        "scope": "literature-Hamiltonian, unbiased finite-temperature path suite for model development and size-held-out evaluation",
        "train_trajectories": train_count,
        "validation_trajectories": sum(item["split_train_validation_test"][1] for item in files),
        "test_trajectories_including_large_holdout": test_count,
        "files": files,
        "checks": checks,
        "physics_limits": [
            "phase-1 sample counts are sufficient for model development, not percent-level rare-event probabilities",
            "0.05 fs was selected conservatively because 0.1 vs 0.05 fs negative-endpoint fractions were not resolved to absolute 0.05 precision",
            "mechanism labels use a spatial-std threshold of 0.25 and still require threshold-sensitivity analysis",
            "the suite contains only the Gomonay d-wave altermagnet at 5 K and three drive values",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
