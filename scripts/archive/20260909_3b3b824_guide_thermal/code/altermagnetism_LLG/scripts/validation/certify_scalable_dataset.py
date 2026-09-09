"""Certify schema-v2 trajectory integrity and attach the convergence-gate status."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--convergence", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.input.with_suffix(".certification.json")
    checks = {}
    groups = {}
    with h5py.File(args.input, "r") as h5:
        checks["schema_v2"] = int(h5.attrs.get("schema_version", 0)) == 2
        checks["write_completed"] = h5.attrs.get("status", "") == "complete"
        checks["time_strictly_increasing"] = bool(np.all(np.diff(h5["time"][:]) > 0))
        checks["periodic_xy_recorded"] = json.loads(h5.attrs["periodic_axes_json"]) == ["x", "y"]
        for name, group in h5.items():
            if not isinstance(group, h5py.Group) or "spins" not in group:
                continue
            spins = group["spins"][:]
            initial = group["initial_spins"][:]
            seeds = group["trajectory_seed"][:]
            weights = group["sample_weight"][:] if "sample_weight" in group else np.asarray([])
            split_counts = {}
            split_ok = True
            split_policy = group.attrs.get("split_policy", "balanced_8_1_1")
            for condition_id in np.unique(group["condition_id"][:]):
                mask = group["condition_id"][:] == condition_id
                counts = np.bincount(group["split"][:][mask], minlength=3).tolist()
                split_counts[str(int(condition_id))] = counts
                total = int(mask.sum())
                expected = {
                    "balanced_8_1_1": [8 * total // 10, total // 10, total // 10],
                    "all_train": [total, 0, 0],
                    "all_validation": [0, total, 0],
                    "all_test": [0, 0, total],
                }[split_policy]
                split_ok &= counts == expected
            metrics = {
                "shape": list(spins.shape),
                "finite": bool(np.isfinite(spins).all()),
                "max_spin_norm_error": float(np.max(np.abs(np.linalg.norm(spins, axis=-1) - 1.0))),
                "initial_matches_first_frame_max_abs": float(np.max(np.abs(initial - spins[:, 0]))),
                "trajectory_seeds_unique": len(np.unique(seeds)) == len(seeds),
                "all_sample_weights_equal_one": bool(len(weights) == len(seeds) and np.all(weights == 1.0)),
                "split_counts_by_condition": split_counts,
                "split_policy": split_policy,
                "split_matches_policy": bool(split_ok),
                "has_dimensionless_model_conditions": "model_condition" in group,
                "has_joint_rotation_vectors": "vector_condition" in group,
            }
            groups[name] = metrics
        checks["all_groups_finite"] = all(item["finite"] for item in groups.values())
        checks["spin_norm_error_below_1e-6"] = all(item["max_spin_norm_error"] < 1e-6 for item in groups.values())
        checks["initial_matches_first_frame"] = all(item["initial_matches_first_frame_max_abs"] == 0.0 for item in groups.values())
        checks["trajectory_seeds_independently_replayable"] = all(item["trajectory_seeds_unique"] for item in groups.values())
        checks["unbiased_equal_weights"] = all(item["all_sample_weights_equal_one"] for item in groups.values())
        checks["split_matches_declared_policy"] = all(item["split_matches_policy"] for item in groups.values())
        checks["conditioning_complete"] = all(item["has_dimensionless_model_conditions"] and item["has_joint_rotation_vectors"] for item in groups.values())

    manifest_path = args.input.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    checks["manifest_sha256_matches"] = manifest.get("sha256") == sha256(args.input)
    convergence = None
    if args.convergence is not None:
        convergence = json.loads(args.convergence.read_text(encoding="utf-8"))
    checks["convergence_gate_passed"] = convergence is not None and convergence.get("status") == "certified_for_nonuniform_path_training"
    integrity_keys = [key for key in checks if key != "convergence_gate_passed"]
    integrity_pass = all(checks[key] for key in integrity_keys)
    report = {
        "dataset": str(args.input.resolve()),
        "integrity_status": "pass" if integrity_pass else "fail",
        "physics_training_status": "ready" if integrity_pass and checks["convergence_gate_passed"] else "not_ready",
        "groups": groups,
        "checks": checks,
        "claim_boundary": "integrity pass means the file is numerically usable; only a passed convergence gate permits nonuniform-mechanism training claims",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
