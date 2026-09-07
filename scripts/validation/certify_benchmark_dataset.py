"""Certify schema, numerics, split integrity, and literature observables."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "core"))
sys.path.insert(0, str(ROOT / "scripts" / "datasets"))

from hdf5_trajectory_dataset import LLGTrajectoryDataset  # noqa: E402
from unified_llg import NishinoFreeMomentHamiltonian  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Certify the unified LLG HDF5 dataset")
    p.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "training_benchmark" / "llg_spatial_paths.h5",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "training_benchmark" / "benchmark_certification.json",
    )
    return p


def split_counts(group, condition_id: int) -> list[int]:
    mask = group["condition_id"][:] == condition_id
    return np.bincount(group["split"][:][mask], minlength=3).astype(int).tolist()


def spatial_incoherence(spins: np.ndarray) -> np.ndarray:
    local_neel = 0.5 * (spins[:, :, 0] - spins[:, :, 1])
    numerator = np.linalg.norm(local_neel.mean(axis=(2, 3)), axis=-1)
    denominator = np.linalg.norm(local_neel, axis=-1).mean(axis=(2, 3))
    return 1.0 - numerator / np.clip(denominator, 1e-12, None)


def main() -> None:
    args = parser().parse_args()
    metrics: dict = {
        "dataset": str(args.input.resolve()),
        "certification_scope": (
            "schema/numerics/split + Nishino exact equilibrium + existing Gomonay spin-wave"
        ),
        "systems": {},
    }
    all_finite = True
    max_norm_error = 0.0
    split_exact = True
    raw_spin_bytes = 0

    with h5py.File(args.input, "r") as h5:
        metrics["hdf5_status"] = h5.attrs.get("status", "missing")
        for name, group in h5.items():
            spins = group["spins"][:]
            raw_spin_bytes += spins.nbytes
            finite = bool(np.isfinite(spins).all())
            norm_error = float(np.max(np.abs(np.linalg.norm(spins, axis=-1) - 1.0)))
            all_finite &= finite
            max_norm_error = max(max_norm_error, norm_error)
            condition_metrics = []
            for condition_id in np.unique(group["condition_id"][:]):
                mask = group["condition_id"][:] == condition_id
                counts = split_counts(group, int(condition_id))
                split_exact &= counts == [16, 2, 2]
                entry = {
                    "condition_id": int(condition_id),
                    "condition": group["condition"][mask][0].tolist(),
                    "split_counts_train_validation_test": counts,
                }
                if name == "ordinary_free_moments":
                    temperature = float(group["condition"][mask][0, 0])
                    per_path = group["magnetization"][:][mask, :, 2].mean(axis=1)
                    measured = float(per_path.mean())
                    sem = float(per_path.std(ddof=1) / math.sqrt(len(per_path)))
                    exact = NishinoFreeMomentHamiltonian().exact_magnetization_z(temperature)
                    entry.update(
                        {
                            "temperature": temperature,
                            "measured_mean_mz": measured,
                            "trajectory_sem": sem,
                            "exact_langevin_mz": exact,
                            "absolute_error": abs(measured - exact),
                        }
                    )
                else:
                    neel_z = group["neel"][:][mask, :, 2]
                    crossed = np.any(neel_z < 0, axis=1)
                    final_negative = neel_z[:, -1] < 0
                    incoherence = spatial_incoherence(spins[mask])
                    entry.update(
                        {
                            "drive_T": float(group["condition"][mask][0, 2]),
                            "no_crossing_fraction": float((~crossed).mean()),
                            "cross_and_return_fraction": float((crossed & ~final_negative).mean()),
                            "final_negative_fraction": float(final_negative.mean()),
                            "mean_final_neel_z": float(neel_z[:, -1].mean()),
                            "std_final_neel_z": float(neel_z[:, -1].std()),
                            "mean_peak_spatial_incoherence": float(incoherence.max(axis=1).mean()),
                        }
                    )
                condition_metrics.append(entry)
            metrics["systems"][name] = {
                "shape": list(spins.shape),
                "finite": finite,
                "max_spin_norm_error": norm_error,
                "conditions": condition_metrics,
            }

    ordinary_errors = [
        c["absolute_error"]
        for c in metrics["systems"]["ordinary_free_moments"]["conditions"]
    ]
    spinwave_path = ROOT / "data" / "literature_validation" / "spinwave_validation.json"
    spinwave = json.loads(spinwave_path.read_text(encoding="utf-8"))
    loaders = {}
    for split in ("train", "validation", "test"):
        dataset = LLGTrajectoryDataset(args.input, split)
        sample = dataset[0]
        loaders[split] = {
            "trajectories": len(dataset),
            "sample_spin_shape_after_padding": list(sample["spins"].shape),
        }
        dataset.close()

    metrics["storage"] = {
        "file_bytes": args.input.stat().st_size,
        "raw_spin_bytes_only": raw_spin_bytes,
        "file_to_raw_spin_ratio": args.input.stat().st_size / raw_spin_bytes,
        "loader_smoke_test": loaders,
    }
    metrics["checks"] = {
        "hdf5_complete": metrics["hdf5_status"] == "complete",
        "all_spins_finite": all_finite,
        "max_norm_error_below_1e-6": max_norm_error < 1e-6,
        "split_is_exact_8_1_1_per_condition": split_exact,
        "nishino_langevin_max_abs_error_below_0p03": max(ordinary_errors) < 0.03,
        "gomonay_spinwave_existing_validation_pass": spinwave["status"] == "pass",
    }
    metrics["certification_tiers"] = {
        "ordinary_free_moments": (
            "analytic equilibrium observable certified at three temperatures; "
            "paper parameters used, pilot ensemble rather than exact paper run length"
        ),
        "d_wave_altermagnet": (
            "published Hamiltonian and spin-wave frequencies certified; finite-temperature "
            "switching remains a benchmark dataset, not a literature reproduction"
        ),
        "conventional_afm_control": (
            "numerically certified J_tilde=0 ablation of the published model; not a separate "
            "published material parameterization"
        ),
    }
    metrics["overall_status"] = (
        "pass_as_partially_literature_certified_training_benchmark"
        if all(metrics["checks"].values())
        else "fail"
    )
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
