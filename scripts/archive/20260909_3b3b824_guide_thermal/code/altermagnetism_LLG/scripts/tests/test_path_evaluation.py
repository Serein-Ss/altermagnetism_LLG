from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from altermagnetism_LLG.model.evaluation import (
    BasinCalibration,
    calibrate_basins,
    path_descriptors,
    summarize,
)


CALIBRATION = BasinCalibration(
    negative_threshold=-0.6,
    positive_threshold=0.6,
    residence_ps=0.2,
    component_means=(-0.9, 0.0, 0.9),
    component_std=(0.05, 0.2, 0.05),
    component_weights=(0.4, 0.2, 0.4),
    trajectories=100,
    sources=("synthetic",),
)


def spins_from_order(order: np.ndarray) -> np.ndarray:
    spins = np.zeros((len(order), 2, 2, 2, 3), dtype=np.float64)
    spins[:, 0, ..., 2] = order[:, None, None]
    spins[:, 1, ..., 2] = -order[:, None, None]
    transverse = np.sqrt(1.0 - order**2)
    spins[:, 0, ..., 0] = transverse[:, None, None]
    spins[:, 1, ..., 0] = -transverse[:, None, None]
    return spins


def describe(order: list[float]) -> dict:
    time = np.arange(len(order), dtype=np.float64) * 0.1e-12
    return path_descriptors(
        spins_from_order(np.asarray(order)), time, CALIBRATION
    )


def test_negative_endpoint_is_not_automatically_a_committed_switch():
    row = describe([0.9, 0.7, 0.2, -0.1, -0.2])
    assert row["crossed_zero"]
    assert row["negative_endpoint"]
    assert not row["committed_switch"]
    assert row["unresolved_transition"]


def test_committed_switch_requires_terminal_residence_in_reverse_basin():
    row = describe([0.9, 0.2, -0.7, -0.8, -0.9])
    assert row["negative_endpoint"]
    assert row["committed_switch"]
    assert row["outcome"] == "committed_switch"


def test_crossing_return_requires_return_to_positive_basin():
    row = describe([0.9, -0.1, 0.7, 0.8, 0.9])
    assert row["crossed_zero"]
    assert not row["negative_endpoint"]
    assert row["crossing_return"]
    assert row["outcome"] == "crossing_return"


def test_calibration_does_not_split_one_stable_basin(tmp_path):
    path = tmp_path / "two_basins.h5"
    terminal = np.concatenate((
        np.linspace(-0.98, -0.95, 30),
        np.linspace(-0.93, -0.91, 10),
        np.linspace(0.95, 0.99, 50),
    ))
    with h5py.File(path, "w") as h5:
        h5.create_dataset("time", data=np.linspace(0.0, 0.1e-12, 11))
        group = h5.create_group("system")
        neel = np.zeros((len(terminal), 11, 3))
        neel[:, :, 2] = terminal[:, None]
        group.create_dataset("neel", data=neel)
        split = np.zeros(len(terminal), dtype=np.int8)
        split[-9:] = 2
        group.create_dataset("split", data=split)

    calibration = calibrate_basins([path], "system")
    assert calibration.component_count == 2
    assert calibration.negative_threshold < -0.5
    assert calibration.positive_threshold > 0.5


def test_summary_keeps_each_path_for_distribution_plots():
    rows = [
        describe([0.9, 0.8, 0.7]),
        describe([0.9, 0.2, -0.8]),
    ]
    summary = summarize(rows)
    assert len(summary["neel_z_paths"]) == 2
    assert len(summary["spatial_std_paths"]) == 2


def test_negative_initial_commits_only_after_crossing_to_positive_basin():
    row = describe([-0.9, -0.7, 0.7, 0.8, 0.9])
    assert row["initial_basin"] == "negative"
    assert row["crossed_zero"]
    assert not row["negative_endpoint"]
    assert row["committed_switch"]
    assert row["outcome"] == "committed_switch"


def test_negative_initial_without_crossing_stays_in_original_basin():
    row = describe([-0.9, -0.8, -0.7])
    assert row["initial_basin"] == "negative"
    assert not row["crossed_zero"]
    assert not row["committed_switch"]
    assert not row["unresolved_transition"]
    assert row["outcome"] == "no_crossing"
