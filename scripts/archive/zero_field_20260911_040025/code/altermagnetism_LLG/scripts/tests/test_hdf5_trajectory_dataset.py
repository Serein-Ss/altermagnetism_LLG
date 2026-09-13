from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "datasets"))

from hdf5_trajectory_dataset import LLGTrajectoryDataset  # noqa: E402


def test_schema_v2_root_time_and_conditions_are_loaded(tmp_path):
    path = tmp_path / "schema_v2.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("time", data=np.array([0.0, 1.0, 2.0], dtype=np.float64))
        group = h5.create_group("d_wave_altermagnet")
        group.attrs["system_id"] = 2
        group.create_dataset("split", data=np.array([0, 2], dtype=np.int8))
        group.create_dataset("trajectory_id", data=np.array([17, 18], dtype=np.int64))
        group.create_dataset("trajectory_seed", data=np.array([101, 102], dtype=np.int64))
        group.create_dataset("condition_id", data=np.array([0, 0], dtype=np.int64))
        group.create_dataset("sample_weight", data=np.ones(2, dtype=np.float32))
        group.create_dataset("model_condition", data=np.ones((2, 10), dtype=np.float32))
        group.create_dataset("physical_condition", data=np.ones((2, 8), dtype=np.float32))
        group.create_dataset("vector_condition", data=np.ones((2, 5, 3), dtype=np.float32))
        spins = np.zeros((2, 3, 2, 2, 2, 3), dtype=np.float32)
        spins[..., 2] = 1.0
        group.create_dataset("spins", data=spins)

    dataset = LLGTrajectoryDataset(path, "train")
    sample = dataset[0]
    assert len(dataset) == 1
    assert tuple(sample["spins"].shape) == (3, 2, 2, 2, 3)
    assert tuple(sample["condition"].shape) == (10,)
    assert tuple(sample["physical_condition"].shape) == (8,)
    assert tuple(sample["vector_condition"].shape) == (5, 3)
    assert sample["time"].tolist() == [0.0, 1.0, 2.0]
    assert sample["trajectory_id"].item() == 17
    assert sample["split"].item() == 0
    dataset.close()
