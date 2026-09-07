from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from altermagnetism_LLG.model.data import (  # noqa: E402
    ScalablePathDataset,
    SizeBucketBatchSampler,
)


def write_dataset(path: Path, size: int, trajectory_start: int) -> None:
    paths, frames = 2, 3
    with h5py.File(path, "w") as h5:
        h5.attrs["schema_version"] = 2
        h5.create_dataset(
            "time", data=np.array([0.0, 0.25, 1.0], dtype=np.float64)
        )
        group = h5.create_group("d_wave_altermagnet")
        group.create_dataset("split", data=np.zeros(paths, dtype=np.int8))
        group.create_dataset(
            "trajectory_id",
            data=np.arange(trajectory_start, trajectory_start + paths),
        )
        group.create_dataset(
            "model_condition", data=np.ones((paths, 10), dtype=np.float32)
        )
        group.create_dataset(
            "vector_condition", data=np.ones((paths, 5, 3), dtype=np.float32)
        )
        group.create_dataset(
            "sample_weight", data=np.ones(paths, dtype=np.float32)
        )
        spins = np.zeros(
            (paths, frames, 2, size, size, 3), dtype=np.float32
        )
        spins[..., 2] = 1.0
        group.create_dataset("spins", data=spins)
        group.create_dataset("initial_spins", data=spins[:, 0])


def test_variable_sizes_stay_lazy_and_form_homogeneous_batches(tmp_path):
    small = tmp_path / "small.h5"
    large = tmp_path / "large.h5"
    write_dataset(small, 4, 0)
    write_dataset(large, 6, 10)

    dataset = ScalablePathDataset(
        [small, large], "train", random_crop=False
    )
    assert dataset._files == {}
    assert len(dataset) == 4
    sampler = SizeBucketBatchSampler(
        dataset, 2, shuffle=False
    )
    batches = list(sampler)
    assert len(batches) == 2
    assert all(
        len({dataset.batch_key(index) for index in batch}) == 1
        for batch in batches
    )

    loader = DataLoader(dataset, batch_sampler=sampler)
    spatial_shapes = {
        tuple(batch["spins"].shape[-3:-1]) for batch in loader
    }
    assert spatial_shapes == {(4, 4), (6, 6)}
    assert dataset[0]["physical_time"].tolist() == [
        0.0,
        0.25,
        1.0,
    ]
    assert dataset._files
    dataset.close()


def test_fixed_crop_allows_mixed_source_sizes(tmp_path):
    small = tmp_path / "small.h5"
    large = tmp_path / "large.h5"
    write_dataset(small, 4, 0)
    write_dataset(large, 6, 10)
    dataset = ScalablePathDataset(
        [small, large], "train", crop_size=4, random_crop=False
    )
    assert {dataset.batch_key(index) for index in range(len(dataset))} == {
        (3, 2, 4, 4, 3)
    }
    loader = DataLoader(
        dataset,
        batch_sampler=SizeBucketBatchSampler(
            dataset, 4, shuffle=False
        ),
    )
    batch = next(iter(loader))
    assert tuple(batch["spins"].shape) == (4, 3, 2, 4, 4, 3)
    dataset.close()
