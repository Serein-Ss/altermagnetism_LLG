"""Lazy schema-v2 HDF5 loader with shape-bucketed variable-size batches."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}


class ScalablePathDataset(Dataset):
    def __init__(
        self,
        paths: list[str | Path],
        split: str,
        crop_size: int | None = None,
        *,
        random_crop: bool = True,
    ):
        if split not in SPLIT_CODES:
            raise ValueError(f"split must be one of {tuple(SPLIT_CODES)}")
        self.paths = [str(Path(path).resolve()) for path in paths]
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.index: list[tuple[int, str, int]] = []
        self._batch_keys: list[tuple[int, int, int, int, int]] = []
        self._files: dict[int, h5py.File] = {}
        for file_index, path in enumerate(self.paths):
            with h5py.File(path, "r") as h5:
                if int(h5.attrs.get("schema_version", 0)) != 2:
                    raise ValueError(f"{path} is not a schema-v2 scalable dataset")
                for name, group in h5.items():
                    if not isinstance(group, h5py.Group) or "spins" not in group:
                        continue
                    frames, sublattices, nx, ny, components = group["spins"].shape[1:]
                    if crop_size is not None:
                        if min(nx, ny) < crop_size:
                            raise ValueError(
                                f"crop_size={crop_size} exceeds {name} lattice {nx}x{ny}"
                            )
                        nx = ny = crop_size
                    batch_key = (frames, sublattices, nx, ny, components)
                    matches = np.flatnonzero(
                        group["split"][:] == SPLIT_CODES[split]
                    )
                    for match in matches:
                        self.index.append((file_index, name, int(match)))
                        self._batch_keys.append(batch_key)

    def __len__(self) -> int:
        return len(self.index)

    def batch_key(self, item: int) -> tuple[int, int, int, int, int]:
        """Shape key used to batch variable-size trajectories without padding."""
        return self._batch_keys[item]

    def _file(self, index: int) -> h5py.File:
        if index not in self._files:
            self._files[index] = h5py.File(self.paths[index], "r", swmr=True)
        return self._files[index]

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        file_index, system, trajectory = self.index[item]
        h5 = self._file(file_index)
        group = h5[system]
        spins = torch.from_numpy(group["spins"][trajectory])
        initial = torch.from_numpy(group["initial_spins"][trajectory])
        if self.crop_size is not None:
            nx, ny = spins.shape[-3:-1]
            shift_x = int(torch.randint(nx, ()).item()) if self.random_crop else 0
            shift_y = int(torch.randint(ny, ()).item()) if self.random_crop else 0
            spins = torch.roll(spins, (-shift_x, -shift_y), dims=(-3, -2))
            initial = torch.roll(initial, (-shift_x, -shift_y), dims=(-3, -2))
            spins = spins[..., : self.crop_size, : self.crop_size, :]
            initial = initial[..., : self.crop_size, : self.crop_size, :]
        weight = (
            float(group["sample_weight"][trajectory])
            if "sample_weight" in group
            else 1.0
        )
        return {
            "spins": spins,
            "initial": initial,
            "physical_time": torch.from_numpy(h5["time"][:].astype(np.float32)),
            "scalar_condition": torch.from_numpy(
                group["model_condition"][trajectory].astype(np.float32)
            ),
            "vector_condition": torch.from_numpy(
                group["vector_condition"][trajectory].astype(np.float32)
            ),
            "sample_weight": torch.tensor(weight, dtype=torch.float32),
            "trajectory_id": torch.tensor(
                int(group["trajectory_id"][trajectory]), dtype=torch.long
            ),
        }

    def close(self) -> None:
        for handle in self._files.values():
            handle.close()
        self._files.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_files"] = {}
        return state

    def __del__(self):
        self.close()


class SizeBucketBatchSampler(Sampler[list[int]]):
    """Yield homogeneous-shape batches without padding periodic lattices."""

    def __init__(
        self,
        dataset: ScalablePathDataset,
        batch_size: int,
        *,
        shuffle: bool,
        drop_last: bool = False,
        seed: int = 0,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        buckets: dict[tuple[int, int, int, int, int], list[int]] = {}
        for index in range(len(self.dataset)):
            buckets.setdefault(self.dataset.batch_key(index), []).append(index)

        batches = []
        for indices in buckets.values():
            if self.shuffle:
                order = torch.randperm(
                    len(indices), generator=generator
                ).tolist()
                indices = [indices[index] for index in order]
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in order]
        yield from batches

    def __len__(self) -> int:
        counts: dict[tuple[int, int, int, int, int], int] = {}
        for key in self.dataset._batch_keys:
            counts[key] = counts.get(key, 0) + 1
        if self.drop_last:
            return sum(count // self.batch_size for count in counts.values())
        return sum(
            (count + self.batch_size - 1) // self.batch_size
            for count in counts.values()
        )
