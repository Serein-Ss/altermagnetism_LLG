"""Lazy PyTorch loader for the unified LLG HDF5 dataset."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}


class LLGTrajectoryDataset(Dataset):
    """Return full trajectories while opening HDF5 lazily per worker process."""

    def __init__(
        self,
        path: str | Path,
        split: str,
        *,
        systems: tuple[str, ...] | None = None,
        pad_sublattices_to: int = 2,
    ):
        if split not in SPLIT_CODE:
            raise ValueError(f"split must be one of {tuple(SPLIT_CODE)}")
        self.path = str(Path(path).resolve())
        self.split = split
        self.pad_sublattices_to = pad_sublattices_to
        self._h5 = None
        self.index: list[tuple[str, int]] = []
        with h5py.File(self.path, "r") as h5:
            selected = tuple(systems) if systems is not None else tuple(
                name
                for name, obj in h5.items()
                if isinstance(obj, h5py.Group) and "split" in obj
            )
            for system in selected:
                codes = h5[system]["split"][:]
                self.index.extend(
                    (system, int(i)) for i in np.flatnonzero(codes == SPLIT_CODE[split])
                )

    def __len__(self) -> int:
        return len(self.index)

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r", swmr=True)
        return self._h5

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str]:
        system, local_id = self.index[item]
        group = self._file()[system]
        spins = torch.from_numpy(group["spins"][local_id])
        sublattices = spins.shape[1]
        if sublattices > self.pad_sublattices_to:
            raise ValueError("pad_sublattices_to is smaller than stored sublattice count")
        mask = torch.zeros(self.pad_sublattices_to, dtype=torch.bool)
        mask[:sublattices] = True
        if sublattices < self.pad_sublattices_to:
            padded = torch.zeros(
                (spins.shape[0], self.pad_sublattices_to, *spins.shape[2:]),
                dtype=spins.dtype,
            )
            padded[:, :sublattices] = spins
            spins = padded
        condition_key = "model_condition" if "model_condition" in group else "condition"
        condition = torch.from_numpy(group[condition_key][local_id].astype(np.float32))
        time_source = group["time"] if "time" in group else self._file()["time"]
        time = torch.from_numpy(time_source[:].astype(np.float32))
        result = {
            "spins": spins,
            "sublattice_mask": mask,
            "condition": condition,
            "time": time,
            "system_id": torch.tensor(int(group.attrs["system_id"]), dtype=torch.long),
            "trajectory_id": torch.tensor(
                int(group["trajectory_id"][local_id]) if "trajectory_id" in group else local_id,
                dtype=torch.long,
            ),
            "system": system,
        }
        for name in (
            "physical_condition",
            "vector_condition",
            "sample_weight",
            "trajectory_seed",
            "condition_id",
            "split",
        ):
            if name in group:
                value = np.asarray(group[name][local_id])
                if np.issubdtype(value.dtype, np.floating):
                    value = value.astype(np.float32)
                result[name] = torch.from_numpy(value) if value.ndim else torch.tensor(value.item())
        return result

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self):
        self.close()


def describe(path: str | Path) -> dict:
    with h5py.File(path, "r") as h5:
        return json.loads(h5.attrs["manifest_json"])
