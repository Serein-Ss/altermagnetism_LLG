"""Minimal example: load one training batch without reading the full HDF5 file."""
from pathlib import Path
import sys

from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "datasets"))

from hdf5_trajectory_dataset import LLGTrajectoryDataset  # noqa: E402


def main() -> None:
    path = ROOT / "data" / "training_benchmark" / "llg_spatial_paths.h5"
    dataset = LLGTrajectoryDataset(path, "train")
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)))
    print("spins:", tuple(batch["spins"].shape))
    print("sublattice_mask:", tuple(batch["sublattice_mask"].shape))
    print("condition:", tuple(batch["condition"].shape))
    print("time:", tuple(batch["time"].shape))
    print("system_id:", batch["system_id"].tolist())
    dataset.close()


if __name__ == "__main__":
    main()
