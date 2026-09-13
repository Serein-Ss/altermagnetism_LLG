"""Visualize the first-loop simulator references, without changing source data."""
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/research/runs/bauer_first_loop/20260913_v1"
DEST = ROOT / "assets/research/bauer_L25_first_loop/20260913_v1/reference_data"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    space, grids = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    summary = {"source": "Simulator reference data, not model-generated trajectories",
               "temperature": "theta = k_B T / J", "time": "tau = t J / hbar",
               "selection": "Overview: all 32 paths from initial group 0; histogram: all 128 paths. Spatial examples: group 0 path 0 and first negative-endpoint path (else path 1). Initial groups are not paired across temperatures.",
               "runs": []}
    for row, theta in enumerate((0.11, 0.13)):
        path = SOURCE / f"reference_T{theta:.2f}.h5"
        before = digest(path)
        with h5py.File(path, "r") as handle:
            assert handle.attrs["complete"]
            spins = handle["spins"][:]
            initial = handle["initial"][:]
            times = np.arange(spins.shape[2]) * float(handle.attrs["save_dt"])
        assert spins.shape == (4, 32, 201, 25, 3)
        assert np.isfinite(spins).all()
        assert np.allclose(spins[:, :, 0], initial[:, None], atol=1e-6)
        mz = spins[..., 2].mean(axis=-1)
        spatial_std = spins[..., 2].std(axis=-1)
        color = ("#0072B2", "#D55E00")[row]
        ax = axes[row, 0]
        ax.plot(times, mz[0].T, color=color, alpha=0.24, lw=0.7)
        ax.plot(times, mz[0].mean(axis=0), color="black", lw=2, label="32-path mean")
        ax.axhline(0, color="grey", lw=0.7, ls="--")
        ax.set(title=fr"$\theta={theta}$: same initial state, 32 paths",
               ylabel=r"$M_z = L^{-1}\sum_i s_{i,z}$", ylim=(-1.05, 1.05))
        ax.legend(fontsize=8)
        ax = axes[row, 1]
        ax.hist(mz[..., -1].ravel(), bins=np.linspace(-1, 1, 21), color=color,
                edgecolor="white")
        ax.axvline(0, color="grey", lw=0.7, ls="--")
        ax.set(title="Endpoint distribution: all 128 paths",
               xlabel=r"$M_z(16000)$", ylabel="Number of paths", xlim=(-1, 1))
        ax = axes[row, 2]
        ax.plot(times, spatial_std[0].T, color=color, alpha=0.24, lw=0.7)
        ax.plot(times, spatial_std[0].mean(axis=0), color="black", lw=2)
        ax.set(title="Spatial nonuniformity: same 32 paths",
               ylabel=r"$\mathrm{std}_{i}(s_{i,z})$", ylim=(0, 1))
        for col in (0, 2):
            axes[row, col].set(xlabel=r"Reduced time $\tau=tJ/\hbar$", xlim=(0, times[-1]))
        negative = np.flatnonzero(mz[0, :, -1] < 0)
        second = next((int(i) for i in negative if i != 0), 1)
        for col, index in enumerate((0, second)):
            im = grids[row, col].pcolormesh(times, np.arange(1, 26),
                     spins[0, index, :, :, 2].T, shading="nearest", cmap="RdBu_r",
                     vmin=-1, vmax=1, rasterized=True)
            grids[row, col].set(title=fr"$\theta={theta}$; initial 0, path {index}; endpoint $M_z={mz[0,index,-1]:.2f}$",
                              xlabel=r"Reduced time $\tau$", ylabel="Site (open chain)")
        assert digest(path) == before, "Source changed during visualization"
        summary["runs"].append({"theta": theta, "file": str(path.relative_to(ROOT)),
            "sha256": before, "shape": list(spins.shape),
            "negative_endpoint_count": int((mz[..., -1] < 0).sum()),
            "ever_negative_at_saved_frames_count": int((mz < 0).any(axis=-1).sum()),
            "endpoint_mean": float(mz[..., -1].mean()),
            "max_saved_spin_norm_error": float(np.abs(np.linalg.norm(spins, axis=-1)-1).max()),
            "spatial_example_path_indices": [0, second]})
    fig.suptitle("Bauer L=25 reference trajectories | 4 initial states x 32 noise paths per temperature", fontsize=14)
    space.suptitle("Individual trajectories: local spin z-component (selected examples, not frequency estimates)")
    space.colorbar(im, ax=grids.ravel().tolist(), label=r"$s_{i,z}$", shrink=0.8)
    for figure, name in ((fig, "reference_overview"), (space, "reference_spacetime")):
        figure.savefig(DEST / f"{name}.png", dpi=110)
        plt.close(figure)
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
