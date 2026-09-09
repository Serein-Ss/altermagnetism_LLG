"""Generate unbiased Bauer-chain path ensembles using the paper's LL convention."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "core"))

from unified_llg import (  # noqa: E402
    BauerChainParameters,
    BauerOpenChainHamiltonian,
    UnifiedLLGSolver,
)


PROFILES = {
    "accelerated_pilot": {
        "length": 40,
        "temperature": 0.20,
        "duration": 500.0,
        "dt": 0.02,
        "trajectories": 40,
        "save_every": 25,
    },
    "published_fig2": {
        "length": 100,
        "temperature": 0.11,
        "duration": 750000.0,
        "dt": 0.02,
        "trajectories": 10,
        "save_every": 250,
    },
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unbiased Bauer 2011 chain paths")
    p.add_argument("--profile", choices=tuple(PROFILES), default="accelerated_pilot")
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p


def split_codes(count: int, seed: int) -> np.ndarray:
    if count % 10:
        raise ValueError("trajectory count must be divisible by 10")
    order = np.random.default_rng(seed).permutation(count)
    result = np.empty(count, dtype=np.int8)
    result[order[: 8 * count // 10]] = 0
    result[order[8 * count // 10 : 9 * count // 10]] = 1
    result[order[9 * count // 10 :]] = 2
    return result


@torch.no_grad()
def main() -> None:
    args = parser().parse_args()
    config = PROFILES[args.profile]
    output = args.output or (
        ROOT
        / "data"
        / "path_literature_validation"
        / f"bauer2011_{args.profile}_paths.h5"
    )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        partial.unlink()

    length = int(config["length"])
    trajectories = int(config["trajectories"])
    dt = float(config["dt"])
    steps = round(float(config["duration"]) / dt)
    save_every = int(config["save_every"])
    frames = steps // save_every + 1
    temperature = float(config["temperature"])
    alpha = 0.1
    model = BauerOpenChainHamiltonian(BauerChainParameters(anisotropy_k=0.1))
    solver = UnifiedLLGSolver(model, alpha=alpha, equation="bauer_ll")
    spins = torch.zeros(
        (trajectories, 1, length, 1, 3),
        dtype=torch.float64,
        device=args.device,
    )
    spins[..., 2] = 1.0
    generator_device = args.device if args.device.startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)

    manifest = {
        "schema_version": 2,
        "profile": args.profile,
        "scientific_role": (
            "pipeline_and_mechanism_pilot_not_a_quantitative_Fig2_reproduction"
            if args.profile == "accelerated_pilot"
            else "published_Fig2_conditions_with_an_explicit_unreported_timestep_choice"
        ),
        "source": model.metadata(),
        "equation": (
            "Bauer Eq.(2): LL precession with noise, deterministic damping; "
            "Stratonovich stochastic Heun"
        ),
        "thermal_noise": "epsilon^2=2 lambda k_B T; isotropic/site/time independent",
        "parameters": {
            **config,
            "alpha_lambda": alpha,
            "steps": steps,
            "seed": args.seed,
        },
        "selection": "none; every trajectory has unit sample weight",
        "split": "whole trajectories 80:10:10",
        "limitations": [
            "The paper does not report its numerical timestep or random seed.",
            "The accelerated profile changes length and temperature and is not Fig. 2 reproduction.",
        ],
    }

    with h5py.File(partial, "w") as h5:
        h5.attrs["manifest_json"] = json.dumps(manifest)
        h5.attrs["status"] = "writing"
        h5.create_dataset(
            "spins",
            shape=(trajectories, frames, 1, length, 1, 3),
            dtype="f4",
            chunks=(1, min(frames, 128), 1, length, 1, 3),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        h5.create_dataset("magnetization_z", shape=(trajectories, frames), dtype="f4")
        h5.create_dataset("energy", shape=(trajectories, frames), dtype="f8")
        h5.create_dataset("time", data=np.arange(frames) * save_every * dt)
        h5.create_dataset("split", data=split_codes(trajectories, args.seed + 17))
        h5.create_dataset("sample_weight", data=np.ones(trajectories, dtype=np.float32))
        h5.create_dataset("trajectory_id", data=np.arange(trajectories, dtype=np.int64))
        h5.create_dataset("rng_lane", data=np.arange(trajectories, dtype=np.int32))

        frame = 0
        for step in range(steps + 1):
            if step % save_every == 0:
                h5["spins"][:, frame] = spins.to(torch.float32).cpu().numpy()
                h5["magnetization_z"][:, frame] = (
                    spins[..., 2].mean(dim=(1, 2, 3)).to(torch.float32).cpu().numpy()
                )
                h5["energy"][:, frame] = model.energy(spins).cpu().numpy()
                frame += 1
            if step != steps:
                spins = solver.stochastic_heun_step(
                    spins, step * dt, dt, temperature, generator
                )
        h5.attrs["status"] = "complete"
        h5.flush()
    os.replace(partial, output)
    manifest["file"] = output.name
    manifest["bytes"] = output.stat().st_size
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "bytes": manifest["bytes"]}, indent=2))


if __name__ == "__main__":
    main()
