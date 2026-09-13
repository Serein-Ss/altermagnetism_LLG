"""Generate a compact, train-ready, full-spatial LLG benchmark dataset."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
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
    CommonSOT,
    GomonayModelAdapter,
    NishinoFreeMomentHamiltonian,
    UnifiedLLGSolver,
    collinear_state,
    common_observables,
)


SPLIT_NAMES = ("train", "validation", "test")
CONDITION_NAMES = (
    "temperature_raw",
    "alpha",
    "drive_damping_like_raw",
    "static_field_z_raw",
    "dt_raw",
    "duration_raw",
)


@dataclass(frozen=True)
class RunSpec:
    name: str
    system_id: int
    temperatures: tuple[float, ...]
    drives: tuple[float, ...]
    alpha: float
    dt: float
    equilibration_steps: int
    steps: int
    save_every: int
    pulse_end: float
    sublattices: int
    antiferromagnetic: bool
    unit_system: str


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate unified full-spatial LLG paths")
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "datasets" / "training_benchmark" / "llg_spatial_paths.h5",
    )
    p.add_argument("--trajectories-per-condition", type=int, default=20)
    p.add_argument("--size", type=int, nargs=2, default=(8, 8), metavar=("NX", "NY"))
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true")
    return p


def split_codes(count: int, seed: int) -> np.ndarray:
    if count % 10:
        raise ValueError("trajectories-per-condition must be divisible by 10")
    codes = np.empty(count, dtype=np.int8)
    order = np.random.default_rng(seed).permutation(count)
    n_train = 8 * count // 10
    n_validation = count // 10
    codes[order[:n_train]] = 0
    codes[order[n_train : n_train + n_validation]] = 1
    codes[order[n_train + n_validation :]] = 2
    return codes


def make_specs() -> list[RunSpec]:
    return [
        RunSpec(
            name="ordinary_free_moments",
            system_id=0,
            temperatures=(1.0, 2.0, 5.0),
            drives=(0.0, 0.0, 0.0),
            alpha=0.05,
            dt=0.005,
            equilibration_steps=10_000,
            steps=2_000,
            save_every=40,
            pulse_end=0.0,
            sublattices=1,
            antiferromagnetic=False,
            unit_system="Nishino2015_dimensionless_gamma_equals_kB_equals_1",
        ),
        RunSpec(
            name="conventional_afm_control",
            system_id=1,
            temperatures=(5.0, 5.0, 5.0),
            drives=(0.6, 0.8, 1.0),
            alpha=0.01,
            dt=1e-16,
            equilibration_steps=2_000,
            steps=10_000,
            save_every=200,
            pulse_end=0.5e-12,
            sublattices=2,
            antiferromagnetic=True,
            unit_system="SI_energy_J_field_T_time_s_temperature_K",
        ),
        RunSpec(
            name="d_wave_altermagnet",
            system_id=2,
            temperatures=(5.0, 5.0, 5.0),
            drives=(0.6, 0.8, 1.0),
            alpha=0.01,
            dt=1e-16,
            equilibration_steps=2_000,
            steps=10_000,
            save_every=200,
            pulse_end=0.5e-12,
            sublattices=2,
            antiferromagnetic=True,
            unit_system="SI_energy_J_field_T_time_s_temperature_K",
        ),
    ]


def make_model(spec: RunSpec):
    if spec.name == "ordinary_free_moments":
        return NishinoFreeMomentHamiltonian()
    return GomonayModelAdapter(alternating_exchange=spec.name == "d_wave_altermagnet")


def make_solver(spec: RunSpec, model, drive: float) -> UnifiedLLGSolver:
    sot = CommonSOT(
        damping_like=drive,
        start=0.0,
        end=spec.pulse_end if spec.pulse_end > 0 else float("inf"),
    )
    return UnifiedLLGSolver(model, alpha=spec.alpha, sot=sot)


def allocate_group(
    root: h5py.File,
    spec: RunSpec,
    num_conditions: int,
    paths_per_condition: int,
    size: tuple[int, int],
):
    total = num_conditions * paths_per_condition
    frames = spec.steps // spec.save_every + 1
    group = root.create_group(spec.name)
    spin_shape = (total, frames, spec.sublattices, *size, 3)
    spin_chunks = (1, frames, spec.sublattices, *size, 3)
    group.create_dataset(
        "spins",
        shape=spin_shape,
        dtype="f4",
        chunks=spin_chunks,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )
    for name in ("magnetization", "neel"):
        group.create_dataset(
            name,
            shape=(total, frames, 3),
            dtype="f4",
            chunks=(1, frames, 3),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
    group.create_dataset(
        "energy",
        shape=(total, frames),
        dtype="f8",
        chunks=(1, frames),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )
    group.create_dataset("condition", shape=(total, len(CONDITION_NAMES)), dtype="f8")
    group.create_dataset("condition_id", shape=(total,), dtype="i2")
    group.create_dataset("trajectory_id", shape=(total,), dtype="i8")
    group.create_dataset("split", shape=(total,), dtype="i1")
    group.create_dataset("rng_stream_seed", shape=(total,), dtype="i8")
    group.create_dataset("rng_lane", shape=(total,), dtype="i4")
    group.create_dataset(
        "time",
        data=np.arange(frames, dtype=np.float64) * spec.save_every * spec.dt,
    )
    group.attrs["system_id"] = spec.system_id
    group.attrs["unit_system"] = spec.unit_system
    group.attrs["condition_names_json"] = json.dumps(CONDITION_NAMES)
    group.attrs["run_spec_json"] = json.dumps(spec.__dict__)
    return group


@torch.no_grad()
def generate_condition(
    group,
    spec: RunSpec,
    model,
    temperature: float,
    drive: float,
    condition_id: int,
    paths_per_condition: int,
    size: tuple[int, int],
    stream_seed: int,
    device: str,
) -> None:
    solver = make_solver(spec, model, drive)
    dtype = torch.float64
    spins = collinear_state(
        paths_per_condition,
        spec.sublattices,
        *size,
        antiferromagnetic=spec.antiferromagnetic,
        device=device,
        dtype=dtype,
    )
    generator_device = device if device.startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(stream_seed)

    preparation = UnifiedLLGSolver(model, alpha=spec.alpha)
    for step in range(spec.equilibration_steps):
        spins = preparation.stochastic_heun_step(
            spins, step * spec.dt, spec.dt, temperature, generator
        )

    first = condition_id * paths_per_condition
    last = first + paths_per_condition
    frame = 0
    for step in range(spec.steps + 1):
        if step % spec.save_every == 0:
            magnetization, neel = common_observables(spins)
            group["spins"][first:last, frame] = spins.to(torch.float32).cpu().numpy()
            group["magnetization"][first:last, frame] = (
                magnetization.to(torch.float32).cpu().numpy()
            )
            group["neel"][first:last, frame] = neel.to(torch.float32).cpu().numpy()
            group["energy"][first:last, frame] = model.energy(spins).cpu().numpy()
            frame += 1
        if step != spec.steps:
            spins = solver.stochastic_heun_step(
                spins, step * spec.dt, spec.dt, temperature, generator
            )

    field_z = 2.0 if spec.name == "ordinary_free_moments" else 0.0
    condition = np.array(
        [temperature, spec.alpha, drive, field_z, spec.dt, spec.steps * spec.dt],
        dtype=np.float64,
    )
    group["condition"][first:last] = condition
    group["condition_id"][first:last] = condition_id
    group["trajectory_id"][first:last] = np.arange(first, last, dtype=np.int64)
    group["split"][first:last] = split_codes(paths_per_condition, stream_seed + 17)
    group["rng_stream_seed"][first:last] = stream_seed
    group["rng_lane"][first:last] = np.arange(paths_per_condition, dtype=np.int32)


def main() -> None:
    args = parser().parse_args()
    if args.trajectories_per_condition < 10 or args.trajectories_per_condition % 10:
        raise ValueError("trajectories-per-condition must be a positive multiple of 10")
    if min(args.size) < 1:
        raise ValueError("lattice dimensions must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --overwrite")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    specs = make_specs()
    manifest = {
        "schema_version": 1,
        "dataset_role": "trainable_pilot_benchmark_not_production_scale",
        "full_spatial_state": True,
        "spin_layout": "trajectory,time,sublattice,x,y,xyz",
        "storage": "HDF5 float32 spins, per-trajectory gzip chunks, lazy loading",
        "split": "80:10:10 within every system-condition; whole trajectories only",
        "split_codes": {"0": "train", "1": "validation", "2": "test"},
        "selection": "none; every generated trajectory retained with equal weight",
        "rng": (
            "one seeded PyTorch counter stream per system-condition; each trajectory "
            "occupies a distinct tensor lane and receives disjoint Gaussian draws"
        ),
        "thermal_noise": {
            "variable": "B_th in each model's field units",
            "mean": "zero",
            "covariance": (
                "<B_mu,i(t) B_nu,j(t')> = 2 alpha k_B T/(gamma moment) "
                "delta_mu,nu delta_i,j delta(t-t')"
            ),
            "energy_variable_conversion": (
                "R=moment*B_th gives 2 alpha (moment/gamma) k_B T; "
                "a^2 delta(r-r') becomes delta_i,j for a cell of area a^2"
            ),
        },
        "device": args.device,
        "size": list(args.size),
        "trajectories_per_condition": args.trajectories_per_condition,
        "systems": {},
    }

    with h5py.File(partial, "w") as h5:
        h5.attrs["schema_version"] = 1
        h5.attrs["status"] = "writing"
        for spec in specs:
            model = make_model(spec)
            group = allocate_group(
                h5,
                spec,
                len(spec.temperatures),
                args.trajectories_per_condition,
                tuple(args.size),
            )
            group.attrs["model_metadata_json"] = json.dumps(model.metadata())
            for condition_id, (temperature, drive) in enumerate(
                zip(spec.temperatures, spec.drives)
            ):
                stream_seed = args.seed + spec.system_id * 100_000 + condition_id * 1_000
                generate_condition(
                    group,
                    spec,
                    model,
                    temperature,
                    drive,
                    condition_id,
                    args.trajectories_per_condition,
                    tuple(args.size),
                    stream_seed,
                    args.device,
                )
                print(
                    f"generated {spec.name} condition {condition_id}: "
                    f"T={temperature}, drive={drive}"
                )
            manifest["systems"][spec.name] = {
                "system_id": spec.system_id,
                "model": model.metadata(),
                "run_spec": spec.__dict__,
                "conditions": [
                    {"condition_id": i, "temperature": t, "drive": d}
                    for i, (t, d) in enumerate(zip(spec.temperatures, spec.drives))
                ],
            }
        h5.attrs["status"] = "complete"
        h5.attrs["manifest_json"] = json.dumps(manifest)
        h5.flush()
    os.replace(partial, args.output)
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest["file"] = args.output.name
    manifest["bytes"] = args.output.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "bytes": manifest["bytes"]}, indent=2))


if __name__ == "__main__":
    main()
