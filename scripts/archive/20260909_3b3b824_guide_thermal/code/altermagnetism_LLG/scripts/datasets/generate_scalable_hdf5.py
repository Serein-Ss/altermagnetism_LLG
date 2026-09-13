"""Generate replayable, size-aware AFM/altermagnet full-path datasets (schema v2)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    UnifiedLLGSolver,
    collinear_state,
    common_observables,
)


PHYSICAL_CONDITION_NAMES = (
    "temperature_K",
    "alpha",
    "drive_damping_like_T",
    "drive_field_like_T",
    "dt_s",
    "saved_dt_s",
    "duration_s",
    "pulse_duration_s",
)
MODEL_CONDITION_NAMES = (
    "kBT_over_J1",
    "alpha",
    "drive_dl_over_exchange_field",
    "drive_fl_over_exchange_field",
    "saved_dt_gamma_exchange_field",
    "duration_gamma_exchange_field",
    "pulse_fraction",
    "J2_over_J1",
    "Jtilde_over_J1",
    "K_over_J1",
)
VECTOR_CONDITION_NAMES = (
    "sot_polarization",
    "external_field_direction",
    "crystal_x",
    "crystal_y",
    "crystal_z",
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system", choices=("altermagnet", "afm_control", "both"), default="altermagnet")
    p.add_argument("--size", type=int, nargs=2, default=(48, 48), metavar=("NX", "NY"))
    p.add_argument("--temperature", type=float, nargs="+", default=(5.0,))
    p.add_argument("--drive", type=float, nargs="+", default=(0.6, 0.8, 1.0))
    p.add_argument("--paths-per-condition", type=int, default=100)
    p.add_argument("--dt", type=float, default=1e-16)
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--equilibration-steps", type=int, default=2_000)
    p.add_argument("--pulse-duration", type=float, default=0.5e-12)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--initial-state", choices=("collinear", "wall_pair"), default="collinear")
    p.add_argument("--wall-width-nm", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--dataset-role", default="size_scalable_full_spatial_paths")
    p.add_argument(
        "--split-policy",
        choices=("balanced_8_1_1", "all_train", "all_validation", "all_test"),
        default="balanced_8_1_1",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p


def split_codes(count: int, seed: int, policy: str = "balanced_8_1_1") -> np.ndarray:
    fixed = {"all_train": 0, "all_validation": 1, "all_test": 2}
    if policy in fixed:
        return np.full(count, fixed[policy], dtype=np.int8)
    if policy != "balanced_8_1_1":
        raise ValueError(f"unknown split policy: {policy}")
    if count < 10 or count % 10:
        raise ValueError("paths-per-condition must be a positive multiple of 10")
    result = np.empty(count, dtype=np.int8)
    order = np.random.default_rng(seed).permutation(count)
    result[order[: 8 * count // 10]] = 0
    result[order[8 * count // 10 : 9 * count // 10]] = 1
    result[order[9 * count // 10 :]] = 2
    return result


def wall_pair_state(
    batch: int,
    nx: int,
    ny: int,
    width_cells: float,
    *,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a smooth periodic pair of 180-degree AFM walls along x."""
    if nx < math.ceil(4.0 * width_cells):
        raise ValueError("wall_pair requires nx >= 4 wall widths to limit wall overlap")
    x = torch.arange(nx, device=device, dtype=dtype)
    scale = max(2.0 * math.pi * width_cells / nx, 1e-6)
    nz = torch.tanh(torch.sin(2.0 * math.pi * x / nx) / scale)
    nx_component = torch.sqrt(torch.clamp(1.0 - nz.square(), min=0.0))
    neel = torch.zeros((nx, ny, 3), device=device, dtype=dtype)
    neel[..., 0] = nx_component[:, None]
    neel[..., 2] = nz[:, None]
    pair = torch.stack((neel, -neel), dim=0)
    return pair.unsqueeze(0).repeat(batch, 1, 1, 1, 1)


def model_conditions(model: GomonayModelAdapter, args, drive: float, temperature: float) -> np.ndarray:
    p = model.parameters
    exchange_field = p.j1 / p.mu_s
    exchange_frequency = p.gamma * exchange_field
    duration = args.steps * args.dt
    return np.asarray(
        [
            model.boltzmann * temperature / p.j1,
            args.alpha,
            drive / exchange_field,
            0.0,
            args.save_every * args.dt * exchange_frequency,
            duration * exchange_frequency,
            min(args.pulse_duration, duration) / duration,
            p.j2 / p.j1,
            p.j_tilde / p.j1,
            p.anisotropy / p.j1,
        ],
        dtype=np.float64,
    )


def condition_vectors() -> np.ndarray:
    inv_sqrt_two = 2.0**-0.5
    return np.asarray(
        [
            [inv_sqrt_two, inv_sqrt_two, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def allocate_group(h5: h5py.File, name: str, total: int, frames: int, size: tuple[int, int]):
    nx, ny = size
    group = h5.create_group(name)
    chunk = (1, min(frames, 8), 2, min(nx, 16), min(ny, 16), 3)
    group.create_dataset(
        "spins",
        shape=(total, frames, 2, nx, ny, 3),
        dtype="f4",
        chunks=chunk,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )
    group.create_dataset(
        "initial_spins",
        shape=(total, 2, nx, ny, 3),
        dtype="f4",
        chunks=(1, 2, min(nx, 16), min(ny, 16), 3),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )
    group.create_dataset("magnetization", shape=(total, frames, 3), dtype="f4")
    group.create_dataset("neel", shape=(total, frames, 3), dtype="f4")
    group.create_dataset("energy", shape=(total, frames), dtype="f8")
    group.create_dataset("physical_condition", shape=(total, len(PHYSICAL_CONDITION_NAMES)), dtype="f8")
    group.create_dataset("model_condition", shape=(total, len(MODEL_CONDITION_NAMES)), dtype="f4")
    group.create_dataset("vector_condition", shape=(total, len(VECTOR_CONDITION_NAMES), 3), dtype="f4")
    for key, dtype in (("condition_id", "i2"), ("trajectory_id", "i8"), ("split", "i1"), ("trajectory_seed", "i8")):
        group.create_dataset(key, shape=(total,), dtype=dtype)
    group.create_dataset("sample_weight", shape=(total,), dtype="f4")
    group.attrs["condition_names_json"] = json.dumps(MODEL_CONDITION_NAMES)
    group.attrs["physical_condition_names_json"] = json.dumps(PHYSICAL_CONDITION_NAMES)
    group.attrs["vector_condition_names_json"] = json.dumps(VECTOR_CONDITION_NAMES)
    return group


@torch.no_grad()
def generate_group(group, model: GomonayModelAdapter, args, temperatures: list[float], drives: list[float], system_seed: int) -> None:
    paths = args.paths_per_condition
    nx, ny = args.size
    generator_device = args.device if args.device.startswith("cuda") else "cpu"
    for condition_id, (temperature, drive) in enumerate(zip(temperatures, drives)):
        first = condition_id * paths
        last = first + paths
        seeds = np.arange(system_seed + condition_id * 1_000_000, system_seed + condition_id * 1_000_000 + paths, dtype=np.int64)
        generators = [torch.Generator(device=generator_device).manual_seed(int(seed)) for seed in seeds]
        if args.initial_state == "collinear":
            spins = collinear_state(paths, 2, nx, ny, antiferromagnetic=True, device=args.device, dtype=torch.float64)
        else:
            width_cells = args.wall_width_nm / model.parameters.lattice_nm
            spins = wall_pair_state(paths, nx, ny, width_cells, device=args.device, dtype=torch.float64)

        preparation = UnifiedLLGSolver(model, alpha=args.alpha)
        for step in range(args.equilibration_steps):
            spins = preparation.stochastic_heun_step(spins, step * args.dt, args.dt, temperature, generators)

        group["initial_spins"][first:last] = spins.to(torch.float32).cpu().numpy()
        solver = UnifiedLLGSolver(
            model,
            alpha=args.alpha,
            sot=CommonSOT(damping_like=drive, start=0.0, end=args.pulse_duration),
        )
        frame = 0
        for step in range(args.steps + 1):
            if step % args.save_every == 0:
                magnetization, neel = common_observables(spins)
                group["spins"][first:last, frame] = spins.to(torch.float32).cpu().numpy()
                group["magnetization"][first:last, frame] = magnetization.to(torch.float32).cpu().numpy()
                group["neel"][first:last, frame] = neel.to(torch.float32).cpu().numpy()
                group["energy"][first:last, frame] = model.energy(spins).cpu().numpy()
                frame += 1
            if step < args.steps:
                spins = solver.stochastic_heun_step(spins, step * args.dt, args.dt, temperature, generators)

        duration = args.steps * args.dt
        physical = np.asarray(
            [temperature, args.alpha, drive, 0.0, args.dt, args.save_every * args.dt, duration, args.pulse_duration],
            dtype=np.float64,
        )
        group["physical_condition"][first:last] = physical
        group["model_condition"][first:last] = model_conditions(model, args, drive, temperature)
        group["vector_condition"][first:last] = condition_vectors()
        group["condition_id"][first:last] = condition_id
        group["trajectory_id"][first:last] = np.arange(first, last)
        group["split"][first:last] = split_codes(
            paths, system_seed + condition_id * 101 + 17, args.split_policy
        )
        group["trajectory_seed"][first:last] = seeds
        group["sample_weight"][first:last] = 1.0
        print(f"generated {group.name} condition={condition_id} T={temperature:g} K drive={drive:g} T")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_namespace(base_seed: int, system_id: int, size: tuple[int, int]) -> int:
    """Create a deterministic, practically collision-free namespace per system/size."""
    payload = f"{base_seed}:{system_id}:{size[0]}:{size[1]}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:7], "little")


def main() -> None:
    args = parser().parse_args()
    if len(args.temperature) not in (1, len(args.drive)):
        raise ValueError("temperature must contain one value or one value per drive")
    temperatures = list(args.temperature) * len(args.drive) if len(args.temperature) == 1 else list(args.temperature)
    if min(args.size) < 3 or args.dt <= 0 or args.steps < 1 or args.save_every < 1:
        raise ValueError("require dimensions >=3, dt>0, steps>=1 and save-every>=1")
    if args.steps % args.save_every:
        raise ValueError("steps must be divisible by save-every")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    systems = {
        "altermagnet": ("d_wave_altermagnet", True, 2),
        "afm_control": ("conventional_afm_control", False, 1),
    }
    selected = tuple(systems) if args.system == "both" else (args.system,)
    if args.output is None:
        nx, ny = args.size
        args.output = ROOT / "data" / "scalable_paths" / f"{args.system}_{nx}x{ny}_{args.initial_state}.h5"
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    if partial.exists():
        partial.unlink()

    total = len(args.drive) * args.paths_per_condition
    frames = args.steps // args.save_every + 1
    manifest = {
        "schema_version": 2,
        "dataset_role": args.dataset_role,
        "status": "generated_not_convergence_certified",
        "size": list(args.size),
        "initial_state": args.initial_state,
        "split": (
            "80:10:10 within each physical condition, whole trajectories"
            if args.split_policy == "balanced_8_1_1"
            else args.split_policy
        ),
        "randomness": (
            "one stored and independently replayable PyTorch seed per trajectory; "
            "system/size namespaces are derived by SHA-256"
        ),
        "selection": "none; every simulated trajectory retained with sample_weight=1",
        "thermal_noise": "Stratonovich stochastic Heun; FDT covariance 2 alpha kBT/(gamma mu_s) delta(t-t')",
        "training_gate": "requires separate dt/save-cadence/size convergence and mechanism audit",
        "systems": {},
    }
    with h5py.File(partial, "w") as h5:
        h5.attrs["schema_version"] = 2
        h5.attrs["status"] = "writing"
        h5.create_dataset("time", data=np.arange(frames, dtype=np.float64) * args.save_every * args.dt)
        h5.attrs["periodic_axes_json"] = json.dumps(["x", "y"])
        h5.attrs["lattice_shape_json"] = json.dumps(list(args.size))
        for key in selected:
            name, alternating, system_id = systems[key]
            model = GomonayModelAdapter(alternating_exchange=alternating)
            group = allocate_group(h5, name, total, frames, tuple(args.size))
            group.attrs["system_id"] = system_id
            group.attrs["split_policy"] = args.split_policy
            group.attrs["model_metadata_json"] = json.dumps(model.metadata())
            group.attrs["initial_state"] = args.initial_state
            namespace = seed_namespace(args.seed, system_id, tuple(args.size))
            group.attrs["seed_namespace"] = namespace
            generate_group(group, model, args, temperatures, list(args.drive), namespace)
            manifest["systems"][name] = model.metadata()
        h5.attrs["status"] = "complete"
        h5.attrs["manifest_json"] = json.dumps(manifest)
        h5.flush()
    os.replace(partial, args.output)
    manifest["status"] = "generated_not_convergence_certified"
    manifest["file"] = args.output.name
    manifest["bytes"] = args.output.stat().st_size
    manifest["sha256"] = sha256(args.output)
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "manifest": str(manifest_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
