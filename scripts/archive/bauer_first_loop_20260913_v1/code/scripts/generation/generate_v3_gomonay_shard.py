"""Generate one raw Standard-V3 Gomonay trajectory shard on a CPU node."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
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


ENERGY_COMPONENTS = ("J1_inter", "J2_intra", "Jtilde_alternating", "anisotropy")
STRUCTURE_COMPONENTS = ("q0_power", "peak_power", "peak_kx_index", "peak_ky_index")
OUTCOME_PENDING = "pending_offline_detection"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("pilot", "production"), required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--initial-pool", type=Path)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--temperature-k", type=float, default=5.0)
    p.add_argument("--drive-t", type=float, required=True)
    p.add_argument("--polarization", type=float, nargs=3, default=(2**-0.5, 2**-0.5, 0.0))
    p.add_argument("--condition-id", type=int, required=True)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--paths", type=int, default=50)
    p.add_argument("--dt-fs", type=float, required=True)
    p.add_argument("--duration-ps", type=float, required=True)
    p.add_argument("--saved-dt-fs", type=float, required=True)
    p.add_argument("--equilibration-ps", type=float, default=0.2)
    p.add_argument("--pulse-ps", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--base-seed", type=int, default=20260908)
    p.add_argument("--device", choices=("cpu",), default="cpu")
    return p


def load_mapping(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one mapping")
    return value


def seed64(base_seed: int, *parts: object) -> int:
    payload = ":".join(str(value) for value in (base_seed, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def split_from_initial_id(initial_state_id: int) -> int:
    bucket = seed64(0, "split", initial_state_id) % 10
    return 0 if bucket < 8 else 1 if bucket == 8 else 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, cwd=ROOT,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True, cwd=ROOT,
    ).stdout.strip())
    return commit, dirty


def validate_protocol(args: argparse.Namespace, protocol: dict) -> None:
    if protocol.get("schema_version") != 3:
        raise ValueError("protocol must use schema version 3")
    if args.mode == "pilot":
        if protocol.get("status") != "candidate_not_frozen":
            raise ValueError("pilot mode requires the registered candidate protocol")
        return
    if protocol.get("status") != "frozen" or protocol.get("production_enabled") is not True:
        raise ValueError("production mode requires a frozen, production-enabled protocol")
    if args.initial_pool is None:
        raise ValueError("production mode requires an equilibrium-certified initial pool")
    required = protocol.get("certification_inputs", {})
    if not required or not all(Path(path).is_file() for path in required.values()):
        raise ValueError("production protocol certification inputs are missing")


def energy_components(model: GomonayModelAdapter, spins: torch.Tensor) -> torch.Tensor:
    p = model.parameters
    m1, m2 = spins.unbind(dim=1)
    roll = lambda value, sx, sy: torch.roll(value, (sx, sy), (1, 2))
    cross = m2 + roll(m2, 1, 0) + roll(m2, 0, 1) + roll(m2, 1, 1)
    inter = p.j1 * (m1 * cross).sum(dim=(1, 2, 3))
    intra = torch.zeros_like(inter)
    for sublattice in (m1, m2):
        intra -= p.j2 * (
            sublattice * (roll(sublattice, -1, 0) + roll(sublattice, 0, -1))
        ).sum(dim=(1, 2, 3))
    alt = p.j_tilde * (
        (m1 * roll(m1, 1, -1)).sum(dim=(1, 2, 3))
        - (m2 * roll(m2, 1, -1)).sum(dim=(1, 2, 3))
        - (m1 * roll(m1, -1, -1)).sum(dim=(1, 2, 3))
        + (m2 * roll(m2, -1, -1)).sum(dim=(1, 2, 3))
    )
    anisotropy = -p.anisotropy * spins[..., 2].square().sum(dim=(1, 2, 3))
    return torch.stack((inter, intra, alt, anisotropy), dim=1)


def spatial_summaries(spins: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    local = 0.5 * (spins[:, 0, ..., 2] - spins[:, 1, ..., 2])
    power = torch.fft.fft2(local, norm="ortho").abs().square()
    flat = power.flatten(1)
    peak = flat.argmax(dim=1)
    ny = local.shape[2]
    structure = torch.stack(
        (
            power[:, 0, 0],
            flat.gather(1, peak[:, None])[:, 0],
            torch.div(peak, ny, rounding_mode="floor").to(local.dtype),
            (peak % ny).to(local.dtype),
        ),
        dim=1,
    )
    sign = local < 0
    wall = 0.5 * (
        (sign != torch.roll(sign, -1, 1)).to(local.dtype).mean(dim=(1, 2))
        + (sign != torch.roll(sign, -1, 2)).to(local.dtype).mean(dim=(1, 2))
    )
    return structure, wall


def geometry(group: h5py.Group, model: GomonayModelAdapter, size: int) -> None:
    template = np.asarray(
        [
            [0, 1, 0, 0, 0], [0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0], [0, 1, 1, 1, 0],
            [0, 0, -1, 0, 1], [0, 0, 0, -1, 1],
            [1, 1, -1, 0, 1], [1, 1, 0, -1, 1],
            [0, 0, 1, -1, 2], [0, 0, -1, -1, 3],
            [1, 1, 1, -1, 3], [1, 1, -1, -1, 2],
        ],
        dtype=np.int16,
    )
    p = model.parameters
    group.create_dataset("neighbor_template", data=template)
    group.create_dataset("bond_type", data=template[:, 4])
    group.create_dataset("bond_vector", data=template[:, 2:4])
    group.create_dataset(
        "coupling_by_bond_type",
        data=np.asarray((p.j1, -p.j2, p.j_tilde, -p.j_tilde), dtype=np.float64),
    )
    group.create_dataset("sublattice_id", data=np.asarray((0, 1), dtype=np.int8))
    group.create_dataset("site_mask", data=np.ones((2, size, size), dtype=np.bool_))
    group.create_dataset("periodic_axes", data=np.asarray((True, True), dtype=np.bool_))
    group.attrs["parameter_names_json"] = json.dumps(
        ["J1", "J2", "J_tilde", "K", "mu_s", "a0", "gamma"]
    )
    group.attrs["parameter_values_json"] = json.dumps(
        [p.j1, p.j2, p.j_tilde, p.anisotropy, p.mu_s, p.lattice, p.gamma]
    )
    group.attrs["parameter_units_json"] = json.dumps(
        ["J", "J", "J", "J", "J/T", "m", "rad/(s*T)"]
    )


def load_initial_states(
    args: argparse.Namespace,
    preparation_seeds: np.ndarray,
    initial_ids: np.ndarray,
    model: GomonayModelAdapter,
    dt: float,
) -> tuple[torch.Tensor, np.ndarray]:
    if args.initial_pool is not None:
        with h5py.File(args.initial_pool, "r", swmr=True) as pool:
            if pool.attrs.get("certification_status", "") != "equilibrium_certified":
                raise ValueError("initial pool is not equilibrium_certified")
            if tuple(pool["spins"].shape[1:]) != (2, args.size, args.size, 3):
                raise ValueError("initial pool lattice shape does not match")
            offset = args.shard_index * args.paths
            if offset + args.paths > len(pool["spins"]):
                raise ValueError("initial pool does not contain this shard")
            spins = torch.from_numpy(pool["spins"][offset : offset + args.paths]).to(torch.float64)
            ids = pool["initial_state_id"][offset : offset + args.paths].astype(np.int64)
            splits = pool["split"][offset : offset + args.paths].astype(np.int8)
        if not np.array_equal(ids, initial_ids):
            raise ValueError("initial pool IDs do not match the deterministic namespace")
        return spins, splits

    spins = collinear_state(
        args.paths, 2, args.size, args.size,
        antiferromagnetic=True, device="cpu", dtype=torch.float64,
    )
    generators = [torch.Generator().manual_seed(int(seed)) for seed in preparation_seeds]
    solver = UnifiedLLGSolver(model, alpha=args.alpha)
    equilibration_steps = round(args.equilibration_ps * 1e-12 / dt)
    for step in range(equilibration_steps):
        spins = solver.stochastic_heun_step(
            spins, step * dt, dt, args.temperature_k, generators
        )
    splits = np.asarray([split_from_initial_id(int(value)) for value in initial_ids], dtype=np.int8)
    return spins, splits


@torch.no_grad()
def main() -> None:
    args = parser().parse_args()
    protocol = load_mapping(args.protocol)
    validate_protocol(args, protocol)
    if not 1 <= args.paths <= 100 or args.size < 3:
        raise ValueError("a shard must contain 1-100 paths and size must be >=3")
    if min(args.dt_fs, args.duration_ps, args.saved_dt_fs) <= 0:
        raise ValueError("time parameters must be positive")
    if args.mode == "production" and args.paths != 50:
        raise ValueError("production shards must contain exactly 50 paths")
    polarization = np.asarray(args.polarization, dtype=np.float64)
    if not np.isclose(np.linalg.norm(polarization), 1.0, atol=1e-12):
        raise ValueError("polarization must be normalized")
    dt = args.dt_fs * 1e-15
    duration = args.duration_ps * 1e-12
    saved_dt = args.saved_dt_fs * 1e-15
    steps = round(duration / dt)
    save_every = round(saved_dt / dt)
    if not np.isclose(steps * dt, duration) or save_every < 1 or steps % save_every:
        raise ValueError("duration and saved interval must be exact multiples of dt")
    frames = steps // save_every + 1
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    if partial.exists():
        partial.unlink()

    model = GomonayModelAdapter(alternating_exchange=True)
    global_ids = args.shard_index * args.paths + np.arange(args.paths, dtype=np.int64)
    initial_ids = np.asarray(
        [seed64(args.base_seed, "initial", args.size, int(index)) for index in global_ids],
        dtype=np.int64,
    )
    preparation_seeds = np.asarray(
        [seed64(args.base_seed, "prepare", args.size, args.temperature_k, int(index)) for index in global_ids],
        dtype=np.int64,
    )
    dynamics_seeds = np.asarray(
        [seed64(args.base_seed, "dynamics", args.size, args.condition_id, int(index)) for index in global_ids],
        dtype=np.int64,
    )
    spins, splits = load_initial_states(
        args, preparation_seeds, initial_ids, model, dt
    )
    initial_spins = spins.to(torch.float32).numpy()
    generators = [torch.Generator().manual_seed(int(seed)) for seed in dynamics_seeds]
    solver = UnifiedLLGSolver(
        model,
        alpha=args.alpha,
        sot=CommonSOT(
            damping_like=args.drive_t,
            field_like=0.0,
            polarization=tuple(float(value) for value in polarization),
            start=0.0,
            end=args.pulse_ps * 1e-12,
        ),
    )
    time = np.arange(frames, dtype=np.float64) * saved_dt
    string_dtype = h5py.string_dtype("utf-8")
    commit, dirty = git_state()
    with h5py.File(partial, "w") as h5:
        h5.attrs["schema_version"] = 3
        h5.attrs["status"] = "writing"
        h5.attrs["dataset_stage"] = args.mode
        h5.attrs["label_status"] = "pending_separate_offline_detection"
        h5.attrs["protocol"] = str(args.protocol.resolve())
        h5.attrs["code_commit"] = commit
        h5.attrs["working_tree_dirty"] = dirty
        h5.create_dataset("time", data=time)
        spin_shape = (args.paths, frames, 2, args.size, args.size, 3)
        h5.create_dataset(
            "spins", shape=spin_shape, dtype="f4",
            chunks=(1, min(8, frames), 2, min(16, args.size), min(16, args.size), 3),
            compression="gzip", compression_opts=4, shuffle=True,
        )
        h5.create_dataset("initial_spins", data=initial_spins, compression="gzip", shuffle=True)
        h5.create_dataset("energy_total", shape=(args.paths, frames), dtype="f8")
        h5.create_dataset("energy_components", shape=(args.paths, frames, 4), dtype="f8")
        h5.create_dataset("magnetization", shape=(args.paths, frames, 3), dtype="f4")
        h5.create_dataset("neel", shape=(args.paths, frames, 3), dtype="f4")
        h5.create_dataset("structure_factor_summary", shape=(args.paths, frames, 4), dtype="f4")
        h5.create_dataset("wall_or_defect_density", shape=(args.paths, frames), dtype="f4")
        waveform = np.zeros((args.paths, frames, 3), dtype=np.float32)
        waveform[:, time < args.pulse_ps * 1e-12] = (
            args.drive_t * polarization
        ).astype(np.float32)
        h5.create_dataset("drive_waveform", data=waveform, compression="gzip", shuffle=True)

        h5.create_dataset("system_id", data=np.full(args.paths, "gomonay2024_d_wave_altermagnet", dtype=object), dtype=string_dtype)
        h5.create_dataset("hamiltonian_id", data=np.full(args.paths, "gomonay2024_supplement_eq_S1", dtype=object), dtype=string_dtype)
        h5.create_dataset("condition_id", data=np.full(args.paths, args.condition_id, dtype=np.int32))
        h5.create_dataset("trajectory_id", data=global_ids)
        h5.create_dataset("initial_state_id", data=initial_ids)
        h5.create_dataset("preparation_seed", data=preparation_seeds)
        h5.create_dataset("dynamics_seed", data=dynamics_seeds)
        h5.create_dataset("split", data=splits)
        h5.create_dataset("sample_weight", data=np.ones(args.paths, dtype=np.float32))
        h5.create_dataset("first_crossing_time", data=np.full(args.paths, np.nan))
        h5.create_dataset("steady_time", data=np.full(args.paths, np.nan))
        h5.create_dataset("steady_state_id", data=np.full(args.paths, -1, dtype=np.int16))
        h5.create_dataset("event_observed", data=np.full(args.paths, -1, dtype=np.int8))
        h5.create_dataset("censor_time", data=np.full(args.paths, duration))
        h5.create_dataset("outcome_label", data=np.full(args.paths, OUTCOME_PENDING, dtype=object), dtype=string_dtype)

        conditions = h5.create_group("physical_conditions")
        for name, value in (
            ("temperature", args.temperature_k), ("alpha", args.alpha),
            ("gamma", model.gamma), ("moment_or_Ms", model.moment),
            ("integration_dt", dt), ("saved_dt", saved_dt),
            ("fixed_duration", duration), ("pulse_duration", args.pulse_ps * 1e-12),
            ("field_like_amplitude", 0.0), ("damping_like_amplitude", args.drive_t),
            ("lattice_constant_or_cell_size", model.parameters.lattice),
            ("number_of_sublattices", 2),
        ):
            conditions.create_dataset(name, data=value)
        conditions.create_dataset("polarization_vector", data=polarization)
        conditions.create_dataset("crystal_frame", data=np.eye(3, dtype=np.float64))
        conditions.create_dataset("lattice_shape", data=np.asarray((args.size, args.size), dtype=np.int32))
        conditions.attrs["boundary_condition"] = "periodic_xy"
        geometry(h5.create_group("geometry"), model, args.size)
        h5.attrs["energy_component_names_json"] = json.dumps(ENERGY_COMPONENTS)
        h5.attrs["structure_factor_component_names_json"] = json.dumps(STRUCTURE_COMPONENTS)

        frame = 0
        for step in range(steps + 1):
            if step % save_every == 0:
                magnetization, neel = common_observables(spins)
                components = energy_components(model, spins)
                structure, wall = spatial_summaries(spins)
                h5["spins"][:, frame] = spins.to(torch.float32).numpy()
                h5["energy_components"][:, frame] = components.numpy()
                h5["energy_total"][:, frame] = components.sum(dim=1).numpy()
                h5["magnetization"][:, frame] = magnetization.to(torch.float32).numpy()
                h5["neel"][:, frame] = neel.to(torch.float32).numpy()
                h5["structure_factor_summary"][:, frame] = structure.to(torch.float32).numpy()
                h5["wall_or_defect_density"][:, frame] = wall.to(torch.float32).numpy()
                frame += 1
            if step != steps:
                spins = solver.stochastic_heun_step(
                    spins, step * dt, dt, args.temperature_k, generators
                )
        h5.attrs["status"] = "complete_raw_unlabelled"
        h5.flush()
    os.replace(partial, args.output)
    manifest = {
        "schema_version": 3,
        "status": "complete_raw_unlabelled_not_certified",
        "file": args.output.name,
        "bytes": args.output.stat().st_size,
        "sha256": sha256(args.output),
        "protocol": str(args.protocol.resolve()),
        "code_commit": commit,
        "working_tree_dirty": dirty,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "h5py": h5py.__version__,
        },
        "selection": "none; all paths retained with sample_weight=1",
        "labels": "pending separate offline steady-state detection",
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "manifest": str(manifest_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
