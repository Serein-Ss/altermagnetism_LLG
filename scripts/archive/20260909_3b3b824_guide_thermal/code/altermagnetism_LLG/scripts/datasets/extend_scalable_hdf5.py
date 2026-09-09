"""Continue schema-v2 LLG trajectories from their stored endpoint."""
from __future__ import annotations

import argparse
import hashlib
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
    UnifiedLLGSolver,
    common_observables,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target-duration-ps", type=float, default=2.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--overwrite", action="store_true")
    return p


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def continuation_seed(seed: int, completed_steps: int) -> int:
    payload = f"{seed}:continuation:{completed_steps}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:7], "little")


def create_extended_group(
    output: h5py.File,
    source: h5py.Group,
    frames: int,
) -> h5py.Group:
    target = output.create_group(source.name.rsplit("/", 1)[-1])
    for key, value in source.attrs.items():
        target.attrs[key] = value
    paths, _, sublattices, nx, ny, components = source["spins"].shape
    target.create_dataset(
        "spins",
        shape=(paths, frames, sublattices, nx, ny, components),
        dtype="f4",
        chunks=(1, min(frames, 8), sublattices, min(nx, 16), min(ny, 16), 3),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )
    target.create_dataset(
        "magnetization", shape=(paths, frames, 3), dtype="f4"
    )
    target.create_dataset(
        "neel", shape=(paths, frames, 3), dtype="f4"
    )
    target.create_dataset(
        "energy", shape=(paths, frames), dtype="f8"
    )
    for name, dataset in source.items():
        if name in (
            "spins",
            "magnetization",
            "neel",
            "energy",
            "continuation_seed",
        ):
            continue
        source.file.copy(dataset, target, name=name)
    target.create_dataset(
        "continuation_seed", shape=(paths,), dtype="i8"
    )
    return target


@torch.no_grad()
def main() -> None:
    args = parser().parse_args()
    if args.target_duration_ps <= 0:
        raise ValueError("target-duration-ps must be positive")
    if args.device != "cpu":
        raise ValueError(
            "LLG continuation is a CPU data task; submit it to the fat partition"
        )
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite {args.output}; pass --overwrite"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    if partial.exists():
        partial.unlink()

    input_hash = sha256(args.input)
    with h5py.File(args.input, "r", swmr=True) as source:
        if int(source.attrs.get("schema_version", 0)) != 2:
            raise ValueError("input must use schema version 2")
        groups = [
            group
            for group in source.values()
            if isinstance(group, h5py.Group) and "spins" in group
        ]
        if len(groups) != 1:
            raise ValueError("continuation expects exactly one trajectory group")
        source_group = groups[0]
        physical = source_group["physical_condition"][:]
        if not np.allclose(physical[:, 4], physical[0, 4]):
            raise ValueError("all trajectories must share one integration step")
        if not np.allclose(physical[:, 5], physical[0, 5]):
            raise ValueError("all trajectories must share one saved interval")
        if not np.allclose(physical[:, 6], physical[0, 6]):
            raise ValueError("all trajectories must share one duration")
        dt = float(physical[0, 4])
        saved_dt = float(physical[0, 5])
        current_duration = float(physical[0, 6])
        target_duration = args.target_duration_ps * 1e-12
        if target_duration <= current_duration:
            raise ValueError("target duration must exceed stored duration")
        completed_steps = int(round(current_duration / dt))
        target_steps = int(round(target_duration / dt))
        save_every = int(round(saved_dt / dt))
        if (
            not np.isclose(completed_steps * dt, current_duration)
            or not np.isclose(target_steps * dt, target_duration)
            or target_steps % save_every
        ):
            raise ValueError(
                "stored and target durations must align with dt/save cadence"
            )
        old_frames = len(source["time"])
        target_frames = target_steps // save_every + 1
        if old_frames != completed_steps // save_every + 1:
            raise ValueError("stored time axis is inconsistent with protocol")

        with h5py.File(partial, "w") as output:
            for key, value in source.attrs.items():
                if key != "manifest_json":
                    output.attrs[key] = value
            output.attrs["status"] = "writing"
            output.attrs["continued_from"] = str(args.input.resolve())
            output.attrs["continuation_rule"] = (
                "stored endpoint plus independent white-noise continuation; "
                "continuation seed is SHA256(original seed, completed steps)"
            )
            output.create_dataset(
                "time",
                data=np.arange(target_frames, dtype=np.float64)
                * save_every
                * dt,
            )
            target_group = create_extended_group(
                output, source_group, target_frames
            )
            for trajectory in range(len(source_group["spins"])):
                target_group["spins"][trajectory, :old_frames] = (
                    source_group["spins"][trajectory]
                )
                target_group["magnetization"][trajectory, :old_frames] = (
                    source_group["magnetization"][trajectory]
                )
                target_group["neel"][trajectory, :old_frames] = (
                    source_group["neel"][trajectory]
                )
                target_group["energy"][trajectory, :old_frames] = (
                    source_group["energy"][trajectory]
                )

            model = GomonayModelAdapter(
                alternating_exchange=(
                    source_group.name.rsplit("/", 1)[-1]
                    == "d_wave_altermagnet"
                )
            )
            condition_ids = source_group["condition_id"][:]
            original_seeds = source_group["trajectory_seed"][:]
            all_continuation_seeds = np.asarray(
                [
                    continuation_seed(int(seed), completed_steps)
                    for seed in original_seeds
                ],
                dtype=np.int64,
            )
            target_group["continuation_seed"][:] = all_continuation_seeds
            for condition_id in np.unique(condition_ids):
                indices = np.flatnonzero(
                    condition_ids == condition_id
                )
                spins = torch.from_numpy(
                    source_group["spins"][indices, -1]
                ).to(args.device, dtype=torch.float64)
                temperature = float(physical[indices[0], 0])
                alpha = float(physical[indices[0], 1])
                drive = float(physical[indices[0], 2])
                pulse_duration = float(physical[indices[0], 7])
                generators = [
                    torch.Generator(device="cpu").manual_seed(int(seed))
                    for seed in all_continuation_seeds[indices]
                ]
                solver = UnifiedLLGSolver(
                    model,
                    alpha=alpha,
                    sot=CommonSOT(
                        damping_like=drive,
                        start=0.0,
                        end=pulse_duration,
                    ),
                )
                frame = old_frames
                extension_steps = target_steps - completed_steps
                for step in range(1, extension_steps + 1):
                    absolute_step = completed_steps + step - 1
                    spins = solver.stochastic_heun_step(
                        spins,
                        absolute_step * dt,
                        dt,
                        temperature,
                        generators,
                    )
                    if step % save_every == 0:
                        magnetization, neel = common_observables(spins)
                        target_group["spins"][indices, frame] = (
                            spins.to(torch.float32).cpu().numpy()
                        )
                        target_group["magnetization"][indices, frame] = (
                            magnetization.to(torch.float32).cpu().numpy()
                        )
                        target_group["neel"][indices, frame] = (
                            neel.to(torch.float32).cpu().numpy()
                        )
                        target_group["energy"][indices, frame] = (
                            model.energy(spins).cpu().numpy()
                        )
                        frame += 1
                if frame != target_frames:
                    raise RuntimeError("continuation produced wrong frame count")
                print(
                    f"condition={int(condition_id)} "
                    f"paths={len(indices)} complete",
                    flush=True,
                )

            updated_physical = physical.copy()
            updated_physical[:, 6] = target_duration
            target_group["physical_condition"][:] = updated_physical
            updated_model = target_group["model_condition"][:]
            duration_ratio = target_duration / current_duration
            updated_model[:, 5] *= duration_ratio
            updated_model[:, 6] = (
                np.minimum(updated_physical[:, 7], target_duration)
                / target_duration
            )
            target_group["model_condition"][:] = updated_model
            output.attrs["status"] = "complete"
            output.flush()
    os.replace(partial, args.output)

    report = {
        "schema_version": 2,
        "status": "extended_observation_not_retrained",
        "source": str(args.input.resolve()),
        "source_sha256": input_hash,
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "target_duration_ps": args.target_duration_ps,
        "continuation": (
            "exact stored state at 1 ps; statistically valid independent "
            "white-noise continuation from that Markov state"
        ),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
