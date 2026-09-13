"""Reproduce the published zero-field helix period of CrNb3S6."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "core"))

from unified_llg import CrNb3S6Helimagnet, UnifiedLLGSolver  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate the CrNb3S6 helical ground state")
    p.add_argument("--length", type=int, default=80)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--dt", type=float, default=1e-15)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "noncollinear_validation" / "crnb3s6",
    )
    return p


def helix(length: int, winding: int, *, dtype=torch.float64) -> torch.Tensor:
    phase = torch.arange(length, dtype=dtype) * (2.0 * math.pi * winding / length)
    spins = torch.zeros((1, 1, length, 1, 3), dtype=dtype)
    spins[0, 0, :, 0, 0] = torch.cos(phase)
    spins[0, 0, :, 0, 1] = torch.sin(phase)
    return spins


def measured_wavevector(spins: torch.Tensor, cell_m: float) -> float:
    planar = spins[0, 0, :, 0, 0] + 1j * spins[0, 0, :, 0, 1]
    phase_step = torch.angle(torch.conj(planar) * torch.roll(planar, -1)).mean()
    return float(phase_step / cell_m)


@torch.no_grad()
def main() -> None:
    args = parser().parse_args()
    if args.length < 4 or args.steps < 1 or args.save_every < 1 or args.dt <= 0:
        raise ValueError("invalid validation settings")
    model = CrNb3S6Helimagnet()
    solver = UnifiedLLGSolver(model, alpha=args.alpha)
    spins = helix(args.length, 1)
    generator = torch.Generator().manual_seed(args.seed)

    energies = {
        str(winding): float(model.energy(helix(args.length, winding))[0])
        for winding in (0, 1, 2)
    }
    initial_energy = float(model.energy(spins)[0])
    initial_torque = float(
        torch.linalg.vector_norm(
            torch.linalg.cross(spins, model.field(spins), dim=-1), dim=-1
        ).max()
    )
    frames = [spins.cpu().numpy()[0]]
    for step in range(args.steps):
        spins = solver.stochastic_heun_step(
            spins, step * args.dt, args.dt, 0.0, generator
        )
        if (step + 1) % args.save_every == 0:
            frames.append(spins.cpu().numpy()[0])

    q_measured = measured_wavevector(spins, model.parameters.cell_m)
    final_energy = float(model.energy(spins)[0])
    metrics = {
        "status": "literature_period_and_stationarity_validation",
        "source": model.metadata(),
        "discretization": {
            "length_cells": args.length,
            "cell_nm": model.parameters.cell_m * 1e9,
            "periodic_length_nm": args.length * model.parameters.cell_m * 1e9,
            "winding": 1,
        },
        "period": {
            "published_nm": 48.0,
            "continuum_from_parameters_nm": model.continuum_period_m * 1e9,
            "discrete_unconstrained_from_parameters_nm": (
                2.0 * math.pi * model.parameters.cell_m
                / model.discrete_twist_rad
                * 1e9
            ),
            "discrete_parameter_prediction_relative_error": abs(
                2.0 * math.pi * model.parameters.cell_m
                / model.discrete_twist_rad
                * 1e9
                - 48.0
            )
            / 48.0,
            "simulated_commensurate_nm": 2.0 * math.pi / q_measured * 1e9,
            "commensurability_note": (
                "The 80-cell periodic box is deliberately one published 48 nm period; "
                "the independent checks are the parameter-predicted pitch, stationarity, "
                "and energy ordering of winding sectors."
            ),
        },
        "stationarity": {
            "initial_max_torque_T": initial_torque,
            "relative_energy_drift": abs(final_energy - initial_energy)
            / max(abs(initial_energy), 1e-300),
            "spin_norm_max_error": float(
                torch.max(torch.abs(torch.linalg.vector_norm(spins, dim=-1) - 1.0))
            ),
        },
        "winding_sector_energies_J": energies,
        "one_turn_is_lowest_of_tested_sectors": energies["1"] < min(
            energies["0"], energies["2"]
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "helix_trajectory.npz",
        time_s=np.arange(len(frames), dtype=np.float64) * args.save_every * args.dt,
        spins=np.asarray(frames, dtype=np.float32),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
