"""Diagnose configuration-dependent response to directional thermal impulses.

Directional noise is used only as a response probe. Production finite-temperature
trajectories retain the isotropic fluctuation-dissipation covariance.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "core"))

from altermagnet_dynamics import (  # noqa: E402
    LLGDynamics,
    MEV_TO_J,
    LiteratureParameters,
    Ruo2DoubleLayerHamiltonian,
    antiferromagnetic_state,
    thermal_field_std,
    unit,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Directional thermal-response diagnostic")
    p.add_argument("--samples", type=int, default=4096)
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--temperature-k", type=float, default=5.0)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--dt", type=float, default=1e-16)
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "research" / "configuration_noise_response.json",
    )
    return p


def planar_spiral(batch: int, size: int, device: str) -> torch.Tensor:
    x = torch.arange(size, device=device, dtype=torch.float64)[:, None]
    phase = 2 * math.pi * x / size
    neel = torch.zeros((size, size, 3), device=device, dtype=torch.float64)
    neel[..., 0] = torch.cos(phase)
    neel[..., 1] = torch.sin(phase)
    spins = torch.empty((batch, 2, size, size, 3), device=device, dtype=torch.float64)
    spins[:, 0] = neel
    spins[:, 1] = -neel
    return spins


def heun_given_field(
    spins: torch.Tensor,
    noise: torch.Tensor,
    dt: float,
    dynamics: LLGDynamics,
) -> torch.Tensor:
    f0 = dynamics.rhs(spins, 0.0, noise)
    predictor = unit(spins + dt * f0)
    f1 = dynamics.rhs(predictor, dt, noise)
    return unit(spins + 0.5 * dt * (f0 + f1))


@torch.no_grad()
def diagnose_state(
    name: str,
    base_state: torch.Tensor,
    samples: int,
    model: Ruo2DoubleLayerHamiltonian,
    dynamics: LLGDynamics,
    temperature: float,
    dt: float,
    seed: int,
) -> dict:
    state = base_state.expand(samples, -1, -1, -1, -1).clone()
    n_spins = state.shape[1] * state.shape[2] * state.shape[3]
    initial_energy = model.energy(state)
    deterministic = heun_given_field(state, torch.zeros_like(state), dt, dynamics)
    deterministic_delta = model.energy(deterministic) - initial_energy
    torque = torch.linalg.cross(state, model.field(state))

    generator_device = str(state.device) if state.is_cuda else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    std = thermal_field_std(model.parameters, temperature, dt, dynamics.alpha)
    half = samples // 2
    scalar_half = torch.randn(
        (half, 2, state.shape[2], state.shape[3], 1),
        generator=generator,
        device=state.device,
        dtype=state.dtype,
    ) * std
    scalar = torch.cat((scalar_half, -scalar_half), dim=0)
    axes = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
    response = {}
    for axis, vector in axes.items():
        direction = state.new_tensor(vector).view(1, 1, 1, 1, 3)
        evolved = heun_given_field(state, scalar * direction, dt, dynamics)
        excess = (model.energy(evolved) - initial_energy - deterministic_delta) / (
            n_spins * MEV_TO_J
        )
        paired_even_response = 0.5 * (excess[:half] + excess[half:])
        response[axis] = {
            "mean_excess_energy_meV_per_spin": float(paired_even_response.mean()),
            "sem_meV_per_spin": float(
                paired_even_response.std(unbiased=True) / math.sqrt(half)
            ),
            "positive_fraction": float((excess > 0).to(torch.float64).mean()),
        }
    return {
        "state": name,
        "initial_energy_meV_per_spin": float(initial_energy[0] / (n_spins * MEV_TO_J)),
        "zero_noise_delta_meV_per_spin": float(
            deterministic_delta.mean() / (n_spins * MEV_TO_J)
        ),
        "rms_deterministic_torque_T": float(
            torch.sqrt(torch.mean(torch.linalg.vector_norm(torque, dim=-1) ** 2))
        ),
        "directional_response": response,
    }


@torch.no_grad()
def main() -> None:
    args = parser().parse_args()
    if (
        args.samples < 2
        or args.samples % 2
        or args.size < 2
        or args.temperature_k < 0
        or args.dt <= 0
    ):
        raise ValueError("invalid diagnostic settings")
    params = LiteratureParameters()
    model = Ruo2DoubleLayerHamiltonian(params)
    dynamics = LLGDynamics(model, alpha=args.alpha)
    collinear = antiferromagnetic_state(
        1, args.size, args.size, device=args.device, dtype=torch.float64
    )
    spiral = planar_spiral(1, args.size, args.device)
    results = {
        "purpose": "configuration-dependent directional response check",
        "warning": (
            "axis-restricted noise is not a physical production bath; it is a paired "
            "susceptibility diagnostic. The planar spiral is imposed and is not a "
            "validated equilibrium phase of this easy-axis Hamiltonian."
        ),
        "estimator": "antithetic +/- field pairs cancel odd-in-noise energy response",
        "production_noise": "isotropic Cartesian FDT field",
        "parameters": {
            "temperature_K": args.temperature_k,
            "alpha": args.alpha,
            "dt_s": args.dt,
            "samples": args.samples,
            "lattice": [args.size, args.size],
            "thermal_field_component_std_T": thermal_field_std(
                params, args.temperature_k, args.dt, args.alpha
            ),
        },
        "states": [
            diagnose_state(
                "collinear_neel_z",
                collinear,
                args.samples,
                model,
                dynamics,
                args.temperature_k,
                args.dt,
                args.seed,
            ),
            diagnose_state(
                "imposed_planar_spiral_xy",
                spiral,
                args.samples,
                model,
                dynamics,
                args.temperature_k,
                args.dt,
                args.seed,
            ),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
