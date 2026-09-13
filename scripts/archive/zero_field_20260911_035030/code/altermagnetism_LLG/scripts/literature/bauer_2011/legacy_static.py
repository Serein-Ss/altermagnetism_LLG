"""Validate the Bauer et al. chain Hamiltonian against its domain-wall barrier."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "core"))

from unified_llg import BauerChainParameters, BauerOpenChainHamiltonian  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bauer 2011 chain barrier validation")
    p.add_argument("--length", type=int, default=100)
    p.add_argument("--output", type=Path, default=ROOT / 'data/literature_reproduction/bauer_2011/derived/legacy_before_reduced_20260909/bauer2011_barrier.json' )
    return p


def optimized_wall(length: int, anisotropy: float) -> dict:
    model = BauerOpenChainHamiltonian(
        BauerChainParameters(anisotropy_k=anisotropy)
    )
    width = model.continuum_domain_wall_width_sites
    x = torch.arange(length, dtype=torch.float64)
    initial_theta = 2.0 * torch.atan(
        torch.exp(-(x - 0.5 * (length - 1)) / max(width / 2.0, 1e-9))
    )
    interior = torch.nn.Parameter(initial_theta[1:-1].clone())
    optimizer = torch.optim.LBFGS(
        [interior], max_iter=500, tolerance_grad=1e-13, tolerance_change=1e-15
    )

    def make_spins() -> torch.Tensor:
        theta = torch.cat(
            (interior.new_tensor([math.pi]), interior, interior.new_tensor([0.0]))
        )
        spins = torch.zeros((1, 1, length, 1, 3), dtype=torch.float64)
        spins[0, 0, :, 0, 0] = torch.sin(theta)
        spins[0, 0, :, 0, 2] = torch.cos(theta)
        return spins

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = model.energy(make_spins())[0]
        loss.backward()
        return loss

    optimizer.step(closure)
    wall = make_spins().detach()
    uniform = torch.zeros_like(wall)
    uniform[..., 2] = 1.0
    excess = float(model.energy(wall)[0] - model.energy(uniform)[0])
    continuum = model.continuum_domain_wall_energy_j
    return {
        "K_over_J": anisotropy,
        "length": length,
        "continuum_wall_width_sites": width,
        "continuum_barrier_J": continuum,
        "optimized_discrete_wall_energy_J": excess,
        "relative_difference": abs(excess - continuum) / continuum,
    }


def main() -> None:
    args = parser().parse_args()
    if args.length < 20:
        raise ValueError("length must be at least 20 for the continuum comparison")
    results = {
        "status": "published_hamiltonian_static_barrier_validation",
        "source": BauerOpenChainHamiltonian().metadata(),
        "checks": [optimized_wall(args.length, k) for k in (0.01, 0.1)],
        "scope": (
            "This validates the Hamiltonian and the domain-wall energy scale. "
            "It does not reproduce the paper's stochastic lifetime without a long run."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
