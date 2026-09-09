"""Independent noise-controlled integrators for the GUIDE numerical audit.

The midpoint method solves the Stratonovich midpoint equation without normalizing
the predictor or result. Failure to converge is an error, never a silent fallback.
Thermal fields are supplied by the caller so coarse/fine paths can share Brownian
increments. These routines do not replace the production solver.
"""
from __future__ import annotations

import torch


def heun_with_field(solver, spins, time, dt, field):
    first = solver.rhs(spins, time, field)
    raw_predictor = spins + dt * first
    predictor = raw_predictor / raw_predictor.norm(dim=-1, keepdim=True)
    raw = spins + 0.5 * dt * (first + solver.rhs(predictor, time + dt, field))
    errors = torch.stack((
        (raw_predictor.norm(dim=-1) - 1).abs().max(),
        (raw.norm(dim=-1) - 1).abs().max(),
    ))
    return raw / raw.norm(dim=-1, keepdim=True), errors


def midpoint_with_field(solver, spins, time, dt, field, *, tolerance=1e-13,
                        max_iterations=100):
    candidate = spins.clone()
    for _ in range(max_iterations):
        updated = spins + dt * solver.rhs(
            (spins + candidate) * 0.5, time + dt * 0.5, field
        )
        error = (updated - candidate).abs().max().item()
        candidate = updated
        if error < tolerance:
            residual = (candidate - spins - dt * solver.rhs(
                (spins + candidate) * 0.5, time + dt * 0.5, field
            )).abs().max().item()
            if residual < tolerance:
                return candidate
    raise RuntimeError("implicit midpoint did not converge; reduce dt")


def coupled_fields(solver, spins, temperature, coarse_dt, generator):
    """Four dt/4 fields, two dt/2 fields and one dt field from identical dW."""
    fine = torch.stack([
        solver.sample_thermal_field(spins, temperature, coarse_dt / 4, generator)
        for _ in range(4)
    ])
    return fine.mean(dim=0), fine.reshape(2, 2, *spins.shape).mean(dim=1), fine
