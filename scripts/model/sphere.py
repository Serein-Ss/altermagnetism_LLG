"""Product-of-spheres operations used by Riemannian flow matching."""
from __future__ import annotations

import math

import torch


def tangent_project(base: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return vector - (base * vector).sum(dim=-1, keepdim=True) * base


def sphere_exp(base: torch.Tensor, tangent: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    tangent = tangent_project(base, tangent)
    norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    direction = tangent / norm.clamp_min(eps)
    result = torch.cos(norm) * base + torch.sin(norm) * direction
    small = norm < eps
    result = torch.where(small, base + tangent, result)
    return result / torch.linalg.vector_norm(result, dim=-1, keepdim=True).clamp_min(eps)


def sphere_log(base: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    dot = (base * target).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(dot)
    tangent = target - dot * base
    tangent_norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    result = theta * tangent / tangent_norm.clamp_min(eps)
    return torch.where(tangent_norm < eps, tangent_project(base, target - base), result)


def _batch_scalar(value: torch.Tensor, ndim: int) -> torch.Tensor:
    return value.reshape(value.shape[0], *([1] * (ndim - 1)))


def geodesic_interpolate(reference: torch.Tensor, target: torch.Tensor, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return S_tau and analytic dS_tau/dtau along shortest spherical arcs."""
    tangent = sphere_log(reference, target)
    tangent_norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    tau_view = _batch_scalar(tau, reference.ndim)
    angle = tau_view * tangent_norm
    direction = tangent / tangent_norm.clamp_min(1e-7)
    state = torch.cos(angle) * reference + torch.sin(angle) * direction
    velocity = -torch.sin(angle) * tangent_norm * reference + torch.cos(angle) * tangent
    small = tangent_norm < 1e-7
    state = torch.where(small, reference + tau_view * tangent, state)
    velocity = torch.where(small, tangent, velocity)
    state = state / torch.linalg.vector_norm(state, dim=-1, keepdim=True).clamp_min(1e-7)
    return state, tangent_project(state, velocity)


def sample_reference_path(
    initial: torch.Tensor,
    frames: int,
    *,
    generator: torch.Generator | None = None,
    sigma: float = 0.9,
    temporal_correlation: float = 0.92,
) -> torch.Tensor:
    """Draw a temporally correlated spherical reference path anchored at frame zero."""
    if initial.ndim != 5 or initial.shape[-1] != 3 or frames < 2:
        raise ValueError("initial must be [B,S,Nx,Ny,3] and frames must be >=2")
    if not 0.0 <= temporal_correlation < 1.0:
        raise ValueError("temporal_correlation must be in [0,1)")
    batch, sublattices, nx, ny, _ = initial.shape
    noise = torch.randn(
        (batch, frames, sublattices, nx, ny, 3),
        generator=generator,
        device=initial.device,
        dtype=initial.dtype,
    )
    innovation = math.sqrt(1.0 - temporal_correlation**2)
    for index in range(1, frames):
        noise[:, index] = temporal_correlation * noise[:, index - 1] + innovation * noise[:, index]
    anchor = initial[:, None].expand_as(noise)
    tangent = tangent_project(anchor, noise)
    norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    tangent = tangent * (sigma / norm.clamp_min(1e-7))
    reference = sphere_exp(anchor, tangent)
    reference[:, 0] = initial
    return reference
