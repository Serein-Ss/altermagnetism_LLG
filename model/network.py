"""PyTorch-only SO(3)-covariant, periodic, arbitrary-size path velocity model."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .sphere import tangent_project


class PeriodicConv3d(nn.Module):
    """Replicate-pad time and circular-pad both spatial axes."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.pad = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        p = self.pad
        inputs = F.pad(inputs, (p, p, p, p, 0, 0), mode="circular")
        inputs = F.pad(inputs, (0, 0, 0, 0, p, p), mode="replicate")
        return self.conv(inputs)


class ResidualScalarBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.first = PeriodicConv3d(channels, channels)
        self.second = PeriodicConv3d(channels, channels)
        self.norm1 = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
        self.norm2 = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.norm1(self.first(inputs)))
        return inputs + self.norm2(self.second(hidden))


class PeriodicEquivariantFlowNet(nn.Module):
    """Predict one tangent velocity per spin without fixing lattice dimensions.

    Vector outputs are linear combinations of covariant local vector bases.  Their
    coefficients are produced solely from scalar invariants, which gives exact joint
    proper-rotation covariance while circular padding gives periodic translation
    equivariance.  This first implementation intentionally avoids an e3nn dependency.
    """

    def __init__(self, scalar_condition_dim: int = 10, vector_condition_count: int = 5, hidden: int = 64, blocks: int = 8):
        super().__init__()
        self.scalar_condition_dim = scalar_condition_dim
        self.vector_condition_count = vector_condition_count
        self.local_basis_count = 6 + vector_condition_count
        # Two frame-wise scalar channels identify physical progress and whether
        # the finite-duration drive pulse is active.  They preserve SO(3)
        # covariance because they only modulate invariant coefficients.
        invariant_channels = 2 * self.local_basis_count + scalar_condition_dim + 5 + 2
        self.input = PeriodicConv3d(invariant_channels, hidden)
        self.blocks = nn.Sequential(*[ResidualScalarBlock(hidden) for _ in range(blocks)])
        self.output = PeriodicConv3d(hidden, 2 * self.local_basis_count)

    @staticmethod
    def _spatial_bases(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lap_x = torch.roll(state, 1, 3) + torch.roll(state, -1, 3) - 2.0 * state
        lap_y = torch.roll(state, 1, 4) + torch.roll(state, -1, 4) - 2.0 * state
        return lap_x, lap_y

    @staticmethod
    def _temporal_difference(state: torch.Tensor) -> torch.Tensor:
        previous = torch.cat((state[:, :1], state[:, :-1]), dim=1)
        following = torch.cat((state[:, 1:], state[:, -1:]), dim=1)
        return 0.5 * (following - previous)

    @staticmethod
    def _dimensionless_hamiltonian_field(
        state: torch.Tensor,
        scalar_condition: torch.Tensor,
        vector_condition: torch.Tensor,
    ) -> torch.Tensor:
        """Return (mu_s/J1) B_eff for the stored Gomonay interaction stencil."""
        if state.shape[2] != 2:
            return torch.zeros_like(state)
        first, second = state.unbind(dim=2)
        cross_first = second + torch.roll(second, 1, 2) + torch.roll(second, 1, 3) + torch.roll(second, (1, 1), (2, 3))
        cross_second = first + torch.roll(first, -1, 2) + torch.roll(first, -1, 3) + torch.roll(first, (-1, -1), (2, 3))
        cross = torch.stack((cross_first, cross_second), dim=2)
        axial = (
            torch.roll(state, 1, 3)
            + torch.roll(state, -1, 3)
            + torch.roll(state, 1, 4)
            + torch.roll(state, -1, 4)
        )
        diagonal = (
            torch.roll(state, (1, -1), (3, 4))
            + torch.roll(state, (-1, 1), (3, 4))
            - torch.roll(state, (-1, -1), (3, 4))
            - torch.roll(state, (1, 1), (3, 4))
        )
        signs = state.new_tensor((1.0, -1.0))[None, None, :, None, None, None]
        crystal_z = vector_condition[:, 4, None, None, None, None, :]
        crystal_z = crystal_z.expand(-1, state.shape[1], state.shape[2], state.shape[3], state.shape[4], -1)
        anisotropy = (state * crystal_z).sum(dim=-1, keepdim=True) * crystal_z
        ratio_j2 = scalar_condition[:, 7, None, None, None, None, None]
        ratio_jtilde = scalar_condition[:, 8, None, None, None, None, None]
        ratio_k = scalar_condition[:, 9, None, None, None, None, None]
        return -cross + ratio_j2 * axial - ratio_jtilde * signs * diagonal + 2.0 * ratio_k * anisotropy

    def forward(
        self,
        state: torch.Tensor,
        tau: torch.Tensor,
        initial: torch.Tensor,
        scalar_condition: torch.Tensor,
        vector_condition: torch.Tensor | None = None,
        physical_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state.ndim != 6 or state.shape[-1] != 3:
            raise ValueError("state must be [B,T,S,Nx,Ny,3]")
        batch, frames, sublattices, nx, ny, _ = state.shape
        if initial.shape != (batch, sublattices, nx, ny, 3):
            raise ValueError("initial shape is incompatible with state")
        if scalar_condition.shape != (batch, self.scalar_condition_dim):
            raise ValueError("scalar_condition has the wrong shape")
        if vector_condition is None:
            vector_condition = state.new_zeros((batch, self.vector_condition_count, 3))
        if vector_condition.shape != (batch, self.vector_condition_count, 3):
            raise ValueError("vector_condition has the wrong shape")
        if physical_time is not None and physical_time.shape != (batch, frames):
            raise ValueError("physical_time must have shape [B,T]")


        lap_x, lap_y = self._spatial_bases(state)
        temporal = self._temporal_difference(state)
        initial_path = initial[:, None].expand(-1, frames, -1, -1, -1, -1)
        other = torch.flip(state, dims=(2,)) if sublattices == 2 else state
        hamiltonian_field = self._dimensionless_hamiltonian_field(state, scalar_condition, vector_condition)
        bases = [initial_path, lap_x, lap_y, temporal, other, hamiltonian_field]
        for index in range(self.vector_condition_count):
            vector = vector_condition[:, index, None, None, None, None, :]
            bases.append(vector.expand(-1, frames, sublattices, nx, ny, -1))
        basis = torch.stack(bases, dim=-2)
        dot = (state.unsqueeze(-2) * basis).sum(dim=-1)
        norm2 = basis.square().sum(dim=-1)

        tau_features = torch.stack(
            (tau, torch.sin(math.pi * tau), torch.cos(math.pi * tau), torch.sin(2 * math.pi * tau), torch.cos(2 * math.pi * tau)),
            dim=1,
        )
        scalar = torch.cat((scalar_condition, tau_features), dim=1)
        scalar = scalar[:, None, None, None, None, :].expand(-1, frames, sublattices, nx, ny, -1)
        if physical_time is None:
            normalized_time = torch.linspace(
                0.0, 1.0, frames, device=state.device, dtype=state.dtype
            )[None].expand(batch, -1)
        else:
            physical_time = physical_time.to(device=state.device, dtype=state.dtype)
            duration = (physical_time[:, -1:] - physical_time[:, :1]).clamp_min(torch.finfo(state.dtype).tiny)
            normalized_time = (physical_time - physical_time[:, :1]) / duration
        time_feature = normalized_time[:, :, None, None, None, None]
        time_feature = time_feature.expand(batch, -1, sublattices, nx, ny, -1)
        pulse_fraction = scalar_condition[:, 6, None, None, None, None, None]
        pulse_active = (time_feature < pulse_fraction).to(state.dtype)
        invariants = torch.cat((dot, norm2, scalar, time_feature, pulse_active), dim=-1)
        invariants = invariants.permute(0, 2, 5, 1, 3, 4).reshape(batch * sublattices, -1, frames, nx, ny)
        hidden = F.silu(self.input(invariants))
        coefficients = self.output(self.blocks(hidden))
        coefficients = coefficients.reshape(batch, sublattices, 2 * self.local_basis_count, frames, nx, ny)
        coefficients = coefficients.permute(0, 3, 1, 4, 5, 2)
        direct, crossed = coefficients.split(self.local_basis_count, dim=-1)
        velocity = (direct.unsqueeze(-1) * basis).sum(dim=-2)
        velocity = velocity + (crossed.unsqueeze(-1) * torch.linalg.cross(state.unsqueeze(-2), basis)).sum(dim=-2)
        velocity = tangent_project(state, velocity)
        velocity[:, 0] = 0.0
        return velocity
