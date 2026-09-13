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


class PointwiseChannelNorm(nn.Module):
    """Normalize across channels separately at every time and lattice site."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=1, keepdim=True)
        variance = (inputs - mean).square().mean(dim=1, keepdim=True)
        normalized = (inputs - mean) * torch.rsqrt(variance + self.eps)
        weight = self.weight[None, :, None, None, None]
        bias = self.bias[None, :, None, None, None]
        return normalized * weight + bias


class ResidualScalarBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        architecture_version: int = 1,
        film_condition_dim: int = 4,
    ):
        super().__init__()
        self.first = PeriodicConv3d(channels, channels)
        self.second = PeriodicConv3d(channels, channels)
        if architecture_version == 1:
            groups = 8 if channels % 8 == 0 else 1
            self.norm1 = nn.GroupNorm(groups, channels)
            self.norm2 = nn.GroupNorm(groups, channels)
            self.film = None
        else:
            self.norm1 = PointwiseChannelNorm(channels)
            self.norm2 = PointwiseChannelNorm(channels)
            self.film = nn.Linear(film_condition_dim, 4 * channels)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    @staticmethod
    def _modulate(
        inputs: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        scale = scale[:, :, None, None, None]
        shift = shift[:, :, None, None, None]
        return inputs * (1.0 + scale) + shift

    def forward(
        self,
        inputs: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.film is None:
            hidden = F.silu(self.norm1(self.first(inputs)))
            return inputs + self.norm2(self.second(hidden))
        if condition is None:
            raise ValueError("FiLM residual block requires a condition")
        scale1, shift1, scale2, shift2 = self.film(condition).chunk(
            4, dim=1
        )
        hidden = self._modulate(
            self.norm1(self.first(inputs)), scale1, shift1
        )
        hidden = F.silu(hidden)
        residual = self._modulate(
            self.norm2(self.second(hidden)), scale2, shift2
        )
        return inputs + residual


class PeriodicEquivariantFlowNet(nn.Module):
    """Predict one tangent velocity per spin without fixing lattice dimensions.

    Vector outputs are linear combinations of covariant local vector bases.  Their
    coefficients are produced solely from scalar invariants, which gives exact joint
    proper-rotation covariance while circular padding gives periodic translation
    equivariance.  This first implementation intentionally avoids an e3nn dependency.
    """

    def __init__(
        self,
        scalar_condition_dim: int = 10,
        vector_condition_count: int = 5,
        hidden: int = 64,
        blocks: int = 8,
        architecture_version: int = 1,
        condition_mean: list[float] | None = None,
        condition_std: list[float] | None = None,
    ):
        super().__init__()
        if architecture_version not in (1, 2):
            raise ValueError("architecture_version must be 1 or 2")
        self.scalar_condition_dim = scalar_condition_dim
        self.vector_condition_count = vector_condition_count
        self.architecture_version = architecture_version
        self.local_basis_count = 6 + vector_condition_count
        if condition_mean is None:
            condition_mean = [0.0] * scalar_condition_dim
        if condition_std is None:
            condition_std = [1.0] * scalar_condition_dim
        if (
            len(condition_mean) != scalar_condition_dim
            or len(condition_std) != scalar_condition_dim
        ):
            raise ValueError("condition statistics have the wrong length")
        if min(condition_std) <= 0:
            raise ValueError(
                "condition standard deviations must be positive"
            )
        self.register_buffer(
            "condition_mean",
            torch.tensor(condition_mean, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "condition_std",
            torch.tensor(condition_std, dtype=torch.float32),
            persistent=False,
        )
        # Two frame-wise scalar channels identify physical progress and whether
        # the finite-duration drive pulse is active.  They preserve SO(3)
        # covariance because they only modulate invariant coefficients.
        invariant_channels = (
            2 * self.local_basis_count + scalar_condition_dim + 5 + 2
        )
        self.input = PeriodicConv3d(invariant_channels, hidden)
        self.blocks = nn.ModuleList(
            [
                ResidualScalarBlock(
                    hidden,
                    architecture_version=architecture_version,
                )
                for _ in range(blocks)
            ]
        )
        self.output = PeriodicConv3d(
            hidden, 2 * self.local_basis_count
        )

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

        standardized_condition = (
            scalar_condition - self.condition_mean.to(scalar_condition)
        ) / self.condition_std.to(scalar_condition)
        tau_features = torch.stack(
            (tau, torch.sin(math.pi * tau), torch.cos(math.pi * tau), torch.sin(2 * math.pi * tau), torch.cos(2 * math.pi * tau)),
            dim=1,
        )
        scalar = torch.cat((standardized_condition, tau_features), dim=1)
        scalar = scalar[:, None, None, None, None, :].expand(-1, frames, sublattices, nx, ny, -1)
        film_condition = standardized_condition[:, :4]
        film_condition = film_condition.repeat_interleave(
            sublattices, dim=0
        )
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
        for block in self.blocks:
            hidden = block(
                hidden,
                film_condition
                if self.architecture_version == 2 else None,
            )
        coefficients = self.output(hidden)
        coefficients = coefficients.reshape(batch, sublattices, 2 * self.local_basis_count, frames, nx, ny)
        coefficients = coefficients.permute(0, 3, 1, 4, 5, 2)
        direct, crossed = coefficients.split(self.local_basis_count, dim=-1)
        velocity = (direct.unsqueeze(-1) * basis).sum(dim=-2)
        velocity = velocity + (crossed.unsqueeze(-1) * torch.linalg.cross(state.unsqueeze(-2), basis)).sum(dim=-2)
        velocity = tangent_project(state, velocity)
        velocity[:, 0] = 0.0
        return velocity
