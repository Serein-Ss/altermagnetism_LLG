"""Deterministic same-backbone baseline for conditional spin paths."""
from __future__ import annotations

import torch
from torch import nn

from .network import PeriodicEquivariantFlowNet
from .sphere import sphere_exp


class DeterministicPathModel(nn.Module):
    """Predict one conditional path without a stochastic reference sample."""

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
        self.backbone = PeriodicEquivariantFlowNet(
            scalar_condition_dim=scalar_condition_dim,
            vector_condition_count=vector_condition_count,
            hidden=hidden,
            blocks=blocks,
            architecture_version=architecture_version,
            condition_mean=condition_mean,
            condition_std=condition_std,
        )

    def forward(
        self,
        initial: torch.Tensor,
        frames: int,
        scalar_condition: torch.Tensor,
        vector_condition: torch.Tensor,
        physical_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if frames < 2:
            raise ValueError("frames must be at least two")
        anchor = initial[:, None].expand(-1, frames, -1, -1, -1, -1)
        tau = torch.ones(initial.shape[0], device=initial.device, dtype=initial.dtype)
        displacement = self.backbone(
            anchor, tau, initial, scalar_condition, vector_condition, physical_time
        )
        prediction = sphere_exp(anchor, displacement)
        prediction[:, 0] = initial
        return prediction
