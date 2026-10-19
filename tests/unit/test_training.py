from __future__ import annotations

from pathlib import Path
import sys
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from altermagnetism_LLG.scripts.training.train import epoch_loss


class ConstantPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        initial,
        frames,
        scalar_condition,
        vector_condition,
        physical_time,
    ):
        shape = (initial.shape[0], frames) + initial.shape[1:]
        return self.value.expand(shape)


def test_partial_gradient_accumulation_group_uses_its_actual_size():
    def batch(target):
        return {
            "spins": torch.full((1, 2, 1, 1, 1, 3), target),
            "initial": torch.zeros((1, 1, 1, 1, 3)),
            "physical_time": torch.tensor([[0.0, 1.0]]),
            "scalar_condition": torch.zeros((1, 10)),
            "vector_condition": torch.zeros((1, 5, 3)),
            "sample_weight": torch.ones(1),
        }

    model = ConstantPath()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = epoch_loss(
        model,
        "deterministic",
        [batch(0.01), batch(0.02), batch(0.03)],
        "cpu",
        optimizer=optimizer,
        gradient_accumulation=4,
    )

    assert loss == pytest.approx(0.0014)
    assert model.value.item() == pytest.approx(0.012)
