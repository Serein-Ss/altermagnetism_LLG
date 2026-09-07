from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "core"))

from unified_llg import (  # noqa: E402
    BauerOpenChainHamiltonian,
    CrNb3S6Helimagnet,
    GomonayModelAdapter,
    NishinoFreeMomentHamiltonian,
    UnifiedLLGSolver,
    collinear_state,
)


def random_unit(shape, seed=0):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(shape, generator=generator, dtype=torch.float64)
    return values / torch.linalg.vector_norm(values, dim=-1, keepdim=True)


def test_nishino_exact_langevin_value():
    model = NishinoFreeMomentHamiltonian()
    assert model.exact_magnetization_z(2.0) == pytest.approx(0.3130352855)


def test_nishino_field_and_energy_share_common_shape():
    model = NishinoFreeMomentHamiltonian()
    spins = collinear_state(3, 1, 4, 5, antiferromagnetic=False, device="cpu")
    assert model.field(spins).shape == spins.shape
    assert model.energy(spins).shape == (3,)
    assert torch.allclose(model.energy(spins), torch.full((3,), -40.0))


def test_unified_thermal_amplitude_is_published_case_a():
    model = NishinoFreeMomentHamiltonian()
    solver = UnifiedLLGSolver(model, alpha=0.05)
    expected = (2 * 0.05 * 2.0 / (1.0 * 1.0 * 0.005)) ** 0.5
    assert solver.thermal_field_std(2.0, 0.005) == pytest.approx(expected)


def test_gomonay_adapter_uses_same_published_field():
    model = GomonayModelAdapter(alternating_exchange=True)
    spins = collinear_state(2, 2, 4, 4, antiferromagnetic=True, device="cpu", dtype=torch.float64)
    assert model.field(spins).shape == spins.shape
    assert model.metadata()["doi"] == "10.1038/s44306-024-00042-3"


@pytest.mark.parametrize(
    "model,shape",
    [
        (CrNb3S6Helimagnet(), (2, 1, 12, 1, 3)),
        (BauerOpenChainHamiltonian(), (2, 1, 12, 1, 3)),
    ],
)
def test_added_literature_models_have_correct_analytic_fields(model, shape):
    spins = random_unit(shape).requires_grad_(True)
    gradient = torch.autograd.grad(model.energy(spins).sum(), spins)[0]
    expected = -gradient / model.moment
    actual = model.field(spins.detach())
    assert torch.allclose(actual, expected, rtol=2e-12, atol=2e-8)


def test_crnb3s6_one_turn_helix_is_stationary_on_matching_period():
    model = CrNb3S6Helimagnet()
    length = 80
    phase = torch.arange(length, dtype=torch.float64) * (2.0 * torch.pi / length)
    spins = torch.zeros((1, 1, length, 1, 3), dtype=torch.float64)
    spins[0, 0, :, 0, 0] = torch.cos(phase)
    spins[0, 0, :, 0, 1] = torch.sin(phase)
    torque = torch.linalg.cross(spins, model.field(spins), dim=-1)
    assert torch.max(torch.linalg.vector_norm(torque, dim=-1)) < 1e-10
    assert model.continuum_period_m * 1e9 == pytest.approx(48.37, rel=2e-3)
