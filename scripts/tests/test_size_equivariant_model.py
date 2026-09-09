from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT / "scripts" / "core"))

from altermagnetism_LLG.scripts.model.baseline import DeterministicPathModel  # noqa: E402
from altermagnetism_LLG.scripts.model.network import PeriodicEquivariantFlowNet  # noqa: E402
from altermagnetism_LLG.scripts.model.sphere import geodesic_interpolate, sample_reference_path  # noqa: E402
from unified_llg import GomonayModelAdapter, UnifiedLLGSolver, collinear_state  # noqa: E402


def normalized(shape):
    value = torch.randn(shape, dtype=torch.float64)
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True)


def test_reference_and_geodesic_preserve_sphere_and_initial_frame():
    initial = normalized((2, 2, 4, 5, 3))
    target = normalized((2, 6, 2, 4, 5, 3))
    target[:, 0] = initial
    reference = sample_reference_path(initial, 6)
    state, velocity = geodesic_interpolate(reference, target, torch.tensor([0.3, 0.8], dtype=torch.float64))
    assert torch.allclose(torch.linalg.vector_norm(state, dim=-1), torch.ones_like(state[..., 0]), atol=1e-6)
    assert torch.allclose(state[:, 0], initial, atol=1e-7)
    assert torch.allclose(velocity[:, 0], torch.zeros_like(velocity[:, 0]), atol=1e-7)


def test_network_accepts_new_sizes_and_is_periodic_translation_equivariant():
    torch.manual_seed(3)
    model = PeriodicEquivariantFlowNet(hidden=16, blocks=1).double().eval()
    for nx, ny in ((4, 5), (7, 9)):
        state = normalized((1, 3, 2, nx, ny, 3))
        initial = state[:, 0].clone()
        scalar = torch.randn((1, 10), dtype=torch.float64)
        vector = torch.randn((1, 5, 3), dtype=torch.float64)
        tau = torch.tensor([0.4], dtype=torch.float64)
        output = model(state, tau, initial, scalar, vector)
        shifted = model(
            torch.roll(state, (1, -2), dims=(3, 4)),
            tau,
            torch.roll(initial, (1, -2), dims=(2, 3)),
            scalar,
            vector,
        )
        assert output.shape == state.shape
        assert torch.allclose(shifted, torch.roll(output, (1, -2), dims=(3, 4)), atol=1e-9, rtol=1e-7)


def test_network_is_joint_rotation_covariant():
    torch.manual_seed(5)
    model = PeriodicEquivariantFlowNet(hidden=16, blocks=1).double().eval()
    state = normalized((2, 3, 2, 4, 5, 3))
    initial = state[:, 0].clone()
    scalar = torch.randn((2, 10), dtype=torch.float64)
    vector = torch.randn((2, 5, 3), dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn((3, 3), dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    rotate = lambda value: torch.einsum("...i,ji->...j", value, q)
    tau = torch.tensor([0.6, 0.2], dtype=torch.float64)
    expected = rotate(model(state, tau, initial, scalar, vector))
    actual = model(rotate(state), tau, rotate(initial), scalar, rotate(vector))
    assert torch.allclose(actual, expected, atol=1e-9, rtol=1e-7)


def test_per_trajectory_thermal_streams_are_replayable():
    model = GomonayModelAdapter()
    solver = UnifiedLLGSolver(model, alpha=0.01)
    spins = collinear_state(2, 2, 3, 3, antiferromagnetic=True, device="cpu", dtype=torch.float64)
    batched = solver.sample_thermal_field(
        spins,
        5.0,
        1e-16,
        [torch.Generator().manual_seed(11), torch.Generator().manual_seed(22)],
    )
    first = solver.sample_thermal_field(spins[:1], 5.0, 1e-16, torch.Generator().manual_seed(11))
    second = solver.sample_thermal_field(spins[:1], 5.0, 1e-16, torch.Generator().manual_seed(22))
    assert torch.equal(batched[0], first[0])
    assert torch.equal(batched[1], second[0])


def test_deterministic_baseline_is_covariant_periodic_and_on_sphere():
    torch.manual_seed(8)
    model = DeterministicPathModel(hidden=16, blocks=1).double().eval()
    initial = normalized((1, 2, 4, 5, 3))
    scalar = torch.randn((1, 10), dtype=torch.float64)
    vector = torch.randn((1, 5, 3), dtype=torch.float64)
    physical_time = torch.tensor([[0.0, 0.2, 1.0]], dtype=torch.float64)
    output = model(initial, 3, scalar, vector, physical_time)
    shifted = model(
        torch.roll(initial, (1, -2), dims=(2, 3)),
        3,
        scalar,
        vector,
        physical_time,
    )
    q, _ = torch.linalg.qr(torch.randn((3, 3), dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    rotate = lambda value: torch.einsum("...i,ji->...j", value, q)
    rotated = model(rotate(initial), 3, scalar, rotate(vector), physical_time)
    assert torch.allclose(torch.linalg.vector_norm(output, dim=-1), torch.ones_like(output[..., 0]), atol=1e-9)
    assert torch.allclose(output[:, 0], initial, atol=1e-9)
    assert torch.allclose(shifted, torch.roll(output, (1, -2), dims=(3, 4)), atol=1e-9, rtol=1e-7)
    assert torch.allclose(rotated, rotate(output), atol=1e-9, rtol=1e-7)

def test_v2_uses_pointwise_norm_and_film_without_breaking_equivariance():
    torch.manual_seed(12)
    model = PeriodicEquivariantFlowNet(
        hidden=16,
        blocks=2,
        architecture_version=2,
        condition_mean=[0.0] * 10,
        condition_std=[1.0] * 10,
    ).double().eval()
    assert not any(
        isinstance(module, torch.nn.GroupNorm)
        for module in model.modules()
    )
    assert all(block.film is not None for block in model.blocks)
    state = normalized((1, 3, 2, 4, 5, 3))
    initial = state[:, 0].clone()
    scalar = torch.randn((1, 10), dtype=torch.float64)
    vector = torch.randn((1, 5, 3), dtype=torch.float64)
    tau = torch.tensor([0.4], dtype=torch.float64)
    output = model(state, tau, initial, scalar, vector)
    shifted = model(
        torch.roll(state, (1, -2), dims=(3, 4)),
        tau,
        torch.roll(initial, (1, -2), dims=(2, 3)),
        scalar,
        vector,
    )
    q, _ = torch.linalg.qr(torch.randn((3, 3), dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    rotate = lambda value: torch.einsum("...i,ji->...j", value, q)
    rotated = model(
        rotate(state), tau, rotate(initial), scalar, rotate(vector)
    )
    assert torch.allclose(
        (output * state).sum(dim=-1),
        torch.zeros_like(output[..., 0]),
        atol=1e-9,
        rtol=1e-7,
    )
    assert torch.allclose(output[:, 0], torch.zeros_like(output[:, 0]))
    assert torch.allclose(
        rotated, rotate(output), atol=1e-9, rtol=1e-7
    )
    assert torch.allclose(
        shifted,
        torch.roll(output, (1, -2), dims=(3, 4)),
        atol=1e-9,
        rtol=1e-7,
    )


def test_v1_state_dict_remains_strictly_loadable():
    original = PeriodicEquivariantFlowNet(hidden=16, blocks=1)
    restored = PeriodicEquivariantFlowNet(hidden=16, blocks=1)
    restored.load_state_dict(original.state_dict(), strict=True)
