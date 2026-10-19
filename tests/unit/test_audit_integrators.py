import torch
import pytest

from scripts.core.audit_integrators import coupled_fields, heun_with_field, midpoint_with_field
from scripts.core.unified_llg import NishinoFreeMomentHamiltonian, UnifiedLLGSolver, collinear_state


def fixture():
    solver = UnifiedLLGSolver(NishinoFreeMomentHamiltonian(), alpha=.05)
    state = collinear_state(1, 1, 8, 1, antiferromagnetic=False, device="cpu", dtype=torch.float64)
    return solver, state


def test_heun_matches_existing_solver_with_identical_noise():
    solver, state = fixture()
    field = solver.sample_thermal_field(state, 2., .005, torch.Generator().manual_seed(1))
    actual, errors = heun_with_field(solver, state, 0., .005, field)
    expected = solver.stochastic_heun_step(state, 0., .005, 2., torch.Generator().manual_seed(1))
    assert torch.allclose(actual, expected, atol=1e-14, rtol=1e-14)
    assert errors.shape == (2,)


def test_coupling_preserves_integrated_brownian_field():
    solver, state = fixture()
    coarse, medium, fine = coupled_fields(solver, state, 2., .005, torch.Generator().manual_seed(3))
    assert torch.allclose(coarse*.005, fine.sum(0)*(.005/4), atol=1e-14)
    assert torch.allclose(medium*.0025, fine.reshape(2, 2, *state.shape).sum(1)*(.005/4), atol=1e-14)


def test_midpoint_keeps_norm_without_projection_and_solves_equation():
    solver, state = fixture()
    field = solver.sample_thermal_field(state, 2., .005, torch.Generator().manual_seed(2))
    result = midpoint_with_field(solver, state, 0., .005, field)
    assert (result.norm(dim=-1)-1).abs().max() < 1e-12
    residual = result-state-.005*solver.rhs((state+result)/2, .0025, field)
    assert residual.abs().max() < 1e-12


def test_midpoint_failure_is_not_silently_normalized():
    solver, state = fixture()
    with pytest.raises(RuntimeError, match="did not converge"):
        midpoint_with_field(solver, state, 0., .005, torch.ones_like(state), max_iterations=1)
