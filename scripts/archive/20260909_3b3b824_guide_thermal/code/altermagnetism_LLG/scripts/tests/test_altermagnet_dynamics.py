from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "core"))

from altermagnet_dynamics import (  # noqa: E402
    LLGDynamics,
    LiteratureParameters,
    Ruo2DoubleLayerHamiltonian,
    antiferromagnetic_state,
    c4_sublattice_transform,
    projected_rk4_step,
    stochastic_heun_step,
    thermal_energy_noise_covariance_prefactor,
    thermal_field_covariance_prefactor,
    thermal_field_std,
    unit,
)


def random_spins(seed=0, size=5):
    generator = torch.Generator().manual_seed(seed)
    return unit(torch.randn((2, 2, size, size, 3), generator=generator,
                            dtype=torch.float64))


def test_analytic_field_matches_autograd():
    model = Ruo2DoubleLayerHamiltonian()
    spins = random_spins().requires_grad_(True)
    energy = model.energy(spins).sum()
    gradient = torch.autograd.grad(energy, spins)[0]
    expected = -gradient / model.parameters.mu_s
    actual = model.field(spins.detach())
    assert torch.allclose(actual, expected, rtol=2e-12, atol=2e-8)


def test_periodic_translation_invariance():
    model = Ruo2DoubleLayerHamiltonian()
    spins = random_spins()
    shifted = torch.roll(spins, shifts=(2, -1), dims=(-3, -2))
    assert torch.allclose(model.energy(spins), model.energy(shifted), rtol=1e-14,
                          atol=1e-32)


def test_c4_rotation_plus_sublattice_exchange_invariance():
    model = Ruo2DoubleLayerHamiltonian()
    spins = random_spins(size=6)
    transformed = c4_sublattice_transform(spins)
    assert torch.allclose(model.energy(spins), model.energy(transformed),
                          rtol=1e-14, atol=1e-32)


def test_damped_zero_temperature_dynamics_conserves_norm_and_lowers_energy():
    model = Ruo2DoubleLayerHamiltonian()
    dynamics = LLGDynamics(model, alpha=0.1)
    spins = random_spins(seed=3, size=4)[:1]
    initial = float(model.energy(spins)[0])
    for step in range(100):
        spins = projected_rk4_step(spins, step*1e-17, 1e-17, dynamics)
    final = float(model.energy(spins)[0])
    assert final < initial
    assert torch.max(torch.abs(torch.linalg.vector_norm(spins, dim=-1)-1)) < 1e-12


def test_thermal_amplitude_and_seed_reproducibility():
    params = LiteratureParameters()
    assert thermal_field_std(params, 0.0, 1e-17, 0.01) == 0.0
    assert thermal_field_std(params, 200.0, 1e-17, 0.01) > 0.0
    model = Ruo2DoubleLayerHamiltonian(params)
    dynamics = LLGDynamics(model, alpha=0.01)
    spins = antiferromagnetic_state(1, 3, 3)
    g1 = torch.Generator().manual_seed(9)
    g2 = torch.Generator().manual_seed(9)
    g3 = torch.Generator().manual_seed(10)
    a = stochastic_heun_step(spins, 0.0, 1e-17, dynamics, 50.0, g1)
    b = stochastic_heun_step(spins, 0.0, 1e-17, dynamics, 50.0, g2)
    c = stochastic_heun_step(spins, 0.0, 1e-17, dynamics, 50.0, g3)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert torch.max(torch.abs(torch.linalg.vector_norm(a, dim=-1)-1)) < 1e-12


def test_field_and_energy_noise_conventions_are_equivalent():
    params = LiteratureParameters()
    temperature, alpha = 80.0, 0.03
    q_field = thermal_field_covariance_prefactor(params, temperature, alpha)
    q_energy = thermal_energy_noise_covariance_prefactor(params, temperature, alpha)
    assert torch.isclose(torch.tensor(q_energy, dtype=torch.float64),
                         torch.tensor(params.mu_s**2 * q_field,
                                      dtype=torch.float64),
                         rtol=1e-12, atol=0.0)
    expected = 2 * alpha * (params.mu_s / params.gamma) * 1.380649e-23 * temperature
    assert torch.isclose(torch.tensor(q_energy, dtype=torch.float64),
                         torch.tensor(expected, dtype=torch.float64),
                         rtol=1e-12, atol=0.0)
