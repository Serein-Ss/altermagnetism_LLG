"""A common spin-tensor LLG interface for published benchmark Hamiltonians.

Every model consumes ``[batch, sublattice, nx, ny, xyz]`` unit-spin tensors.
The time and energy units are model metadata: the Gomonay model uses SI units;
the Nishino benchmark uses the dimensionless units stated in that paper.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal, Protocol, Sequence

import torch

from altermagnet_dynamics import (
    K_B,
    LiteratureParameters,
    Ruo2DoubleLayerHamiltonian,
    unit,
)


class HamiltonianModel(Protocol):
    gamma: float
    moment: float
    boltzmann: float
    unit_system: str

    def energy(self, spins: torch.Tensor) -> torch.Tensor: ...
    def field(self, spins: torch.Tensor) -> torch.Tensor: ...
    def metadata(self) -> dict: ...


def validate_common_spins(spins: torch.Tensor) -> None:
    if spins.ndim != 5 or spins.shape[1] < 1 or spins.shape[-1] != 3:
        raise ValueError("spins must have shape [batch, sublattice, nx, ny, 3]")
    if not torch.isfinite(spins).all():
        raise ValueError("spins must be finite")


@dataclass(frozen=True)
class NishinoFreeMomentParameters:
    """Published Fig. 1 benchmark (PRB 91, 134411, 2015)."""

    field_h: float = 2.0
    moment_m: float = 1.0
    gamma: float = 1.0
    boltzmann: float = 1.0


class NishinoFreeMomentHamiltonian:
    """Noninteracting moments in a z field, with an exact Langevin mean."""

    unit_system = "Nishino2015_dimensionless_gamma_equals_kB_equals_1"

    def __init__(self, parameters: NishinoFreeMomentParameters | None = None):
        self.parameters = parameters or NishinoFreeMomentParameters()
        if min(
            self.parameters.field_h,
            self.parameters.moment_m,
            self.parameters.gamma,
            self.parameters.boltzmann,
        ) <= 0:
            raise ValueError("Nishino benchmark parameters must be positive")
        self.gamma = self.parameters.gamma
        self.moment = self.parameters.moment_m
        self.boltzmann = self.parameters.boltzmann

    def energy(self, spins: torch.Tensor) -> torch.Tensor:
        validate_common_spins(spins)
        if spins.shape[1] != 1:
            raise ValueError("the free-moment benchmark has one sublattice")
        return -self.parameters.field_h * self.moment * spins[..., 2].sum(
            dim=(1, 2, 3)
        )

    def field(self, spins: torch.Tensor) -> torch.Tensor:
        validate_common_spins(spins)
        field = torch.zeros_like(spins)
        field[..., 2] = self.parameters.field_h
        return field

    def metadata(self) -> dict:
        return {
            "model": "Nishino2015_noninteracting_moments_Fig1",
            "doi": "10.1103/PhysRevB.91.134411",
            "erratum_doi": "10.1103/PhysRevB.97.019904",
            "parameters": asdict(self.parameters),
            "unit_system": self.unit_system,
            "boundary": "irrelevant_noninteracting",
        }

    def exact_magnetization_z(self, temperature: float) -> float:
        if temperature <= 0:
            return self.moment
        x = self.parameters.field_h * self.moment / (
            self.parameters.boltzmann * temperature
        )
        return self.moment * (1.0 / math.tanh(x) - 1.0 / x)


class GomonayModelAdapter:
    """Expose the published double-layer model through the common interface."""

    unit_system = "SI_energy_J_field_T_time_s_temperature_K"

    def __init__(self, *, alternating_exchange: bool = True):
        params = LiteratureParameters(
            j_tilde_mev=LiteratureParameters().j_tilde_mev
            if alternating_exchange
            else 0.0
        )
        self.model = Ruo2DoubleLayerHamiltonian(params)
        self.parameters = params
        self.gamma = params.gamma
        self.moment = params.mu_s
        self.boltzmann = K_B
        self.alternating_exchange = alternating_exchange

    def energy(self, spins: torch.Tensor) -> torch.Tensor:
        return self.model.energy(spins)

    def field(self, spins: torch.Tensor) -> torch.Tensor:
        return self.model.field(spins)

    def metadata(self) -> dict:
        return {
            "model": (
                "Gomonay2024_double_layer_altermagnet"
                if self.alternating_exchange
                else "Gomonay2024_Jtilde_zero_control"
            ),
            "doi": "10.1038/s44306-024-00042-3",
            "parameters": asdict(self.parameters),
            "unit_system": self.unit_system,
            "boundary": "periodic_xy",
            "certification_role": (
                "published_hamiltonian"
                if self.alternating_exchange
                else "controlled_ablation_not_a_separate_material_claim"
            ),
        }


@dataclass(frozen=True)
class CrNb3S6Parameters:
    """Micromagnetic parameters reported for CrNb3S6 in Laliena et al. (2020)."""

    exchange_a_j_per_m: float = 1.42e-12
    dmi_d_j_per_m2: float = 369e-6
    anisotropy_k_j_per_m3: float = -124e3
    saturation_magnetization_a_per_m: float = 129e3
    cell_m: float = 0.6e-9
    gamma: float = 1.76085963023e11


class CrNb3S6Helimagnet:
    """Periodic one-dimensional discretization of the published monoaxial model.

    The chiral axis is stored on the x index of the common tensor.  ``ny`` must
    be one.  The continuum energy density is

    ``A |d_x n|^2 - D z.(n x d_x n) - K n_z^2 - M_s B.n``.
    """

    unit_system = "SI_energy_J_field_T_time_s_temperature_K"

    def __init__(
        self,
        parameters: CrNb3S6Parameters | None = None,
        field_t: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.parameters = parameters or CrNb3S6Parameters()
        p = self.parameters
        if min(
            p.exchange_a_j_per_m,
            p.dmi_d_j_per_m2,
            p.saturation_magnetization_a_per_m,
            p.cell_m,
            p.gamma,
        ) <= 0:
            raise ValueError("CrNb3S6 A, D, Ms, cell and gamma must be positive")
        if not all(math.isfinite(v) for v in (*asdict(p).values(), *field_t)):
            raise ValueError("CrNb3S6 parameters and field must be finite")
        self.gamma = p.gamma
        self.moment = p.saturation_magnetization_a_per_m * p.cell_m**3
        self.boltzmann = K_B
        self.field_t = tuple(float(v) for v in field_t)

    @property
    def continuum_wavevector_per_m(self) -> float:
        return self.parameters.dmi_d_j_per_m2 / (
            2.0 * self.parameters.exchange_a_j_per_m
        )

    @property
    def continuum_period_m(self) -> float:
        return 2.0 * math.pi / self.continuum_wavevector_per_m

    @property
    def discrete_twist_rad(self) -> float:
        p = self.parameters
        return math.atan2(
            p.dmi_d_j_per_m2 * p.cell_m,
            2.0 * p.exchange_a_j_per_m,
        )

    def _validate(self, spins: torch.Tensor) -> None:
        validate_common_spins(spins)
        if spins.shape[1] != 1 or spins.shape[3] != 1:
            raise ValueError("CrNb3S6 helimagnet requires [batch,1,nx,1,3]")

    def energy(self, spins: torch.Tensor) -> torch.Tensor:
        self._validate(spins)
        p = self.parameters
        nxt = torch.roll(spins, shifts=-1, dims=2)
        volume = p.cell_m**3
        exchange = p.exchange_a_j_per_m * volume / p.cell_m**2
        dmi = p.dmi_d_j_per_m2 * volume / p.cell_m
        bond = exchange * ((nxt - spins) ** 2).sum(dim=-1)
        chirality = torch.linalg.cross(spins, nxt, dim=-1)[..., 2]
        anisotropy = -p.anisotropy_k_j_per_m3 * volume * spins[..., 2] ** 2
        field = spins.new_tensor(self.field_t)
        zeeman = -p.saturation_magnetization_a_per_m * volume * (
            spins * field
        ).sum(dim=-1)
        return (bond - dmi * chirality + anisotropy + zeeman).sum(
            dim=(1, 2, 3)
        )

    def field(self, spins: torch.Tensor) -> torch.Tensor:
        self._validate(spins)
        p = self.parameters
        nxt = torch.roll(spins, shifts=-1, dims=2)
        prev = torch.roll(spins, shifts=1, dims=2)
        volume = p.cell_m**3
        exchange = p.exchange_a_j_per_m * volume / p.cell_m**2
        dmi = p.dmi_d_j_per_m2 * volume / p.cell_m
        grad = 2.0 * exchange * (2.0 * spins - prev - nxt)
        z_axis = spins.new_tensor((0.0, 0.0, 1.0)).expand_as(spins)
        grad = grad + dmi * torch.linalg.cross(z_axis, nxt - prev, dim=-1)
        grad[..., 2] = grad[..., 2] - (
            2.0 * p.anisotropy_k_j_per_m3 * volume * spins[..., 2]
        )
        return -grad / self.moment + spins.new_tensor(self.field_t)

    def metadata(self) -> dict:
        return {
            "model": "Laliena2020_CrNb3S6_monoaxial_helimagnet",
            "doi": "10.1038/s41598-020-76903-8",
            "parameters": asdict(self.parameters),
            "field_t": self.field_t,
            "unit_system": self.unit_system,
            "boundary": "periodic_chiral_axis",
            "published_zero_field_period_nm": 48.0,
            "continuum_period_nm_from_parameters": self.continuum_period_m * 1e9,
        }


@dataclass(frozen=True)
class BauerChainParameters:
    """Dimensionless Fig. 2 parameters from Bauer et al., JPCM 23, 394204."""

    exchange_j: float = 1.0
    anisotropy_k: float = 0.1
    moment: float = 1.0
    gamma: float = 1.0
    boltzmann: float = 1.0


class BauerOpenChainHamiltonian:
    """Nearest-neighbour open chain with uniaxial easy-z anisotropy."""

    unit_system = "Bauer2011_dimensionless_energy_J_time_hbar_over_J"

    def __init__(self, parameters: BauerChainParameters | None = None):
        self.parameters = parameters or BauerChainParameters()
        p = self.parameters
        if min(p.exchange_j, p.anisotropy_k, p.moment, p.gamma, p.boltzmann) <= 0:
            raise ValueError("Bauer chain parameters must be positive")
        self.gamma = p.gamma
        self.moment = p.moment
        self.boltzmann = p.boltzmann

    def _validate(self, spins: torch.Tensor) -> None:
        validate_common_spins(spins)
        if spins.shape[1] != 1 or spins.shape[3] != 1:
            raise ValueError("Bauer chain requires [batch,1,length,1,3]")

    def energy(self, spins: torch.Tensor) -> torch.Tensor:
        self._validate(spins)
        p = self.parameters
        exchange = -p.exchange_j * (spins[:, :, :-1] * spins[:, :, 1:]).sum(
            dim=(1, 2, 3, 4)
        )
        anisotropy = -p.anisotropy_k * (spins[..., 2] ** 2).sum(dim=(1, 2, 3))
        return exchange + anisotropy

    def field(self, spins: torch.Tensor) -> torch.Tensor:
        self._validate(spins)
        p = self.parameters
        field = torch.zeros_like(spins)
        field[:, :, 1:] += p.exchange_j * spins[:, :, :-1]
        field[:, :, :-1] += p.exchange_j * spins[:, :, 1:]
        field[..., 2] += 2.0 * p.anisotropy_k * spins[..., 2]
        return field / p.moment

    @property
    def continuum_domain_wall_width_sites(self) -> float:
        return 2.0 * math.sqrt(
            self.parameters.exchange_j / self.parameters.anisotropy_k
        )

    @property
    def continuum_domain_wall_energy_j(self) -> float:
        return 2.0 * math.sqrt(
            2.0 * self.parameters.exchange_j * self.parameters.anisotropy_k
        )

    def metadata(self) -> dict:
        return {
            "model": "Bauer2011_open_uniaxial_chain_Fig2",
            "doi": "10.1088/0953-8984/23/39/394204",
            "parameters": asdict(self.parameters),
            "unit_system": self.unit_system,
            "boundary": "open_chain",
            "published_fig2": {
                "length": 100,
                "temperature_kBT_over_J": 0.11,
                "damping_lambda": 0.1,
                "duration_hbar_over_J": 750000.0,
            },
        }


@dataclass(frozen=True)
class CommonSOT:
    damping_like: float = 0.0
    field_like: float = 0.0
    polarization: tuple[float, float, float] = (2**-0.5, 2**-0.5, 0.0)
    start: float = 0.0
    end: float = math.inf

    def validate(self) -> None:
        if self.end <= self.start:
            raise ValueError("SOT end must exceed start")
        if abs(math.sqrt(sum(v * v for v in self.polarization)) - 1.0) > 1e-9:
            raise ValueError("SOT polarization must be normalized")

    def active(self, time: float) -> bool:
        return self.start <= time < self.end


class UnifiedLLGSolver:
    """Gilbert LLG plus optional SOT and FDT-consistent thermal field."""

    def __init__(
        self,
        model: HamiltonianModel,
        *,
        alpha: float,
        sot: CommonSOT | None = None,
        equation: Literal["gilbert", "bauer_ll"] = "gilbert",
    ):
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.model = model
        self.alpha = float(alpha)
        self.sot = sot or CommonSOT()
        self.sot.validate()
        if equation not in ("gilbert", "bauer_ll"):
            raise ValueError("equation must be 'gilbert' or 'bauer_ll'")
        if equation == "bauer_ll" and (
            self.sot.damping_like or self.sot.field_like
        ):
            raise ValueError("the Bauer LL literature convention does not include SOT")
        self.equation = equation

    def rhs(
        self,
        spins: torch.Tensor,
        time: float,
        extra_field: torch.Tensor | None = None,
    ) -> torch.Tensor:
        deterministic_field = self.model.field(spins)
        field = deterministic_field
        if extra_field is not None:
            field = field + extra_field
        if self.equation == "bauer_ll":
            precession = -self.model.gamma * torch.linalg.cross(spins, field)
            damping = -self.alpha * self.model.gamma * torch.linalg.cross(
                spins,
                torch.linalg.cross(spins, deterministic_field),
            )
            return precession + damping
        torque = -self.model.gamma * torch.linalg.cross(spins, field)
        if self.sot.active(time) and (self.sot.damping_like or self.sot.field_like):
            sigma = spins.new_tensor(self.sot.polarization).expand_as(spins)
            torque = torque + self.model.gamma * self.sot.field_like * torch.linalg.cross(
                sigma, spins
            )
            torque = torque + self.model.gamma * self.sot.damping_like * torch.linalg.cross(
                spins, torch.linalg.cross(sigma, spins)
            )
        return (torque + self.alpha * torch.linalg.cross(spins, torque)) / (
            1.0 + self.alpha**2
        )

    def thermal_field_std(self, temperature: float, dt: float) -> float:
        if temperature < 0 or dt <= 0:
            raise ValueError("temperature must be nonnegative and dt positive")
        return math.sqrt(self.thermal_field_covariance_prefactor(temperature) / dt)

    def thermal_field_covariance_prefactor(self, temperature: float) -> float:
        """Return ``Q_B`` for ``<B(t)B(t')> = Q_B delta(t-t')``."""
        if temperature < 0:
            raise ValueError("temperature must be nonnegative")
        return (
            2.0
            * self.alpha
            * self.model.boltzmann
            * temperature
            / (self.model.gamma * self.model.moment)
        )

    def thermal_energy_noise_covariance_prefactor(self, temperature: float) -> float:
        """Return ``Q_R`` for ``R = moment * B`` in the energy convention."""
        return self.model.moment**2 * self.thermal_field_covariance_prefactor(
            temperature
        )

    def sample_thermal_field(
        self,
        spins: torch.Tensor,
        temperature: float,
        dt: float,
        generator: torch.Generator | Sequence[torch.Generator],
    ) -> torch.Tensor:
        """Draw a replayable thermal field for either one stream or one stream/path.

        A sequence of generators makes every trajectory independently replayable while
        retaining batched Hamiltonian evaluation.  The paths must occupy axis zero.
        """
        scale = self.thermal_field_std(temperature, dt)
        if isinstance(generator, torch.Generator):
            return torch.randn(
                spins.shape,
                generator=generator,
                device=spins.device,
                dtype=spins.dtype,
            ) * scale
        generators = tuple(generator)
        if len(generators) != spins.shape[0]:
            raise ValueError("one thermal RNG generator is required per batch path")
        return torch.stack(
            [
                torch.randn(
                    spins[i].shape,
                    generator=path_generator,
                    device=spins.device,
                    dtype=spins.dtype,
                )
                for i, path_generator in enumerate(generators)
            ],
            dim=0,
        ) * scale

    def stochastic_heun_step(
        self,
        spins: torch.Tensor,
        time: float,
        dt: float,
        temperature: float,
        generator: torch.Generator | Sequence[torch.Generator],
    ) -> torch.Tensor:
        noise = self.sample_thermal_field(spins, temperature, dt, generator)
        f0 = self.rhs(spins, time, noise)
        predictor = unit(spins + dt * f0)
        f1 = self.rhs(predictor, time + dt, noise)
        return unit(spins + 0.5 * dt * (f0 + f1))


def collinear_state(
    batch: int,
    sublattices: int,
    nx: int,
    ny: int,
    *,
    antiferromagnetic: bool,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    spins = torch.zeros((batch, sublattices, nx, ny, 3), device=device, dtype=dtype)
    spins[..., 2] = 1.0
    if antiferromagnetic:
        if sublattices != 2:
            raise ValueError("antiferromagnetic initialization requires two sublattices")
        spins[:, 1, ..., 2] = -1.0
    return spins


def common_observables(spins: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean magnetization and Neel vector for every batch member."""
    validate_common_spins(spins)
    sub_mean = spins.mean(dim=(2, 3))
    magnetization = sub_mean.mean(dim=1)
    if spins.shape[1] == 2:
        neel = 0.5 * (sub_mean[:, 0] - sub_mean[:, 1])
    else:
        neel = torch.zeros_like(magnetization)
    return magnetization, neel
