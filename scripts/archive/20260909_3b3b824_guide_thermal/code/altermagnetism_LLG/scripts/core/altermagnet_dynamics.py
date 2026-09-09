"""Literature-anchored atomistic LLG for a double-layer d-wave altermagnet.

Spins have shape ``[..., 2, Nx, Ny, 3]`` and unit length. Energies are in joule,
fields in tesla, time in second, and angular frequencies in rad/s. The discrete
Hamiltonian follows Supplementary Eq. (S.1) of Gomonay et al., npj Spintronics
2, 35 (2024). Periodic boundaries are used in x and y.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Optional

import torch


MEV_TO_J = 1.602176634e-22
MU_B = 9.2740100783e-24
GAMMA_E = 1.76085963023e11
K_B = 1.380649e-23


def unit(spins: torch.Tensor) -> torch.Tensor:
    return spins / torch.linalg.vector_norm(spins, dim=-1, keepdim=True)


def validate_spins(spins: torch.Tensor) -> None:
    if spins.ndim < 5 or spins.shape[-4] != 2 or spins.shape[-1] != 3:
        raise ValueError("spins must have shape [..., 2, Nx, Ny, 3]")
    if not torch.isfinite(spins).all():
        raise ValueError("spins must be finite")


@dataclass(frozen=True)
class LiteratureParameters:
    """Parameters from Supplementary Table I (domain-wall parameter set)."""

    j1_mev: float = 11.1
    j2_mev: float = 1.88
    j_tilde_mev: float = 0.8
    anisotropy_mev: float = 0.047
    mu_s_mu_b: float = 1.0
    lattice_nm: float = 0.448
    gamma: float = GAMMA_E

    def validate(self) -> None:
        values = tuple(asdict(self).values())
        if not all(math.isfinite(x) for x in values):
            raise ValueError("all literature parameters must be finite")
        if min(self.j1_mev, self.j2_mev, self.mu_s_mu_b,
               self.lattice_nm, self.gamma) <= 0 or self.anisotropy_mev < 0:
            raise ValueError(
                "J1, J2, mu_s, a0 and gamma must be positive; K must be non-negative"
            )

    @property
    def j1(self) -> float:
        return self.j1_mev * MEV_TO_J

    @property
    def j2(self) -> float:
        return self.j2_mev * MEV_TO_J

    @property
    def j_tilde(self) -> float:
        return self.j_tilde_mev * MEV_TO_J

    @property
    def anisotropy(self) -> float:
        return self.anisotropy_mev * MEV_TO_J

    @property
    def mu_s(self) -> float:
        return self.mu_s_mu_b * MU_B

    @property
    def lattice(self) -> float:
        return self.lattice_nm * 1e-9


class Ruo2DoubleLayerHamiltonian:
    """Periodic double-layer Hamiltonian with d-wave alternating exchange.

    Sublattice 1 is located at R and sublattice 2 at
    R + a0(ex+ey)/2. The four J1 bonds, axial J2 bonds, and signed diagonal
    J_tilde bonds follow Supplementary Fig. S1 and Eq. (S.1).
    """

    def __init__(self, parameters: Optional[LiteratureParameters] = None):
        self.parameters = parameters or LiteratureParameters()
        self.parameters.validate()

    @staticmethod
    def _roll(x: torch.Tensor, sx: int, sy: int) -> torch.Tensor:
        return torch.roll(x, shifts=(sx, sy), dims=(-3, -2))

    def energy(self, spins: torch.Tensor) -> torch.Tensor:
        validate_spins(spins)
        p = self.parameters
        m1, m2 = spins.unbind(dim=-4)

        # M1(R) couples to the four M2 sites displaced by half diagonals.
        m2_cross = (m2 + self._roll(m2, 1, 0) + self._roll(m2, 0, 1)
                    + self._roll(m2, 1, 1))
        inter = p.j1 * (m1 * m2_cross).sum(dim=(-3, -2, -1))

        intra = spins.new_zeros(inter.shape)
        for m in (m1, m2):
            intra = intra - p.j2 * (
                m * (self._roll(m, -1, 0) + self._roll(m, 0, -1))
            ).sum(dim=(-3, -2, -1))

        # +J_tilde on the [-x,+y] diagonal for M1 and -J_tilde for M2;
        # signs reverse on the [+x,+y] diagonal.
        nw1, nw2 = self._roll(m1, 1, -1), self._roll(m2, 1, -1)
        ne1, ne2 = self._roll(m1, -1, -1), self._roll(m2, -1, -1)
        alt = p.j_tilde * (
            (m1 * nw1).sum(dim=(-3, -2, -1))
            - (m2 * nw2).sum(dim=(-3, -2, -1))
            - (m1 * ne1).sum(dim=(-3, -2, -1))
            + (m2 * ne2).sum(dim=(-3, -2, -1))
        )
        anis = -p.anisotropy * (spins[..., 2] ** 2).sum(dim=(-3, -2, -1))
        return inter + intra + alt + anis

    def field(self, spins: torch.Tensor) -> torch.Tensor:
        """Return B_eff = -(1/mu_s) dH/ds in tesla."""
        validate_spins(spins)
        p = self.parameters
        m1, m2 = spins.unbind(dim=-4)

        cross1 = (m2 + self._roll(m2, 1, 0) + self._roll(m2, 0, 1)
                  + self._roll(m2, 1, 1))
        cross2 = (m1 + self._roll(m1, -1, 0) + self._roll(m1, 0, -1)
                  + self._roll(m1, -1, -1))

        fields = []
        for sublattice, m, cross in ((1, m1, cross1), (2, m2, cross2)):
            axial = (self._roll(m, -1, 0) + self._roll(m, 1, 0)
                     + self._roll(m, 0, -1) + self._roll(m, 0, 1))
            diag_plus = self._roll(m, 1, -1) + self._roll(m, -1, 1)
            diag_minus = self._roll(m, -1, -1) + self._roll(m, 1, 1)
            sign = 1.0 if sublattice == 1 else -1.0
            alt_derivative = sign * p.j_tilde * (diag_plus - diag_minus)
            anis_field = torch.zeros_like(m)
            anis_field[..., 2] = 2.0 * p.anisotropy * m[..., 2]
            field = (-p.j1 * cross + p.j2 * axial - alt_derivative
                     + anis_field) / p.mu_s
            fields.append(field)
        return torch.stack(fields, dim=-4)


@dataclass(frozen=True)
class SOTPulse:
    """Field-like and damping-like SOT fields in the Gilbert equation."""

    damping_like_t: float = 0.0
    field_like_t: float = 0.0
    polarization: tuple[float, float, float] = (2**-0.5, 2**-0.5, 0.0)
    start_s: float = 0.0
    end_s: float = math.inf

    def validate(self) -> None:
        vals = (self.damping_like_t, self.field_like_t, *self.polarization,
                self.start_s)
        if not all(math.isfinite(x) for x in vals):
            raise ValueError("finite SOT parameters are required")
        if not (math.isfinite(self.end_s) or self.end_s == math.inf):
            raise ValueError("end_s must be finite or +inf")
        if self.end_s <= self.start_s:
            raise ValueError("SOT end time must be after start time")
        norm = math.sqrt(sum(x*x for x in self.polarization))
        if abs(norm - 1.0) > 1e-9:
            raise ValueError("SOT polarization must be a unit vector")

    def active(self, t: float) -> bool:
        return self.start_s <= t < self.end_s


class LLGDynamics:
    def __init__(self, model: Ruo2DoubleLayerHamiltonian, alpha: float = 0.01,
                 sot: Optional[SOTPulse] = None):
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        self.model = model
        self.alpha = float(alpha)
        self.sot = sot or SOTPulse()
        self.sot.validate()

    def rhs(self, spins: torch.Tensor, t: float = 0.0,
            extra_field: Optional[torch.Tensor] = None) -> torch.Tensor:
        field = self.model.field(spins)
        if extra_field is not None:
            field = field + extra_field
        gamma = self.model.parameters.gamma
        torque0 = -gamma * torch.linalg.cross(spins, field)
        if self.sot.active(t):
            sigma = spins.new_tensor(self.sot.polarization).expand_as(spins)
            torque0 = torque0 + gamma * self.sot.field_like_t * torch.linalg.cross(sigma, spins)
            torque0 = torque0 + gamma * self.sot.damping_like_t * torch.linalg.cross(
                spins, torch.linalg.cross(sigma, spins)
            )
        return (torque0 + self.alpha * torch.linalg.cross(spins, torque0)) / (
            1.0 + self.alpha**2
        )


def projected_rk4_step(spins: torch.Tensor, t: float, dt: float,
                       dynamics: LLGDynamics) -> torch.Tensor:
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    f = dynamics.rhs
    k1 = f(spins, t)
    k2 = f(unit(spins + 0.5*dt*k1), t + 0.5*dt)
    k3 = f(unit(spins + 0.5*dt*k2), t + 0.5*dt)
    k4 = f(unit(spins + dt*k3), t + dt)
    return unit(spins + dt*(k1 + 2*k2 + 2*k3 + k4)/6)


def thermal_field_std(parameters: LiteratureParameters, temperature_k: float,
                      dt: float, alpha: float) -> float:
    """FDT white-noise magnetic-field standard deviation for one time step.

    The continuous atomistic convention is

    ``<B_mu,i(t) B_nu,j(t')> = Q_B delta_mu,nu delta_i,j delta(t-t')``

    with ``Q_B = 2 alpha k_B T / (gamma mu_s)``.  Averaging white noise over
    a step ``dt`` gives variance ``Q_B / dt``.  If the random variable is
    instead defined as the energy-like ``R = mu_s B``, its covariance is
    ``Q_R = 2 alpha (mu_s/gamma) k_B T``.  A 2-D continuum expression writes
    the same discrete-site covariance with a cell-area factor multiplying the
    spatial Dirac delta.
    """
    if not all(math.isfinite(x) for x in (temperature_k, dt, alpha)):
        raise ValueError("temperature, dt and alpha must be finite")
    if temperature_k < 0 or dt <= 0 or alpha < 0:
        raise ValueError("require temperature >= 0, dt > 0 and alpha >= 0")
    return math.sqrt(
        thermal_field_covariance_prefactor(parameters, temperature_k, alpha) / dt
    )


def thermal_field_covariance_prefactor(
    parameters: LiteratureParameters, temperature_k: float, alpha: float
) -> float:
    """Return ``Q_B`` in ``<B(t)B(t')> = Q_B delta(t-t')``."""
    if not all(math.isfinite(x) for x in (temperature_k, alpha)):
        raise ValueError("temperature and alpha must be finite")
    if temperature_k < 0 or alpha < 0:
        raise ValueError("temperature and alpha must be nonnegative")
    return 2.0 * alpha * K_B * temperature_k / (
        parameters.gamma * parameters.mu_s
    )


def thermal_energy_noise_covariance_prefactor(
    parameters: LiteratureParameters, temperature_k: float, alpha: float
) -> float:
    """Return ``Q_R`` for the energy-like random variable ``R = mu_s B``."""
    return parameters.mu_s**2 * thermal_field_covariance_prefactor(
        parameters, temperature_k, alpha
    )


def sample_thermal_field(
    spins: torch.Tensor,
    parameters: LiteratureParameters,
    temperature_k: float,
    dt: float,
    alpha: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw independent zero-mean Cartesian/site/time thermal fields in tesla."""
    return torch.randn(
        spins.shape,
        generator=generator,
        device=spins.device,
        dtype=spins.dtype,
    ) * thermal_field_std(parameters, temperature_k, dt, alpha)


def stochastic_heun_step(spins: torch.Tensor, t: float, dt: float,
                         dynamics: LLGDynamics, temperature_k: float,
                         generator: torch.Generator) -> torch.Tensor:
    """Stratonovich stochastic Heun step using one noise draw twice."""
    noise = sample_thermal_field(
        spins,
        dynamics.model.parameters,
        temperature_k,
        dt,
        dynamics.alpha,
        generator,
    )
    f0 = dynamics.rhs(spins, t, noise)
    predictor = unit(spins + dt*f0)
    f1 = dynamics.rhs(predictor, t + dt, noise)
    return unit(spins + 0.5*dt*(f0 + f1))


def antiferromagnetic_state(batch: int, nx: int, ny: int, *,
                            device: str = "cpu",
                            dtype: torch.dtype = torch.float64) -> torch.Tensor:
    if min(batch, nx, ny) < 1:
        raise ValueError("batch and lattice dimensions must be positive")
    spins = torch.zeros((batch, 2, nx, ny, 3), device=device, dtype=dtype)
    spins[:, 0, ..., 2] = 1.0
    spins[:, 1, ..., 2] = -1.0
    return spins


def order_parameters(spins: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    validate_spins(spins)
    m1, m2 = spins.unbind(dim=-4)
    neel = 0.5 * (m1 - m2)
    magnetization = 0.5 * (m1 + m2)
    return neel, magnetization


def c4_sublattice_transform(spins: torch.Tensor) -> torch.Tensor:
    """C4 lattice rotation with the RuO2 sublattice/registry exchange.

    Because sublattice 2 is shifted by (a0/2, a0/2), exchanging the two
    sublattices after ``torch.rot90`` requires a one-cell relative y shift.
    This convention is an exact symmetry of Supplementary Eq. (S.1).
    """
    validate_spins(spins)
    m1, m2 = spins.unbind(dim=-4)
    rotated_m1 = torch.rot90(m2, 1, dims=(-3, -2))
    rotated_m2 = torch.roll(torch.rot90(m1, 1, dims=(-3, -2)),
                            shifts=-1, dims=-2)
    return torch.stack((rotated_m1, rotated_m2), dim=-4)


def analytic_spinwave_omega(parameters: LiteratureParameters,
                            kx: torch.Tensor, ky: torch.Tensor,
                            anisotropy_mev: Optional[float] = None
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Published linear spin-wave branches, Supplementary Eq. (S.9)."""
    k = parameters.anisotropy_mev if anisotropy_mev is None else anisotropy_mev
    kappa = k / parameters.j1_mev
    eta = parameters.j2_mev / parameters.j1_mev
    epsilon = parameters.j_tilde_mev / parameters.j1_mev
    ax, ay = parameters.lattice * kx, parameters.lattice * ky
    a_k = torch.cos(ax/2) * torch.cos(ay/2)
    b_k = 1 + kappa/2 + eta*(torch.sin(ax/2)**2 + torch.sin(ay/2)**2)
    c_k = epsilon * torch.sin(ax) * torch.sin(ay)
    base = torch.sqrt(torch.clamp(b_k*b_k - a_k*a_k, min=0.0))
    omega0 = 4 * parameters.gamma * parameters.j1 / parameters.mu_s
    return omega0*(base - c_k), omega0*(base + c_k)
