"""Generate a deterministic LLG trajectory and validate published spin waves."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "core"))

from altermagnet_dynamics import (  # noqa: E402
    LLGDynamics,
    LiteratureParameters,
    Ruo2DoubleLayerHamiltonian,
    analytic_spinwave_omega,
    antiferromagnetic_state,
    projected_rk4_step,
    unit,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reproduce d-wave spin-wave splitting")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "data" / "literature" / "literature_validation")
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--mode", type=int, default=2,
                   help="Fourier mode index with kx=ky=2*pi*mode/(N*a0)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--dt", type=float, default=2e-16)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--amplitude", type=float, default=1e-3)
    return p


def peak_near(frequency: np.ndarray, power: np.ndarray,
              expected: float) -> float:
    bin_width = abs(frequency[1] - frequency[0])
    width = max(0.04 * expected, 2.0 * bin_width)
    mask = (frequency > expected - width) & (frequency < expected + width)
    if not np.any(mask):
        raise RuntimeError("frequency window contains no FFT bins")
    local = np.flatnonzero(mask)
    return float(frequency[local[np.argmax(power[mask])]])


@torch.no_grad()
def main() -> None:
    args = parser().parse_args()
    if args.size < 4 or not 0 < args.mode < args.size // 2:
        raise ValueError("require size >= 4 and 0 < mode < size/2")
    if args.steps < 10 or args.dt <= 0 or args.save_every < 1:
        raise ValueError("invalid integration settings")

    # The paper uses K=0 for the spin-wave calculation.
    params = LiteratureParameters(anisotropy_mev=0.0)
    model = Ruo2DoubleLayerHamiltonian(params)
    dynamics = LLGDynamics(model, alpha=0.0)
    spins = antiferromagnetic_state(1, args.size, args.size)

    ix = torch.arange(args.size, dtype=spins.dtype)[:, None]
    iy = torch.arange(args.size, dtype=spins.dtype)[None, :]
    phase = 2 * np.pi * args.mode * (ix + iy) / args.size
    spins[:, 0, ..., 0] = args.amplitude * torch.cos(phase)
    spins = unit(spins)
    fourier_kernel = torch.exp(-1j * phase)

    times = np.arange(args.steps + 1, dtype=np.float64) * args.dt
    mode_trace = np.empty(args.steps + 1, dtype=np.complex128)
    energy = np.empty(args.steps + 1, dtype=np.float64)
    snapshots = []
    snapshot_steps = []
    for step, t in enumerate(times):
        transverse = spins[0, 0, ..., 0] + 1j * spins[0, 0, ..., 1]
        mode_trace[step] = complex((transverse * fourier_kernel).mean())
        energy[step] = float(model.energy(spins)[0])
        if step % args.save_every == 0 or step == args.steps:
            snapshots.append(spins[0].to(torch.float32).cpu().numpy())
            snapshot_steps.append(step)
        if step != args.steps:
            spins = projected_rk4_step(spins, float(t), args.dt, dynamics)

    fft_frequency = np.fft.fftfreq(len(times), args.dt)
    fft_power = np.abs(np.fft.fft(mode_trace - mode_trace.mean())) ** 2
    nonzero = fft_frequency != 0
    fft_frequency, fft_power = fft_frequency[nonzero], fft_power[nonzero]

    k = 2 * np.pi * args.mode / (args.size * params.lattice)
    k_tensor = torch.tensor(k, dtype=torch.float64)
    omega_minus, omega_plus = analytic_spinwave_omega(
        params, k_tensor, k_tensor, anisotropy_mev=0.0
    )
    expected = np.sort(np.array([float(omega_minus), float(omega_plus)]) /
                       (2*np.pi))
    absolute_frequency = np.abs(fft_frequency)
    measured = np.sort(np.array([
        peak_near(absolute_frequency, fft_power, expected[0]),
        peak_near(absolute_frequency, fft_power, expected[1]),
    ]))
    relative_error = np.abs(measured - expected) / expected
    norm_error = float(torch.max(torch.abs(torch.linalg.vector_norm(spins, dim=-1)-1)))
    energy_drift = float(np.max(np.abs(energy-energy[0])) / abs(energy[0]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "spinwave_trajectory.npz",
        time_s=times,
        mode_real=mode_trace.real,
        mode_imag=mode_trace.imag,
        energy_j=energy,
        snapshots=np.asarray(snapshots),
        snapshot_steps=np.asarray(snapshot_steps, dtype=np.int64),
        fft_frequency_hz=fft_frequency,
        fft_power=fft_power,
        expected_frequency_hz=expected,
        measured_frequency_hz=measured,
    )
    metrics = {
        "reference": "Gomonay et al., npj Spintronics 2, 35 (2024), Supplementary Eq. S.1/S.9",
        "purpose": "small-amplitude deterministic LLG spin-wave validation",
        "parameters": {
            "J1_meV": params.j1_mev,
            "J2_meV": params.j2_mev,
            "J_tilde_meV": params.j_tilde_mev,
            "K_meV": 0.0,
            "mu_s_muB": params.mu_s_mu_b,
            "a0_nm": params.lattice_nm,
        },
        "lattice": [args.size, args.size],
        "mode": [args.mode, args.mode],
        "dt_s": args.dt,
        "steps": args.steps,
        "expected_frequency_THz": (expected / 1e12).tolist(),
        "measured_frequency_THz": (measured / 1e12).tolist(),
        "relative_error": relative_error.tolist(),
        "max_norm_error": norm_error,
        "relative_energy_drift": energy_drift,
        "status": "pass" if (relative_error < 0.05).all() and norm_error < 1e-10
                  and energy_drift < 1e-8 else "fail",
    }
    (args.output_dir / "spinwave_validation.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
