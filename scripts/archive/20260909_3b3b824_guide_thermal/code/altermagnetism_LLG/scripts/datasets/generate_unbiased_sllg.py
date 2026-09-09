"""Generate an equal-weight stochastic-LLG path ensemble without post-selection."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
    SOTPulse,
    antiferromagnetic_state,
    order_parameters,
    stochastic_heun_step,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unbiased finite-temperature sLLG paths")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "data" / "unbiased")
    p.add_argument("--temperature-k", type=float, required=True)
    p.add_argument("--num-trajectories", type=int, default=16)
    p.add_argument("--size", type=int, nargs=2, default=(16, 16),
                   metavar=("NX", "NY"))
    p.add_argument("--equilibration-steps", type=int, default=10000)
    p.add_argument("--steps", type=int, default=25000)
    p.add_argument("--dt", type=float, default=2e-17)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--sot-dl-t", type=float, default=0.0,
                   help="Damping-like SOT field; zero makes an equilibrium control")
    p.add_argument("--pulse-ps", type=float, default=0.0)
    p.add_argument("--polarization", type=float, nargs=3,
                   default=(2**-0.5, 2**-0.5, 0.0))
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--device", default="cpu")
    return p


def summarize(spins: torch.Tensor, model: Ruo2DoubleLayerHamiltonian):
    neel, magnetization = order_parameters(spins)
    dims = (-3, -2)
    return (
        neel.mean(dim=dims)[0].cpu().numpy(),
        magnetization.mean(dim=dims)[0].cpu().numpy(),
        float(model.energy(spins)[0]),
    )


@torch.no_grad()
def main() -> None:
    args = parser().parse_args()
    if args.temperature_k < 0 or args.num_trajectories < 1:
        raise ValueError("temperature must be nonnegative and trajectory count positive")
    if min(*args.size, args.save_every) < 1 or min(args.steps, args.equilibration_steps) < 0:
        raise ValueError("invalid lattice or step settings")
    if args.dt <= 0 or args.alpha <= 0 or args.pulse_ps < 0:
        raise ValueError("require dt > 0, alpha > 0 and pulse >= 0")

    params = LiteratureParameters()
    model = Ruo2DoubleLayerHamiltonian(params)
    prepare = LLGDynamics(model, alpha=args.alpha)
    pulse_end = args.pulse_ps * 1e-12
    driven = LLGDynamics(
        model,
        alpha=args.alpha,
        sot=SOTPulse(
            damping_like_t=args.sot_dl_t,
            polarization=tuple(args.polarization),
            start_s=0.0,
            end_s=pulse_end if pulse_end > 0 else float("inf"),
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for trajectory_id in range(args.num_trajectories):
        seed = args.seed + trajectory_id
        generator_device = args.device if str(args.device).startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        spins = antiferromagnetic_state(1, *args.size, device=args.device)

        # This is an explicit finite-time preparation protocol, not hidden
        # post-selection. Its equilibration length must be certified separately.
        for step in range(args.equilibration_steps):
            spins = stochastic_heun_step(
                spins, step*args.dt, args.dt, prepare, args.temperature_k, generator
            )

        snapshots, snapshot_steps = [], []
        time_s, neel_mean, magnetization_mean, energy_j = [], [], [], []
        for step in range(args.steps + 1):
            t = step * args.dt
            if step % args.save_every == 0 or step == args.steps:
                nbar, mbar, energy = summarize(spins, model)
                snapshots.append(spins[0].to(torch.float32).cpu().numpy())
                snapshot_steps.append(step)
                time_s.append(t)
                neel_mean.append(nbar)
                magnetization_mean.append(mbar)
                energy_j.append(energy)
            if step != args.steps:
                spins = stochastic_heun_step(
                    spins, t, args.dt, driven, args.temperature_k, generator
                )

        path = args.output_dir / f"trajectory_{trajectory_id:06d}.npz"
        np.savez_compressed(
            path,
            snapshots=np.asarray(snapshots),
            snapshot_steps=np.asarray(snapshot_steps, dtype=np.int64),
            time_s=np.asarray(time_s),
            neel_mean=np.asarray(neel_mean),
            magnetization_mean=np.asarray(magnetization_mean),
            energy_j=np.asarray(energy_j),
        )
        entries.append({
            "trajectory_id": trajectory_id,
            "file": path.name,
            "seed": seed,
            "statistical_weight": 1.0 / args.num_trajectories,
            "selected_by_outcome": False,
        })

    metadata = {
        "schema_version": 1,
        "ensemble": "unbiased_equal_weight_given_finite_time_preparation",
        "hamiltonian": "Gomonay2024_supplementary_equation_S1",
        "units": "energy=J, field=T, time=s, temperature=K",
        "parameters": asdict(params),
        "temperature_K": args.temperature_k,
        "size": list(args.size),
        "alpha": args.alpha,
        "dt_s": args.dt,
        "equilibration_steps": args.equilibration_steps,
        "drive_steps": args.steps,
        "save_every": args.save_every,
        "boundary": "periodic_xy",
        "sot": {
            "damping_like_T": args.sot_dl_t,
            "pulse_ps": args.pulse_ps,
            "polarization": list(args.polarization),
        },
        "noise": {
            "variable": "B_th in tesla",
            "mean": "zero",
            "covariance": (
                "<B_mu,i(t) B_nu,j(t')> = "
                "2 alpha k_B T/(gamma mu_s) delta_mu,nu delta_i,j delta(t-t')"
            ),
            "equivalent_energy_variable": (
                "R=mu_s B_th has prefactor 2 alpha (mu_s/gamma) k_B T"
            ),
        },
        "integrator": "projected stochastic Heun, Stratonovich, same noise in predictor/corrector",
        "split_unit": "whole_trajectory",
        "production_warning": "certify equilibration, timestep convergence and ordered phase before scientific use",
        "trajectories": entries,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir.resolve()),
                      "trajectories": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
