"""One independent Nishino run with coupled dt/dt2/dt4 and two integrators.

Formal settings: 1000 free moments, duration 400, preparation 200, dt=0.005.
Task index enumerates 12 temperatures (0.5,...,6.0) and ten independent runs.
Reduced arguments are software tests and explicitly cannot pass certification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "core"))
from unified_llg import NishinoFreeMomentHamiltonian, UnifiedLLGSolver, collinear_state
from audit_integrators import coupled_fields, heun_with_field, midpoint_with_field


def autocorrelation_time(values):
    x = np.asarray(values, dtype=float)
    x = x - x.mean()
    if len(x) < 4 or np.dot(x, x) == 0:
        return None
    n = len(x)
    f = np.fft.rfft(x, n=2*n)
    ac = np.fft.irfft(f * f.conj())[:n]
    ac = ac / np.arange(n, 0, -1)
    ac /= ac[0]
    # Initial positive pair sequence; report in saved-sample units.
    total = 0.
    for i in range(1, n-1, 2):
        pair = ac[i] + ac[i+1]
        if pair <= 0:
            break
        total += pair
    return float(max(1., 1 + 2*total))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--coarse-steps", type=int, default=80000)
    p.add_argument("--moments", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=40)
    p.add_argument("--wall-budget-seconds", type=float, default=172800.)
    args = p.parse_args()
    if not 0 <= args.task < 120 or args.coarse_steps < 4 or args.moments < 2:
        p.error("invalid task, step count or moment count")
    if args.coarse_steps % (2*args.save_every):
        p.error("coarse steps must be a multiple of twice save-every")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"run_{args.task:03d}.h5"
    if output.exists() or output.with_suffix(".h5.partial").exists():
        raise FileExistsError(output)
    temperature = .5 * (args.task // 10 + 1)
    seed = int.from_bytes(hashlib.sha256(f"guide-nishino:{args.task}".encode()).digest()[:7], "little")
    generator = torch.Generator().manual_seed(seed)
    model = NishinoFreeMomentHamiltonian()
    solver = UnifiedLLGSolver(model, alpha=.05)
    initial = collinear_state(1, 1, args.moments, 1, antiferromagnetic=False,
                              device="cpu", dtype=torch.float64)
    states = {(method, factor): initial.clone()
              for method in ("heun", "midpoint") for factor in (1, 2, 4)}
    traces = {key: [] for key in states}
    norm_errors = {key: 0. for key in states}
    raw_errors = {factor: [] for factor in (1, 2, 4)}
    coarse_dt = .005
    begin = time.perf_counter()
    runtime = None
    for step in range(args.coarse_steps):
        noise = coupled_fields(solver, initial, temperature, coarse_dt, generator)
        fields = {1: noise[0][None], 2: noise[1], 4: noise[2]}
        for (method, factor), state in states.items():
            dt = coarse_dt / factor
            for substep, field in enumerate(fields[factor]):
                t = step*coarse_dt + substep*dt
                if method == "heun":
                    state, errors = heun_with_field(solver, state, t, dt, field)
                    raw_errors[factor].append(errors.numpy())
                else:
                    state = midpoint_with_field(solver, state, t, dt, field)
                norm_errors[method, factor] = max(norm_errors[method, factor],
                    float((state.norm(dim=-1)-1).abs().max()))
            states[method, factor] = state
            if step >= args.coarse_steps//2 and (step+1) % args.save_every == 0:
                traces[method, factor].append(state[..., 2].numpy().ravel().copy())
        if step == 9999:
            elapsed = time.perf_counter() - begin
            estimate = elapsed * args.coarse_steps / 10000
            runtime = {"seconds_per_10000_coarse_steps": elapsed,
                       "estimated_compute_seconds": estimate,
                       "required_walltime_with_margin": 1.5*estimate}
            (args.output_dir / f"runtime_{args.task:03d}.json").write_text(
                json.dumps(runtime, indent=2), encoding="utf-8")
            if 1.5*estimate > args.wall_budget_seconds:
                raise RuntimeError("measured runtime exceeds allocation; no automatic continuation")
        if (step+1) % 10000 == 0:
            print(f"task={args.task} step={step+1}/{args.coarse_steps}", flush=True)
    reports = []
    partial = output.with_suffix(".h5.partial")
    with h5py.File(partial, "w") as h5:
        h5.attrs["temperature"] = temperature
        h5.attrs["seed"] = seed
        h5.attrs["model_metadata"] = json.dumps(model.metadata())
        h5.attrs["alpha"] = .05
        h5.attrs["coarse_dt"] = coarse_dt
        h5.attrs["duration"] = args.coarse_steps*coarse_dt
        h5.attrs["burn_in"] = args.coarse_steps*coarse_dt/2
        h5.attrs["save_interval"] = args.save_every*coarse_dt
        for (method, factor), trace in traces.items():
            z = np.asarray(trace)
            group = h5.create_group(f"{method}/dt_{factor}")
            group.create_dataset("cos_theta", data=z, compression="gzip", shuffle=True)
            per_moment_mean = z.mean(axis=0)
            # Distinct noninteracting spins have independent noise streams. Their
            # time averages, not individual saved frames, are independent units.
            se = per_moment_mean.std(ddof=1) / np.sqrt(args.moments)
            exact = model.exact_magnetization_z(temperature)
            x = model.parameters.field_h / temperature
            exact_second = 1 - 2*exact/x
            second = (z*z).mean(axis=0)
            second_se = second.std(ddof=1)/np.sqrt(args.moments)
            tau = autocorrelation_time(z.mean(axis=1))
            checks = {
                "mean": abs(float(z.mean())-exact) <= max(.01, 3*se),
                "second_moment": abs(float(second.mean())-exact_second) <= max(.01*abs(exact_second), 3*second_se),
                "spin_norm": norm_errors[method, factor] <= 1e-10,
                "finite": bool(np.isfinite(z).all()),
            }
            checks = {key: bool(value) for key, value in checks.items()}
            reports.append({"method": method, "factor": factor, "dt": coarse_dt/factor,
                "mean": float(z.mean()), "mean_se": float(se), "exact_mean": exact,
                "second_moment": float(second.mean()), "second_se": float(second_se),
                "exact_second": exact_second, "tau_saved_samples": tau,
                "effective_time_samples": None if tau is None else len(z)/tau,
                "norm_error": norm_errors[method, factor], "checks": checks})
        for factor, values in raw_errors.items():
            h5.create_dataset(f"heun_pre_normalization_max_errors/dt_{factor}",
                              data=np.asarray(values), compression="gzip", shuffle=True)
        h5.attrs["complete"] = True
    partial.replace(output)
    formal = args.coarse_steps == 80000 and args.moments == 1000 and args.save_every == 40
    report = {"task": args.task, "temperature": temperature, "seed": seed,
              "formal_settings": formal, "status": "raw_reference_run_awaiting_ensemble_audit",
              "elapsed_seconds": time.perf_counter()-begin, "runtime": runtime,
              "rows": reports, "basic_checks_passed": all(all(r["checks"].values()) for r in reports),
              "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Per-run fluctuations are retained. Only the preregistered ensemble audit
    # decides scientific acceptance; the generator does not self-certify.


if __name__ == "__main__":
    main()
