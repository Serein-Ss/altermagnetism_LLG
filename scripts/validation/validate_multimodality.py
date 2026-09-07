"""Test whether one finite-temperature protocol produces multiple path classes.

The protocol is fixed before looking at the stochastic ensemble: a deterministic
scan identifies a near-separatrix SOT field, then every finite-temperature path
is retained with equal statistical weight.  The result is a validation dataset,
not a production switching benchmark.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
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
    p = argparse.ArgumentParser(description="Pre-registered path multimodality test")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "data" / "multimodality_validation")
    p.add_argument("--temperature-k", type=float, default=5.0)
    p.add_argument("--sot-dl-t", type=float, default=0.8)
    p.add_argument("--pulse-ps", type=float, default=0.5)
    p.add_argument("--total-ps", type=float, default=1.0)
    p.add_argument("--preparation-ps", type=float, default=0.2)
    p.add_argument("--dt-fs", type=float, default=0.1)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--num-trajectories", type=int, default=256)
    p.add_argument("--size", type=int, nargs=2, default=(8, 8))
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--j-tilde-mev", type=float, default=0.8,
                   help="Set to zero for the conventional AFM control")
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--device", default="cuda")
    return p


def mean_neel(spins: torch.Tensor) -> torch.Tensor:
    neel, _ = order_parameters(spins)
    return neel.mean(dim=(-3, -2))


@torch.no_grad()
def main() -> None:
    args = parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not (0 < args.pulse_ps < args.total_ps):
        raise ValueError("require 0 < pulse-ps < total-ps")
    if min(args.temperature_k, args.preparation_ps) < 0 or args.dt_fs <= 0:
        raise ValueError("temperature/preparation must be nonnegative and dt positive")
    if args.num_trajectories < 20 or min(args.size) < 2:
        raise ValueError("use at least 20 trajectories and a 2 x 2 lattice")

    dt = args.dt_fs * 1e-15
    preparation_steps = round(args.preparation_ps * 1e-12 / dt)
    steps = round(args.total_ps * 1e-12 / dt)
    params = LiteratureParameters(j_tilde_mev=args.j_tilde_mev)
    model = Ruo2DoubleLayerHamiltonian(params)
    prepare = LLGDynamics(model, alpha=args.alpha)
    driven = LLGDynamics(
        model,
        alpha=args.alpha,
        sot=SOTPulse(
            damping_like_t=args.sot_dl_t,
            polarization=(2**-0.5, 2**-0.5, 0.0),
            start_s=0.0,
            end_s=args.pulse_ps * 1e-12,
        ),
    )

    spins = antiferromagnetic_state(
        args.num_trajectories, *args.size, device=args.device
    )
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    for step in range(preparation_steps):
        spins = stochastic_heun_step(
            spins, step*dt, dt, prepare, args.temperature_k, generator
        )

    saved_steps, paths = [], []
    for step in range(steps + 1):
        if step % args.save_every == 0 or step == steps:
            saved_steps.append(step)
            paths.append(mean_neel(spins).cpu().numpy())
        if step != steps:
            spins = stochastic_heun_step(
                spins, step*dt, dt, driven, args.temperature_k, generator
            )

    paths = np.stack(paths, axis=1)
    time_ps = np.asarray(saved_steps) * dt / 1e-12
    nz = paths[..., 2]
    crossed = np.any(nz <= 0.0, axis=1)
    first_index = np.argmax(nz <= 0.0, axis=1)
    first_passage = np.where(crossed, time_ps[first_index], args.total_ps + args.dt_fs/1000)
    crossing_count = np.sum((nz[:, 1:] * nz[:, :-1]) < 0.0, axis=1)
    final_negative = nz[:, -1] < 0.0
    returned = crossed & ~final_negative
    physical_class = np.where(~crossed, 0, np.where(returned, 1, 2))

    features = np.column_stack([
        nz[:, -1],
        nz.min(axis=1),
        np.linalg.norm(paths[..., :2], axis=-1).max(axis=1),
        first_passage,
        crossing_count,
        np.mean(nz < 0.0, axis=1),
    ])
    # Unsupervised evidence is calculated from the continuous n_z(t) paths,
    # not from the hand-labelled crossing classes or discrete crossing counts.
    standardized_paths = StandardScaler().fit_transform(nz)
    pca = PCA(n_components=min(6, standardized_paths.shape[1]),
              random_state=args.seed)
    standardized = pca.fit_transform(standardized_paths)
    models = [GaussianMixture(k, covariance_type="full", n_init=10,
                              random_state=args.seed).fit(standardized)
              for k in range(1, 5)]
    bic = np.asarray([m.bic(standardized) for m in models])
    best_index = int(np.argmin(bic))
    cluster = models[best_index].predict(standardized)

    counts = np.bincount(physical_class, minlength=3)
    crossing_fraction = float(crossed.mean())
    bic_gain = float(bic[0] - bic[best_index])
    pass_outcome_split = 0.1 <= crossing_fraction <= 0.9
    pass_gmm = best_index + 1 >= 2 and bic_gain >= 10.0
    result = {
        "question": "Are equal-weight finite-temperature paths multimodal at one fixed condition?",
        "protocol_registered_from": "independent zero-temperature SOT scan",
        "parameters": asdict(params),
        "condition": {
            "temperature_K": args.temperature_k,
            "sot_damping_like_T": args.sot_dl_t,
            "pulse_ps": args.pulse_ps,
            "total_ps": args.total_ps,
            "preparation_ps": args.preparation_ps,
            "dt_fs": args.dt_fs,
            "alpha": args.alpha,
            "lattice": list(args.size),
            "periodic_boundary": True,
            "trajectories": args.num_trajectories,
            "seed": args.seed,
        },
        "physical_path_classes": {
            "no_zero_crossing": int(counts[0]),
            "cross_and_return_positive": int(counts[1]),
            "negative_at_final_time": int(counts[2]),
        },
        "crossing_fraction": crossing_fraction,
        "gmm_bic_components_1_to_4": bic.tolist(),
        "best_gmm_components": best_index + 1,
        "bic_gain_over_one_component": bic_gain,
        "path_PCA_explained_variance": pca.explained_variance_ratio_.tolist(),
        "pre_registered_pass_rules": {
            "both_crossing_and_non_crossing_at_least_10_percent": pass_outcome_split,
            "best_GMM_has_at_least_2_components_and_DeltaBIC_at_least_10": pass_gmm,
        },
        "status": "pass" if pass_outcome_split and pass_gmm else "not_demonstrated",
        "scope_warning": (
            "This establishes finite-size path bifurcation for the stated protocol only; "
            "larger lattices and timestep/seed replication are required before a physics claim."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "multimodality_ensemble.npz",
        time_ps=time_ps,
        neel_mean=paths.astype(np.float32),
        features=features.astype(np.float32),
        physical_class=physical_class.astype(np.int8),
        gmm_cluster=cluster.astype(np.int8),
        path_pca=standardized.astype(np.float32),
        statistical_weight=np.full(args.num_trajectories,
                                   1.0/args.num_trajectories),
    )
    (args.output_dir / "multimodality_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
