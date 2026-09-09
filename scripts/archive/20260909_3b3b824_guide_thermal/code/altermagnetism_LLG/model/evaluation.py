"""Data-driven basin calibration and path-distribution evaluation helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from .sphere import sample_reference_path, sphere_exp


@dataclass(frozen=True)
class BasinCalibration:
    negative_threshold: float
    positive_threshold: float
    residence_ps: float
    component_means: tuple[float, ...]
    component_std: tuple[float, ...]
    component_weights: tuple[float, ...]
    trajectories: int
    sources: tuple[str, ...]
    component_count: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


def neel_observables(spins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    local = 0.5 * (spins[:, 0, ..., 2] - spins[:, 1, ..., 2])
    return local.mean(axis=(1, 2)), local.std(axis=(1, 2))


def _tail_start(time: np.ndarray, residence_ps: float) -> int:
    cutoff = float(time[-1]) - residence_ps * 1e-12
    return int(np.searchsorted(time, cutoff, side="left"))


def _posterior_boundary(
    model: GaussianMixture, left_component: int, right_component: int,
    left_mean: float, right_mean: float,
) -> float:
    grid = np.linspace(left_mean, right_mean, 20_001, dtype=np.float64)
    posterior = model.predict_proba(grid[:, None])
    difference = np.abs(
        posterior[:, left_component] - posterior[:, right_component]
    )
    return float(grid[int(np.argmin(difference))])


def calibrate_basins(
    paths: list[str | Path],
    system: str,
    *,
    residence_ps: float = 0.10,
    seed: int = 20260908,
) -> BasinCalibration:
    """Fit negative/transition/positive endpoint modes from real LLG paths.

    Only train and validation trajectories (split 0/1) are used.  Each fit value
    is the mean global Neel-z over the requested terminal residence window.
    """
    values = []
    sources = []
    for path in paths:
        resolved = Path(path).resolve()
        with h5py.File(resolved, "r", swmr=True) as h5:
            group = h5[system]
            split = group["split"][:]
            indices = np.flatnonzero(split != 2)
            if not len(indices):
                continue
            time = h5["time"][:]
            start = _tail_start(time, residence_ps)
            values.append(group["neel"][indices, start:, 2].mean(axis=1))
            sources.append(str(resolved))
    if not values:
        raise ValueError("basin calibration requires real train/validation paths")
    terminal = np.concatenate(values).astype(np.float64)
    if len(terminal) < 30:
        raise ValueError("basin calibration requires at least 30 trajectories")
    three_component = GaussianMixture(
        n_components=3,
        covariance_type="full",
        n_init=20,
        random_state=seed,
    ).fit(terminal[:, None])
    raw_means = three_component.means_[:, 0]
    order = np.argsort(raw_means)
    means = raw_means[order]
    has_transition_component = bool(
        means[0] < -0.5
        and abs(means[1]) < 0.5
        and means[2] > 0.5
    )

    if has_transition_component:
        mixture = three_component
        negative = _posterior_boundary(
            mixture, int(order[0]), int(order[1]), means[0], means[1]
        )
        positive = _posterior_boundary(
            mixture, int(order[1]), int(order[2]), means[1], means[2]
        )
    else:
        mixture = GaussianMixture(
            n_components=2,
            covariance_type="full",
            n_init=20,
            random_state=seed,
        ).fit(terminal[:, None])
        raw_means = mixture.means_[:, 0]
        order = np.argsort(raw_means)
        means = raw_means[order]
        if not (means[0] < -0.5 and means[1] > 0.5):
            raise ValueError(
                "real LLG terminal distribution has not resolved both stable basins; "
                "extend the observation window"
            )
        std = np.sqrt(mixture.covariances_[:, 0, 0])[order]
        negative = float(means[0] + 3.0 * std[0])
        positive = float(means[1] - 3.0 * std[1])

    if not (means[0] < 0.0 < means[-1]):
        raise ValueError(
            "real LLG terminal distribution does not resolve both magnetic basins"
        )
    if not negative < positive:
        raise ValueError(
            "fitted stable basins overlap; extend the observation window"
        )
    std = np.sqrt(mixture.covariances_[:, 0, 0])[order]
    weights = mixture.weights_[order]
    return BasinCalibration(
        negative_threshold=negative,
        positive_threshold=positive,
        residence_ps=residence_ps,
        component_means=tuple(float(value) for value in means),
        component_std=tuple(float(value) for value in std),
        component_weights=tuple(float(value) for value in weights),
        trajectories=int(len(terminal)),
        sources=tuple(sources),
        component_count=int(len(means)),
    )


def path_descriptors(
    spins: np.ndarray,
    time: np.ndarray,
    calibration: BasinCalibration,
) -> dict:
    order, spatial_std = neel_observables(spins)
    crossed_indices = np.flatnonzero(
        np.signbit(order[1:]) != np.signbit(order[:-1])
    ) + 1
    negative_endpoint = bool(order[-1] < 0.0)
    start = _tail_start(time, calibration.residence_ps)
    terminal_negative = bool(
        np.all(order[start:] <= calibration.negative_threshold)
    )
    terminal_positive = bool(
        np.all(order[start:] >= calibration.positive_threshold)
    )
    initial_positive = bool(
        order[0] >= calibration.positive_threshold
    )
    initial_negative = bool(
        order[0] <= calibration.negative_threshold
    )
    crossed_zero = bool(len(crossed_indices))
    if initial_positive:
        terminal_reverse = terminal_negative
        terminal_original = terminal_positive
        initial_basin = "positive"
    elif initial_negative:
        terminal_reverse = terminal_positive
        terminal_original = terminal_negative
        initial_basin = "negative"
    else:
        terminal_reverse = terminal_original = False
        initial_basin = "transition"
    committed_switch = crossed_zero and terminal_reverse
    crossing_return = crossed_zero and terminal_original
    unresolved_transition = bool(
        not committed_switch
        and not crossing_return
        and (crossed_zero or not terminal_original)
    )
    outcome = "no_crossing"
    if committed_switch:
        outcome = "committed_switch"
    elif crossing_return:
        outcome = "crossing_return"
    elif unresolved_transition:
        outcome = "unresolved_transition"
    transition_std = float(
        np.where(np.abs(order) < 0.5, spatial_std, 0.0).max()
    )
    return {
        "order": order,
        "spatial_std": spatial_std,
        "endpoint": float(order[-1]),
        "crossed_zero": crossed_zero,
        "negative_endpoint": negative_endpoint,
        "committed_switch": committed_switch,
        "crossing_return": crossing_return,
        "unresolved_transition": unresolved_transition,
        "outcome": outcome,
        "initial_basin": initial_basin,
        "nonuniform_committed_switch": bool(
            committed_switch and transition_std >= 0.25
        ),
        "first_zero_crossing_ps": (
            float(time[crossed_indices[0]] * 1e12)
            if len(crossed_indices)
            else None
        ),
        "peak_spatial_std": float(spatial_std.max()),
        "max_spin_norm_error": float(
            np.abs(np.linalg.norm(spins, axis=-1) - 1.0).max()
        ),
    }


def summarize(rows: list[dict]) -> dict:
    orders = np.stack([row["order"] for row in rows])
    spatial_std = np.stack([row["spatial_std"] for row in rows])
    endpoints = np.asarray([row["endpoint"] for row in rows])
    passages = [
        row["first_zero_crossing_ps"]
        for row in rows
        if row["first_zero_crossing_ps"] is not None
    ]
    fractions = {
        f"{name}_fraction": float(np.mean([row[name] for row in rows]))
        for name in (
            "crossed_zero",
            "negative_endpoint",
            "committed_switch",
            "crossing_return",
            "unresolved_transition",
            "nonuniform_committed_switch",
        )
    }
    return {
        "paths": len(rows),
        **fractions,
        "no_crossing_fraction": float(
            np.mean([row["outcome"] == "no_crossing" for row in rows])
        ),
        "endpoint_mean": float(endpoints.mean()),
        "endpoint_std": float(endpoints.std()),
        "median_first_zero_crossing_ps": (
            float(np.median(passages)) if passages else None
        ),
        "mean_peak_spatial_std": float(
            np.mean([row["peak_spatial_std"] for row in rows])
        ),
        "neel_z_paths": orders.tolist(),
        "spatial_std_paths": spatial_std.tolist(),
        "max_spin_norm_error": float(
            max(row["max_spin_norm_error"] for row in rows)
        ),
        "mean_neel_z_path": orders.mean(axis=0).tolist(),
        "mean_spatial_std_path": spatial_std.mean(axis=0).tolist(),
        "endpoints": endpoints.tolist(),
        "outcomes": [row["outcome"] for row in rows],
    }


def compare(reference: dict, generated: dict) -> dict:
    reference_endpoints = np.sort(np.asarray(reference["endpoints"]))
    generated_endpoints = np.sort(np.asarray(generated["endpoints"]))
    result = {
        "endpoint_wasserstein_equal_count": float(
            np.abs(reference_endpoints - generated_endpoints).mean()
        ),
        "mean_neel_z_path_rmse": float(
            np.sqrt(np.mean((
                np.asarray(reference["mean_neel_z_path"])
                - np.asarray(generated["mean_neel_z_path"])
            ) ** 2))
        ),
        "mean_spatial_std_path_rmse": float(
            np.sqrt(np.mean((
                np.asarray(reference["mean_spatial_std_path"])
                - np.asarray(generated["mean_spatial_std_path"])
            ) ** 2))
        ),
    }
    for name in (
        "negative_endpoint",
        "committed_switch",
        "crossing_return",
        "unresolved_transition",
    ):
        key = f"{name}_fraction"
        result[f"{key}_absolute_error"] = abs(
            generated[key] - reference[key]
        )
    return result


@torch.no_grad()
def generate_flow(
    model,
    initial: torch.Tensor,
    scalar: torch.Tensor,
    vector: torch.Tensor,
    physical_time: torch.Tensor,
    integration_steps: int,
    seed: int,
) -> torch.Tensor:
    generator_device = (
        str(initial.device) if initial.device.type == "cuda" else "cpu"
    )
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    state = sample_reference_path(
        initial, physical_time.shape[1], generator=generator
    )
    step = 1.0 / integration_steps
    for index in range(integration_steps):
        tau = torch.full(
            (initial.shape[0],),
            index * step,
            device=initial.device,
            dtype=initial.dtype,
        )
        velocity = model(
            state, tau, initial, scalar, vector, physical_time
        )
        state = sphere_exp(state, step * velocity)
        state[:, 0] = initial
    return state
