"""Extract outcome and spatial-mechanism descriptors from full LLG paths."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Spatial LLG path mechanism analysis")
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, default=None)
    return p


def outcome_labels(order: np.ndarray) -> np.ndarray:
    crossed = order.min(axis=1) < 0.0
    negative_endpoint = order[:, -1] < 0.0
    labels = np.full(len(order), "no_crossing", dtype="U32")
    labels[crossed & ~negative_endpoint] = "crossing_return"
    labels[negative_endpoint] = "negative_endpoint"
    return labels


def first_passage_times(order: np.ndarray, time: np.ndarray) -> np.ndarray:
    result = np.full(len(order), np.nan, dtype=np.float64)
    for i, path in enumerate(order):
        indices = np.flatnonzero(path < 0.0)
        if len(indices):
            result[i] = time[indices[0]]
    return result


def first_chain_nucleation(local: np.ndarray, minimum_run: int = 3) -> list[str]:
    locations = []
    for path in local:
        location = "not_resolved"
        for frame in path:
            reversed_sites = frame < 0.0
            left = 0
            while left < len(frame) and reversed_sites[left]:
                left += 1
            right = 0
            while right < len(frame) and reversed_sites[-1 - right]:
                right += 1
            longest = current = 0
            for value in reversed_sites:
                current = current + 1 if value else 0
                longest = max(longest, current)
            if longest >= minimum_run:
                location = "edge" if max(left, right) >= minimum_run else "bulk"
                break
        locations.append(location)
    return locations


def summarize_bauer(h5: h5py.File) -> tuple[dict, dict[str, np.ndarray]]:
    spins = h5["spins"][:]
    time = h5["time"][:]
    local = spins[:, :, 0, :, 0, 2]
    order = local.mean(axis=2)
    labels = outcome_labels(order)
    walls = (np.signbit(local[:, :, 1:]) != np.signbit(local[:, :, :-1])).sum(
        axis=2
    )
    locations = np.asarray(first_chain_nucleation(local), dtype="U16")
    summary = {
        "kind": "open_chain",
        "n_trajectories": len(order),
        "outcome_counts": dict(Counter(labels.tolist())),
        "first_resolved_nucleation_counts": dict(Counter(locations.tolist())),
        "mean_max_domain_walls": float(walls.max(axis=1).mean()),
        "interpretation": (
            "Distinct endpoint and return classes are present. Nucleation location in "
            "the accelerated pilot is not expected to match the lower-temperature paper."
        ),
    }
    arrays = {
        "time": time,
        "order": order,
        "outcome": labels,
        "first_passage_time": first_passage_times(order, time),
        "wall_count": walls,
        "nucleation_location": locations,
    }
    return summary, arrays


def periodic_wall_density(local: np.ndarray) -> np.ndarray:
    sign = np.signbit(local)
    x_walls = sign != np.roll(sign, -1, axis=2)
    y_walls = sign != np.roll(sign, -1, axis=3)
    return 0.5 * (x_walls.mean(axis=(2, 3)) + y_walls.mean(axis=(2, 3)))


def summarize_group(
    group: h5py.Group, time: np.ndarray | None = None
) -> tuple[dict, dict[str, np.ndarray]]:
    spins = group["spins"][:]
    if time is None:
        time = group["time"][:]
    if spins.shape[2] == 2:
        local = 0.5 * (spins[:, :, 0, ..., 2] - spins[:, :, 1, ..., 2])
    else:
        local = spins[:, :, 0, ..., 2]
    order = local.mean(axis=(2, 3))
    labels = outcome_labels(order)
    spatial_std = local.std(axis=(2, 3))
    wall_density = periodic_wall_density(local)
    transition = np.abs(order) < 0.5
    transition_std = np.where(transition, spatial_std, 0.0).max(axis=1)
    mechanism = np.full(len(order), "not_switched", dtype="U32")
    switched = labels == "negative_endpoint"
    mechanism[switched & (transition_std < 0.25)] = "coherent_like"
    mechanism[switched & (transition_std >= 0.25)] = "spatially_nonuniform"
    summary = {
        "n_trajectories": len(order),
        "outcome_counts": dict(Counter(labels.tolist())),
        "switched_mechanism_counts": dict(Counter(mechanism[switched].tolist())),
        "mean_max_spatial_std": float(spatial_std.max(axis=1).mean()),
        "mean_max_periodic_wall_density": float(wall_density.max(axis=1).mean()),
        "mechanism_threshold": (
            "coherent_like if max spatial std during |global order|<0.5 is below 0.25"
        ),
    }
    arrays = {
        "time": time,
        "order": order,
        "outcome": labels,
        "first_passage_time": first_passage_times(order, time),
        "spatial_std": spatial_std,
        "wall_density": wall_density,
        "mechanism": mechanism,
    }
    return summary, arrays


def main() -> None:
    args = parser().parse_args()
    output = args.output or args.input.with_name(args.input.stem + "_mechanisms.json")
    array_output = output.with_suffix(".npz")
    report = {
        "input": str(args.input.resolve()),
        "definition_warning": (
            "These are pre-registered geometric descriptors, not ground-truth semantic "
            "labels. Threshold robustness and size convergence are required."
        ),
        "groups": {},
    }
    arrays = {}
    with h5py.File(args.input, "r") as h5:
        if "spins" in h5:
            summary, group_arrays = summarize_bauer(h5)
            report["groups"]["bauer_chain"] = summary
            arrays.update({f"bauer_chain_{k}": v for k, v in group_arrays.items()})
        else:
            for name, group in h5.items():
                if not isinstance(group, h5py.Group) or "spins" not in group:
                    continue
                time = h5["time"][:] if "time" in h5 else None
                summary, group_arrays = summarize_group(group, time)
                report["groups"][name] = summary
                arrays.update({f"{name}_{k}": v for k, v in group_arrays.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(array_output, **arrays)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
