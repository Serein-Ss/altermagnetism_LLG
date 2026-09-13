"""Aggregate full-size evaluation metrics across independent training seeds."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


FRACTIONS = (
    "crossed_zero_fraction",
    "negative_endpoint_fraction",
    "committed_switch_fraction",
    "crossing_return_fraction",
    "unresolved_transition_fraction",
)
ERRORS = (
    "endpoint_wasserstein_equal_count",
    "mean_neel_z_path_rmse",
    "mean_spatial_std_path_rmse",
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=(16, 32, 64, 96))
    p.add_argument("--output", type=Path, required=True)
    return p


def mean_std(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "count": int(len(array)),
        "values": array.tolist(),
    }


def main() -> None:
    args = parser().parse_args()
    values = defaultdict(lambda: defaultdict(list))
    references = {}
    statuses = {}
    evaluation_seeds = {}

    for seed in args.seeds:
        for size in args.sizes:
            path = args.root / f"seed_{seed}" / "evaluation" / f"L{size}" / "evaluation.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            statuses[f"seed_{seed}/L{size}"] = report["status"]
            evaluation_seeds[f"seed_{seed}/L{size}"] = report.get(
                "generation_seed_base"
            )
            for condition in report["conditions"]:
                field = float(condition["drive_T"])
                key = (size, field)
                references.setdefault(
                    key,
                    {
                        metric: condition["reference"][metric]
                        for metric in FRACTIONS
                    },
                )
                for model in ("flow", "deterministic"):
                    for metric in FRACTIONS:
                        values[(key, model)][metric].append(
                            condition[model][metric]
                        )
                    error_key = f"{model}_errors"
                    for metric in ERRORS:
                        values[(key, model)][metric].append(
                            condition[error_key][metric]
                        )
                    runtime_key = (
                        "flow_per_path"
                        if model == "flow"
                        else "deterministic_per_path"
                    )
                    runtime = condition.get(
                        "generation_runtime_seconds", {}
                    ).get(runtime_key)
                    if runtime is not None:
                        values[(key, model)]["seconds_per_path"].append(
                            runtime
                        )

    aggregate = {}
    markdown = [
        "# Three-seed full-size evaluation",
        "",
        "| L | field (T) | model | committed switch | unresolved | endpoint W1 | path RMSE | spatial RMSE | seconds/path |",
        "|---:|---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for (key, model), metrics in sorted(values.items()):
        size, field = key
        summary = {
            metric: mean_std(samples)
            for metric, samples in metrics.items()
        }
        name = f"L{size}/H{field:.2f}/{model}"
        aggregate[name] = summary
        format_value = lambda metric: (
            f"{summary[metric]['mean']:.3f} ± {summary[metric]['std']:.3f}"
        )
        runtime = (
            format_value("seconds_per_path")
            if "seconds_per_path" in summary else "n/a"
        )
        markdown.append(
            f"| {size} | {field:.2f} | {model} | "
            f"{format_value('committed_switch_fraction')} | "
            f"{format_value('unresolved_transition_fraction')} | "
            f"{format_value('endpoint_wasserstein_equal_count')} | "
            f"{format_value('mean_neel_z_path_rmse')} | "
            f"{format_value('mean_spatial_std_path_rmse')} | "
            f"{runtime} |"
        )

    report = {
        "status": "three_seed_evaluation_aggregated",
        "seeds": args.seeds,
        "sizes": args.sizes,
        "evaluation_statuses": statuses,
        "evaluation_seed_bases": evaluation_seeds,
        "reference_fractions": {
            f"L{size}/H{field:.2f}": summary
            for (size, field), summary in sorted(references.items())
        },
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
