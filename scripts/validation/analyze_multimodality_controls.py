"""Compare path-class counts across timestep replication and AFM control."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency


ROOT = Path(__file__).resolve().parents[2]
FOLDERS = {
    "altermagnet_dt_0.10_fs": ROOT / "data" / "multimodality_validation",
    "altermagnet_dt_0.05_fs": ROOT / "data" / "multimodality_validation_dt0p05",
    "afm_control_jtilde_0": ROOT / "data" / "multimodality_afm_control",
}
KEYS = ["no_zero_crossing", "cross_and_return_positive", "negative_at_final_time"]


def wilson(success: int, total: int, z: float = 1.959963984540054):
    p = success/total
    d = 1 + z*z/total
    center = (p + z*z/(2*total))/d
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/d
    return [center-half, center+half]


def main() -> None:
    metrics = {
        name: json.loads((folder / "multimodality_metrics.json").read_text())
        for name, folder in FOLDERS.items()
    }
    rows = {}
    for name, result in metrics.items():
        counts = [result["physical_path_classes"][key] for key in KEYS]
        total = sum(counts)
        crossing = counts[1] + counts[2]
        rows[name] = {
            "counts": counts,
            "fractions": (np.asarray(counts)/total).tolist(),
            "crossing_fraction": crossing/total,
            "crossing_fraction_wilson_95pct": wilson(crossing, total),
            "bic_gain_over_one_component": result["bic_gain_over_one_component"],
        }

    base = np.asarray(rows["altermagnet_dt_0.10_fs"]["counts"])
    dt = np.asarray(rows["altermagnet_dt_0.05_fs"]["counts"])
    afm = np.asarray(rows["afm_control_jtilde_0"]["counts"])
    dt_test = chi2_contingency(np.vstack([base, dt]))
    afm_test = chi2_contingency(np.vstack([base, afm]))
    output = {
        "class_order": KEYS,
        "ensembles": rows,
        "chi_square_class_fraction_tests": {
            "dt_0.10_vs_0.05_fs": {
                "chi2": float(dt_test.statistic), "dof": int(dt_test.dof),
                "p_value": float(dt_test.pvalue),
            },
            "altermagnet_vs_afm_control_at_this_small_uniform_protocol": {
                "chi2": float(afm_test.statistic), "dof": int(afm_test.dof),
                "p_value": float(afm_test.pvalue),
            },
        },
        "interpretation": {
            "multimodality_exists": True,
            "stable_to_halving_timestep": True,
            "altermagnet_specificity_demonstrated": False,
            "reason": (
                "The small near-uniform test shows indistinguishable class fractions "
                "when j_tilde is set to zero. Spatial textures and crystal direction "
                "must be tested to establish an altermagnetic effect."
            ),
        },
    }
    target = ROOT / "data" / "multimodality_control_comparison.json"
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
