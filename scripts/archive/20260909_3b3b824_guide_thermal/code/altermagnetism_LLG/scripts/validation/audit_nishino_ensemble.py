"""Separate ensemble gate for 120 complete preregistered Nishino runs.

Independent runs are bootstrap units; saved frames are never independent units.
Passing this gate does not certify an interacting-material production dataset.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import t as student_t


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(root):
    runs = []
    for task in range(120):
        path = root / f"run_{task:03d}.json"
        run = json.loads(path.read_text())
        if run["task"] != task or not run["formal_settings"] or run["sha256"] != file_hash(path.with_suffix(".h5")):
            raise ValueError(f"run {task}: missing formal settings or hash mismatch")
        runs.append(run)
    if len({r["seed"] for r in runs}) != 120:
        raise ValueError("duplicate seeds")
    rng = np.random.default_rng(20260910)
    grid = np.linspace(-1, 1, 65)
    indices = rng.integers(0, 10, size=(20000, 10))
    summaries, comparisons = [], []
    multiplier = float(student_t.ppf(.975, 9))
    for ti in range(12):
        subset = runs[ti*10:(ti+1)*10]
        temperature = .5*(ti+1)
        x = 2/temperature
        exact_cdf = np.expm1(x*(grid+1))/np.expm1(2*x)
        cache = {}
        for method in ("heun", "midpoint"):
            for factor in (1, 2, 4):
                rows = [next(r for r in run["rows"] if (r["method"],r["factor"]) == (method,factor)) for run in subset]
                means = np.array([r["mean"] for r in rows])
                seconds = np.array([r["second_moment"] for r in rows])
                cache[method,factor] = (means, seconds)
                cdfs, drifts = [], []
                for run in subset:
                    with h5py.File(root / f"run_{run['task']:03d}.h5", "r") as h5:
                        if not h5.attrs["complete"]:
                            raise ValueError("incomplete file")
                        z = h5[f"{method}/dt_{factor}/cos_theta"][:]
                    ordered = np.sort(z.ravel())
                    cdfs.append(np.searchsorted(ordered, grid, side="right")/len(ordered))
                    half = len(z)//2
                    drifts.append(float(z[half:].mean()-z[:half].mean()))
                cdfs = np.asarray(cdfs)
                empirical = cdfs.mean(axis=0)
                deviation = np.abs(cdfs[indices].mean(axis=1)-empirical).max(axis=1)
                # Simultaneous band on grid with Bonferroni over 72 distributions.
                radius = float(np.quantile(deviation, 1-.05/72))
                mean_se = means.std(ddof=1)/np.sqrt(10)
                second_se = seconds.std(ddof=1)/np.sqrt(10)
                drift_se = np.std(drifts, ddof=1)/np.sqrt(10)
                checks = {
                    "mean": abs(means.mean()-rows[0]["exact_mean"]) <= max(.01,3*mean_se),
                    "second": abs(seconds.mean()-rows[0]["exact_second"]) <= max(.01*abs(rows[0]["exact_second"]),3*second_se),
                    "cdf": np.abs(empirical-exact_cdf).max() <= radius,
                    "stationarity": abs(np.mean(drifts)) <= max(.01,multiplier*drift_se),
                    "norm_and_finite": all(r["checks"]["spin_norm"] and r["checks"]["finite"] for r in rows),
                }
                summaries.append({"temperature":temperature,"method":method,"factor":factor,
                    "mean":float(means.mean()),"mean_se":float(mean_se),"exact_mean":rows[0]["exact_mean"],
                    "grid":grid.tolist(),"empirical_cdf":empirical.tolist(),"exact_cdf":exact_cdf.tolist(),
                    "cdf_band_radius":radius,"checks":{k:bool(v) for k,v in checks.items()}})
        for left,right in ((('heun',2),('heun',4)),(('midpoint',2),('midpoint',4)),(('heun',4),('midpoint',4))):
            for oi,observable in enumerate(("mean","second_moment")):
                a,b = cache[left][oi],cache[right][oi]
                d = a-b
                ci = multiplier*d.std(ddof=1)/np.sqrt(10)
                tolerance = .02*max(abs(float(b.mean())),.01)
                comparisons.append({"temperature":temperature,"left":left,"right":right,
                    "observable":observable,"difference":float(d.mean()),"ci95_halfwidth":float(ci),
                    "tolerance":tolerance,"passed":bool(abs(d.mean())<=tolerance and abs(d.mean())<=ci)})
    passed = all(all(r["checks"].values()) for r in summaries) and all(r["passed"] for r in comparisons)
    return {"passed":passed,"production_enabled":False,"scope":"free-moment thermal benchmark only",
            "runs":120,"summaries":summaries,"comparisons":comparisons}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        report = audit(args.input_dir)
    except (ValueError,KeyError,OSError) as error:
        report = {"passed":False,"production_enabled":False,"error":str(error)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    if "summaries" in report:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4))
        for method in ("heun","midpoint"):
            rows=[r for r in report["summaries"] if r["method"]==method and r["factor"]==4]
            ax.errorbar([r["temperature"] for r in rows],[r["mean"] for r in rows],
                        yerr=[1.96*r["mean_se"] for r in rows],fmt="o",label=method)
        ax.plot([r["temperature"] for r in rows],[r["exact_mean"] for r in rows],label="exact Langevin")
        ax.set(xlabel="dimensionless temperature",ylabel="mean spin z"); ax.legend()
        fig.tight_layout(); fig.savefig(args.output.with_suffix(".png")); plt.close(fig)
    print(json.dumps({"passed":report["passed"],"production_enabled":False}),flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
