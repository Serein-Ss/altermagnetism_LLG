"""Million-component CPU FDT audit through the actual solver noise interface."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"/"core"))
from unified_llg import GomonayModelAdapter, NishinoFreeMomentHamiltonian, UnifiedLLGSolver


def inspect_noise(solver,temperature,dt,count,seed):
    state=torch.zeros((2,1,(count+5)//6,1,3),dtype=torch.float64)
    generator=torch.Generator().manual_seed(seed)
    first=solver.sample_thermal_field(state,temperature,dt,generator)
    second=solver.sample_thermal_field(state,temperature,dt,generator)
    draw=lambda value,step=dt: solver.sample_thermal_field(
        state,temperature,step,torch.Generator().manual_seed(value))
    # Independent expression of the Gilbert FDT convention used for this audit.
    sigma=np.sqrt(2*solver.alpha*solver.model.boltzmann*temperature/
                  (solver.model.gamma*solver.model.moment*dt))
    x=(first/sigma).numpy(); flat=x.ravel(); n=len(flat)
    correlations={
        "xyz":float(np.corrcoef(x[...,0].ravel(),x[...,1].ravel())[0,1]),
        "site":float(np.corrcoef(x[0].ravel(),x[1].ravel())[0,1]),
        "consecutive_time_draws":float(np.corrcoef(flat,(second/sigma).numpy().ravel())[0,1]),
    }
    half_ratio=float(draw(seed,dt/2).std()/first.std())
    checks={
        "mean":abs(float(flat.mean()))*np.sqrt(n)<5,
        "variance":abs(float(flat.var(ddof=1))-1)<max(.01,5*np.sqrt(2/(n-1))),
        "independence":max(map(abs,correlations.values()))<max(.01,5/np.sqrt(n)),
        "replay":torch.equal(first,draw(seed)),
        "different_seed":not torch.equal(first,draw(seed+1)),
        "dt_scaling":abs(half_ratio/np.sqrt(2)-1)<.01,
        "zero_temperature":torch.count_nonzero(solver.sample_thermal_field(
            state,0,dt,torch.Generator().manual_seed(seed))).item()==0,
    }
    checks={k:bool(v) for k,v in checks.items()}
    return {"temperature":temperature,"dt":dt,"alpha":solver.alpha,"components":n,
        "sigma":float(sigma),"mean_zscore":float(flat.mean()*np.sqrt(n)),
        "variance_ratio":float(flat.var(ddof=1)),"correlations":correlations,
        "half_dt_sigma_ratio":half_ratio,"checks":checks,"passed":all(checks.values())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--samples",type=int,default=1_000_000)
    args=p.parse_args()
    if args.samples<1_000_000:
        p.error("formal audit requires at least one million components")
    if args.output.exists():
        raise FileExistsError(args.output)
    rows=[]
    for model,temperatures,steps in (
        (GomonayModelAdapter(),(1.,5.,100.),(1e-16,5e-17)),
        (NishinoFreeMomentHamiltonian(),(1.,5.),(.005,.0025)),
    ):
        for alpha in (.01,.05):
            solver=UnifiedLLGSolver(model,alpha=alpha)
            for temperature in temperatures:
                for dt in steps:
                    row=inspect_noise(solver,temperature,dt,args.samples,20260910+len(rows))
                    row["model"]=model.metadata(); rows.append(row)
    report={"passed":all(r["passed"] for r in rows),"rows":rows,
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot([r["mean_zscore"] for r in rows],"o")
    axes[0].axhline(5,color="red");axes[0].axhline(-5,color="red")
    axes[0].set_ylabel("noise mean z-score")
    axes[1].plot([r["variance_ratio"] for r in rows],"o")
    axes[1].axhline(1.01,color="red");axes[1].axhline(.99,color="red")
    axes[1].set_ylabel("variance / theoretical variance")
    for ax in axes:
        ax.set_xlabel("preregistered condition index")
    fig.tight_layout();fig.savefig(args.output.with_suffix(".png"));plt.close(fig)
    print(json.dumps({"passed":report["passed"],"conditions":len(rows)}),flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
