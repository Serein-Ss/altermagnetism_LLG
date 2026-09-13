"""Multichain equilibrium checks and finite-size temperature summaries.

Does not declare thermodynamic TN from one noisy peak or from an incomplete grid.
Binder intersections are candidate estimates requiring refined-grid bootstrapping.
"""
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"/"core"))
from equilibrium_statistics import diagnostics


def inspect(path):
    with h5py.File(path,"r") as h5:
        if not h5.attrs.get("complete",False) or h5.attrs["drive"]!=0 or h5.attrs["external_field"]!=0:
            raise ValueError("incomplete or driven scan")
        obs=h5["observables"][:]
        names=json.loads(h5.attrs["observable_names"])
        burn=obs.shape[1]//2
        tail=obs[:,burn:]
        # Even observables permit symmetry-related basins without requiring rare
        # global reversals. This is not a claim of measured inter-basin weights.
        checks={name:diagnostics(tail[:,:,i]) for i,name in enumerate(names) if name!="nz"}
        tau=max(v["tau_max_saved_samples"] for v in checks.values())
        equilibrium=all(v["passed"] for v in checks.values()) and burn>=20*tau
        size=int(h5.attrs["size"]); temperature=float(h5.attrs["temperature"])
        nz2=float(tail[:,:,3].mean()); nz4=float(tail[:,:,4].mean())
        energy=tail[:,:,0]*(2*size**2)
        kB=1.380649e-23
        return {"source":str(path.resolve()),"size":size,"temperature":temperature,
            "equilibrium_even_observables_passed":bool(equilibrium),"production_enabled":False,
            "diagnostics":checks,"burn_frames":burn,"tau_max_saved_samples":tau,
            "mean_neel_norm":float(tail[:,:,1].mean()),"mean_abs_nz":float(tail[:,:,2].mean()),
            "binder":None if nz2==0 else 1-nz4/(3*nz2*nz2),
            "susceptibility":float(2*size**2/(kB*temperature)*(nz2-tail[:,:,2].mean()**2)),
            "heat_capacity":float(energy.var()/(kB*temperature**2)),
            "max_spin_norm_error":float(h5.attrs["max_spin_norm_error"])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows=[inspect(path) for path in args.input]
    if len({(r["size"],r["temperature"]) for r in rows})!=len(rows):
        raise ValueError("duplicate size/temperature; split model variants into separate analyses")
    report={"status":"temperature_pilot_diagnostics","production_enabled":False,
            "TN_certified":False,"rows":rows,
            "remaining":"refined temperature grid, block-bootstrap Binder intersections and dt/size certification"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for size in sorted({r["size"] for r in rows}):
        part=sorted([r for r in rows if r["size"]==size],key=lambda r:r["temperature"])
        for ax,key in zip(axes,("mean_abs_nz","susceptibility","binder")):
            ax.plot([r["temperature"] for r in part],[r[key] for r in part],"o-",label=f"L={size}")
            ax.set(xlabel="temperature K",ylabel=key); ax.legend()
    fig.tight_layout();fig.savefig(args.output.with_suffix(".png"));plt.close(fig)


if __name__=="__main__":
    main()
