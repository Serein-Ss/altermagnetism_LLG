"""Extract decorrelated positive-basin candidates from an audited four-chain run.

This writes pool_candidates, not equilibrium_certified: dt, save-cadence and basin
calibration certificates must be supplied by the independent production audit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--diagnostics",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report=json.loads(args.diagnostics.read_text())
    row=next(r for r in report["rows"] if Path(r["source"]).resolve()==args.input.resolve())
    if not row["equilibrium_even_observables_passed"]:
        raise ValueError("equilibrium diagnostics failed; do not create a pool")
    stride=max(1,int(np.ceil(5*row["tau_max_saved_samples"])))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    partial=args.output.with_suffix(".h5.partial")
    if partial.exists():
        raise FileExistsError(partial)
    with h5py.File(args.input,"r") as source,h5py.File(partial,"w") as target:
        obs=source["observables"][:]
        indices=[(chain,frame) for chain in range(4)
                 for frame in range(row["burn_frames"],obs.shape[1],stride)
                 if obs[chain,frame,7]>0]
        if not indices:
            raise ValueError("no positive-basin candidates")
        shape=(len(indices),)+source["spins"].shape[2:]
        spins=target.create_dataset("spins",shape=shape,dtype="f4",compression="gzip",shuffle=True)
        ids=[]
        source_hash=hashlib.sha256(args.input.read_bytes()).hexdigest()
        for k,(chain,frame) in enumerate(indices):
            spins[k]=source["spins"][chain,frame]
            payload=f"{source_hash}:{chain}:{frame}".encode()
            ids.append(int.from_bytes(hashlib.sha256(payload).digest()[:7],"little"))
        ids=np.asarray(ids,dtype="i8")
        target.create_dataset("initial_state_id",data=ids)
        target.create_dataset("source_chain_frame",data=indices)
        # Pool membership controls splits before any dynamics are generated.
        ordering=np.argsort(ids)
        splits=np.full(len(ids),2,dtype="i1")
        splits[ordering[:int(.8*len(ids))]]=0
        splits[ordering[int(.8*len(ids)):int(.9*len(ids))]]=1
        target.create_dataset("split",data=splits)
        target.attrs.update({"certification_status":"pool_candidates_not_production_certified",
            "source_sha256":source_hash,"temperature":row["temperature"],"size":row["size"],
            "afm_control":bool(source.attrs["afm_control"]),"sample_spacing_frames":stride,
            "conditioning":"positive nz; stable-basin calibration still required"})
    partial.replace(args.output)


if __name__=="__main__":
    main()
