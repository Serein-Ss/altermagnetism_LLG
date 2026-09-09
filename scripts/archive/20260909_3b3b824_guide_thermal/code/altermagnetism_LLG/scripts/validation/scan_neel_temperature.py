"""One zero-field four-chain temperature pilot; never self-certifies equilibrium."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"scripts"/"core"))
from unified_llg import GomonayModelAdapter, UnifiedLLGSolver, collinear_state, common_observables
from scripts.datasets.generate_v3_gomonay_shard import energy_components, spatial_summaries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size",type=int,required=True)
    p.add_argument("--temperature",type=float,required=True)
    p.add_argument("--steps",type=int,required=True)
    p.add_argument("--dt-fs",type=float,default=.05)
    p.add_argument("--save-every",type=int,default=200)
    p.add_argument("--alpha",type=float,default=.1)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--afm-control",action="store_true")
    p.add_argument("--thermal-certificate",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if not json.loads(args.thermal_certificate.read_text()).get("passed",False):
        raise ValueError("thermal gate has not passed")
    if args.size<4 or args.temperature<=0 or args.steps<10000 or args.steps%args.save_every or args.dt_fs<=0:
        p.error("invalid scan size, temperature, steps or time grid")
    if args.output.exists() or args.output.with_suffix(".h5.partial").exists():
        raise FileExistsError(args.output)
    model=GomonayModelAdapter(alternating_exchange=not args.afm_control)
    solver=UnifiedLLGSolver(model,alpha=args.alpha)
    state=collinear_state(4,2,args.size,args.size,antiferromagnetic=True,device="cpu",dtype=torch.float64)
    state[1]=-state[1]
    seeds=[int.from_bytes(hashlib.sha256(f"tn:{args.seed}:{i}".encode()).digest()[:7],"little") for i in range(4)]
    for i in (2,3):
        v=torch.randn(state[i].shape,generator=torch.Generator().manual_seed(seeds[i]+1),dtype=torch.float64)
        state[i]=v/v.norm(dim=-1,keepdim=True)
    generators=[torch.Generator().manual_seed(seed) for seed in seeds]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    partial=args.output.with_suffix(".h5.partial")
    frames=args.steps//args.save_every+1
    dt=args.dt_fs*1e-15
    start=time.perf_counter()
    with h5py.File(partial,"w") as h5:
        h5.attrs.update({"status":"raw_equilibrium_pilot","temperature":args.temperature,"size":args.size,
                         "dt":dt,"alpha":args.alpha,"drive":0.,"external_field":0.,
                         "metadata":json.dumps(model.metadata()),"afm_control":args.afm_control,
                         "thermal_certificate_sha256":hashlib.sha256(args.thermal_certificate.read_bytes()).hexdigest()})
        h5.create_dataset("time",data=np.arange(frames)*dt*args.save_every)
        h5.create_dataset("chain_seed",data=seeds)
        spins=h5.create_dataset("spins",shape=(4,frames,2,args.size,args.size,3),dtype="f4",
            chunks=(1,1,2,min(16,args.size),min(16,args.size),3),compression="gzip",shuffle=True)
        obs=h5.create_dataset("observables",shape=(4,frames,8),dtype="f8")
        h5.attrs["observable_names"]=json.dumps(["energy_per_spin","neel_norm","abs_nz","nz2","nz4","q0_power","wall_density","nz"])
        h5.create_dataset("energy_components",shape=(4,frames,4),dtype="f8")
        max_error=0.
        for step in range(args.steps+1):
            if step%args.save_every==0:
                frame=step//args.save_every
                _,neel=common_observables(state)
                structure,wall=spatial_summaries(state)
                energy=energy_components(model,state)
                z=neel[:,2]
                values=torch.stack((energy.sum(1)/(2*args.size**2),neel.norm(dim=1),z.abs(),z*z,z**4,
                                    structure[:,0],wall,z),dim=1)
                spins[:,frame]=state.numpy().astype("f4")
                obs[:,frame]=values.numpy()
                h5["energy_components"][:,frame]=energy.numpy()
                max_error=max(max_error,float((state.norm(dim=-1)-1).abs().max()))
            if step<args.steps:
                state=solver.stochastic_heun_step(state,step*dt,dt,args.temperature,generators)
            if step and step%10000==0:
                h5.flush()
                print(f"step={step}/{args.steps} elapsed={time.perf_counter()-start:.1f}s",flush=True)
        h5.attrs["max_spin_norm_error"]=max_error
        h5.attrs["complete"]=True
    partial.replace(args.output)


if __name__=="__main__":
    main()
