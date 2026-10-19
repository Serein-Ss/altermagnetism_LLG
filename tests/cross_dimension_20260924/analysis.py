"""Whole-branch 2D metrics; reference Hamiltonian, no synthetic energy surrogate."""
import numpy as np
import torch
from tests.physical_audit_20260917.gallery import triangular_charge


@torch.no_grad()
def observables(raw,cell):
    raw=np.asarray(raw,dtype=np.float64);b,t=raw.shape[:2]
    s=raw.reshape(b,t,-1,3);d=np.diff(s,axis=1)
    energy=[]
    for branch in raw:
        energy.append((cell.energy(torch.tensor(branch))/s.shape[2]).numpy())
    q,qi=triangular_charge(raw)
    spatial=[]
    for axis in (2,3):
        for lag in (1,2,4):
            a=[slice(None)]*raw.ndim;c=a.copy();a[axis]=slice(lag,None);c[axis]=slice(None,-lag)
            spatial.append((raw[tuple(a)]*raw[tuple(c)]).sum(-1).reshape(b,t,-1).mean(-1))
    z=raw[...,2].mean(-1);z=z-z.mean((-2,-1),keepdims=True)
    spectrum=abs(np.fft.rfft2(z,axes=(-2,-1)))**2/(z.shape[-2]*z.shape[-1])
    return dict(mz=s[...,2].mean(-1),energy=np.array(energy),Q=q,Q_interior=qi,
        overlap=(s*s[:,:1]).sum(-1).mean(-1),spatial=np.stack(spatial,-1),
        two_time=np.einsum('btnc,bunc->btu',s,s)/s.shape[2],
        increment_power=d.square().sum(-1).mean(-1) if hasattr(d,'square') else (d*d).sum(-1).mean(-1),
        adjacent_increment_dot=(d[:,1:]*d[:,:-1]).sum(-1).mean(-1),
        low_q=spectrum[...,:4,:4],increment=d,
        norm_error=float(abs(np.linalg.norm(s,axis=-1)-1).max()))


def increment_variance(d):
    return d.var(0,ddof=1).sum(-1).mean(-1)


def compare(g,r,seed=924):
    rng=np.random.default_rng(seed);out={};series={}
    for key in ('mz','energy','Q','Q_interior','overlap','spatial','two_time','increment_power','adjacent_increment_dot','low_q'):
        a,b=g[key],r[key]
        out[key+'_mean_rmse']=float(np.sqrt(np.mean((a.mean(0)-b.mean(0))**2)))
        out[key+'_var_rmse']=float(np.sqrt(np.mean((a.var(0,ddof=1)-b.var(0,ddof=1))**2)))
        if key in ('mz','energy','overlap','Q','increment_power','adjacent_increment_dot'):
            boot=[]
            for _ in range(100):
                aa=a[rng.integers(len(a),size=len(a))];bb=b[rng.integers(len(b),size=len(b))]
                boot.append(aa.mean(0)-bb.mean(0))
            series[key]=dict(generated_mean=a.mean(0).tolist(),reference_mean=b.mean(0).tolist(),
                generated_sd=a.std(0,ddof=1).tolist(),reference_sd=b.std(0,ddof=1).tolist(),
                difference_ci=np.quantile(boot,[.025,.975],axis=0).tolist())
    a,b=g['increment'],r['increment'];va,vb=increment_variance(a),increment_variance(b)
    boot=[]
    for _ in range(100):
        boot.append(increment_variance(a[rng.integers(len(a),size=len(a))])-increment_variance(b[rng.integers(len(b),size=len(b))]))
    series['increment_variance']=dict(generated_mean=va.tolist(),reference_mean=vb.tolist(),
        difference_ci=np.quantile(boot,[.025,.975],axis=0).tolist())
    out['first_increment_variance_ratio']=float(va[0]/vb[0])
    out['increment_variance_rmse']=float(np.sqrt(np.mean((va-vb)**2)))
    cov=lambda d:np.einsum('btnc,btnd->tncd',d-d.mean(0),d-d.mean(0))/(len(d)-1)
    out['increment_cov_rmse']=float(np.sqrt(np.mean((cov(a)-cov(b))**2)))
    out['norm_error']=g['norm_error']
    return dict(metrics=out,series=series,reference_branches=len(b),generated_branches=len(a),
        uncertainty='100 independent whole-branch bootstrap resamples per fixed parent; pointwise, not equivalence',
        scientific_certification=False)
