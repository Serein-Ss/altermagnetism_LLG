"""Validated CPU reference entry point; no default sample size or certification.

Callers must supply registered initial states and disjoint integer noise seeds.
The returned manifest describes a numerical reference, not a continuum certificate.
"""
import numpy as np
from scripts.core.bauer_fast import ensemble


def validate_inputs(initials,seeds,*,dt,duration,save_dt,anisotropy,damping,theta,method,forbidden_seeds=()):
    states=np.asarray(initials,dtype=np.float64)
    seeds=np.asarray(seeds)
    if states.ndim!=3 or states.shape[-1]!=3 or min(states.shape[:2])<1:
        raise ValueError('expected nonempty [path,site,3] initial states')
    if not np.isfinite(states).all() or np.max(np.abs(np.linalg.norm(states,axis=-1)-1))>1e-10:
        raise ValueError('finite float64 unit initial states required; do not silently normalize')
    if seeds.shape!=(len(states),) or seeds.dtype.kind not in 'iu' or np.any(seeds<0) or np.any(seeds>2**32-1):
        raise ValueError('one uint32-range integer noise seed per path required')
    if len(np.unique(seeds))!=len(seeds) or np.intersect1d(seeds,forbidden_seeds).size:
        raise ValueError('duplicate or forbidden noise seed')
    values=np.asarray([dt,duration,save_dt,anisotropy,damping,theta])
    if not np.isfinite(values).all() or min(dt,duration,save_dt)<=0 or damping<0 or theta<0:
        raise ValueError('invalid reduced parameters')
    if method not in ('mt','heun'):raise ValueError('unknown integrator')
    steps=int(round(duration/dt));stride=int(round(save_dt/dt))
    if min(steps,stride)<1 or not np.isclose(steps*dt,duration,rtol=0,atol=1e-10) or not np.isclose(stride*dt,save_dt,rtol=0,atol=1e-10) or steps%stride:
        raise ValueError('duration and save_dt must align exactly with integration grid')
    return states,seeds.astype(np.int64),steps,stride


def generate(initials,seeds,*,dt,duration,save_dt,anisotropy,damping,theta,method,forbidden_seeds=()):
    states,noise,steps,stride=validate_inputs(initials,seeds,dt=dt,duration=duration,save_dt=save_dt,
        anisotropy=anisotropy,damping=damping,theta=theta,method=method,forbidden_seeds=forbidden_seeds)
    paths,errors=ensemble(states,noise,dt,steps,stride,anisotropy,damping,theta,method=0 if method=='mt' else 1)
    if not np.array_equal(paths[:,0],states) or not np.isfinite(paths).all():
        raise RuntimeError('invalid reference trajectories')
    unit_error=float(np.max(np.abs(np.linalg.norm(paths,axis=-1)-1)))
    if unit_error>1e-10:raise RuntimeError('reference spin constraint violated')
    return paths,dict(method=method,device='cpu',equation='bauer_ll',exchange=1.,boundary='open',
        anisotropy=anisotropy,damping=damping,theta=theta,dt=dt,duration=duration,save_dt=save_dt,
        steps=steps,stride=stride,noise_seeds=noise.tolist(),paths=len(paths),frames=paths.shape[1],
        projection=True,max_raw_norm_error=float(errors.max()),max_saved_norm_error=unit_error,
        scope='Numerical reference only; initial-family split, external seed provenance, timestep convergence and sample precision require separate audit.')
