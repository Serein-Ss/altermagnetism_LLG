"""Whole-path bootstrap weights and nonstationary weighted path moments.

Use the same R1/R2 weights across all model comparisons. Paired generated
ensembles also share weights. Chunk weight rows at the caller to bound memory.
These are descriptive estimators, not a calibrated simultaneous confidence gate.
"""
import numpy as np
from tests.experiments.path_observables import path_observables


def bootstrap_weights(n,replicates,seed):
    if not isinstance(n,int) or not isinstance(replicates,int) or n<2 or replicates<1:
        raise ValueError('at least two independent paths and positive replicate count required')
    indices=np.random.default_rng(seed).integers(n,size=(replicates,n))
    return np.stack([np.bincount(row,minlength=n) for row in indices])/n


def weighted_observables(paths,weights,temporal_lags=(1,2,4,8,16,32),spatial_lags=(1,2,4,8,12)):
    s=np.asarray(paths,dtype=np.float64);w=np.asarray(weights,dtype=np.float64)
    # Reuse established geometry/initial/lag validation on this one family.
    path_observables(s,temporal_lags=temporal_lags,spatial_lags=spatial_lags)
    if w.ndim!=2 or w.shape[1]!=len(s) or len(w)<1 or not np.isfinite(w).all() or (w<0).any() or not np.allclose(w.sum(1),1,atol=1e-12,rtol=0):
        raise ValueError('nonnegative normalized whole-path weights required')
    def average(a):return (w@a.reshape(len(s),-1)).reshape(len(w),*a.shape[1:])
    mean=average(s);mag=s.mean(axis=2);magmean=average(mag)
    result=dict(magnetization_mean=magmean,
        magnetization_variance=average(mag**2)-magmean**2,
        spin_ensemble_variance=average((s*s).sum(-1).mean(-1))-(mean*mean).sum(-1).mean(-1),
        spatial_spin_variance=average(np.square(s-mag[:,:,None]).sum(-1).mean(-1)))
    for lag in temporal_lags:
        dot=average((s[:,:-lag]*s[:,lag:]).sum(-1).mean(-1))
        result[f'temporal_dot_lag{lag}']=dot
        result[f'temporal_connected_lag{lag}']=dot-(mean[:,:-lag]*mean[:,lag:]).sum(-1).mean(-1)
        result[f'increment_second_moment_lag{lag}']=average(np.square(s[:,lag:]-s[:,:-lag]).sum(-1).mean(-1))
    for lag in spatial_lags:
        dot=average((s[:,:,:-lag]*s[:,:,lag:]).sum(-1).mean(-1))
        result[f'spatial_dot_lag{lag}']=dot
        result[f'spatial_connected_lag{lag}']=dot-(mean[:,:,:-lag]*mean[:,:,lag:]).sum(-1).mean(-1)
    return result
