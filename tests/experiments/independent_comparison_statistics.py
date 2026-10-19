"""Prefix moment discrepancies with shared-reference whole-path bootstrap.

Percentile intervals here are descriptive, pointwise and uncalibrated for the
nonsmooth RMSE difference near zero. They cannot implement an equivalence gate.
All bootstrap draws are returned so later summaries can preserve shared samples.
"""
import numpy as np
from tests.experiments.trajectory_bootstrap import bootstrap_weights,weighted_observables


def prefix_discrepancies(g,r1,r2,ends=(10,50,200)):
    if set(g)!=set(r1) or set(r1)!=set(r2):raise ValueError('observable sets differ')
    frames=g['magnetization_mean'].shape[1]
    if any(not isinstance(end,int) or not 0<=end<frames for end in ends):raise ValueError('prefix outside observed path')
    result={}
    for key in g:
        if g[key].shape!=r1[key].shape or r1[key].shape!=r2[key].shape:raise ValueError('observable shapes differ')
        lag=int(key.rsplit('lag',1)[1]) if key.startswith(('temporal_','increment_')) else 0
        for end in ends:
            stop=end+1-lag
            if stop<=0:continue # A pair must have BOTH endpoints inside the prefix.
            delta=g[key][:,:stop]-r1[key][:,:stop]
            baseline=r1[key][:,:stop]-r2[key][:,:stop]
            axes=tuple(range(1,delta.ndim))
            a=np.sqrt(np.square(delta).mean(axis=axes));b=np.sqrt(np.square(baseline).mean(axis=axes))
            prefix=f'frame{end}/{key}'
            result[prefix+'/generated_reference_rmse']=a
            result[prefix+'/reference_reference_rmse']=b
            result[prefix+'/excess_rmse']=a-b
    return result


def compare_ensembles(g,r1,r2,*,reference_seed,generated_seed,replicates=2000,batch_size=32,
                      ends=(10,50,200),temporal_lags=(1,2,4,8,16,32),spatial_lags=(1,2,4,8,12)):
    if g.shape!=r1.shape or r1.shape!=r2.shape:raise ValueError('matched sample counts and grids required')
    if not np.array_equal(g[0,0].astype('f4'),r1[0,0].astype('f4')) or not np.array_equal(r1[0,0],r2[0,0]):
        raise ValueError('initial families do not match')
    if batch_size<1:raise ValueError('positive bootstrap batch required')
    options=dict(temporal_lags=temporal_lags,spatial_lags=spatial_lags)
    n=len(g);uniform=np.full((1,n),1/n)
    point=prefix_discrepancies(*(weighted_observables(a,uniform,**options) for a in (g,r1,r2)),ends=ends)
    # Independent R1/R2, shared across model comparisons by fixed seed.
    weights=[bootstrap_weights(n,replicates,s) for s in (generated_seed,reference_seed,reference_seed+1)]
    draws={key:[] for key in point}
    for start in range(0,replicates,batch_size):
        values=prefix_discrepancies(*(weighted_observables(a,w[start:start+batch_size],**options)
            for a,w in zip((g,r1,r2),weights)),ends=ends)
        for key,value in values.items():draws[key].append(value)
    draws={key:np.concatenate(value) for key,value in draws.items()}
    summary={key:dict(estimate=float(point[key][0]),
        descriptive_percentile_interval95=np.quantile(draws[key],[.025,.975]).tolist()) for key in point}
    return summary,draws
