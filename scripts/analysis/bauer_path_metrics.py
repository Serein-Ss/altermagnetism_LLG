"""Two-frame statistics: never evaluate unavailable lag2/lag5 observables."""
import numpy as np
from scripts.analysis.bauer_energy import energy


def statistics(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=4 or x.shape[1]!=2 or len(x)<2:raise ValueError('independent two-frame ensemble required')
    d=x[:,1]-x[:,0];centered=d-d.mean(0)
    cov=np.einsum('bni,bnj->nij',centered,centered)/(len(x)-1)
    e=energy(x[:,1]);m=x[:,1,:,2].mean(-1)
    return dict(increment_mean=d.mean(0),increment_cov=cov,
                increment_variance=np.array(np.trace(cov,axis1=-2,axis2=-1).mean()),
                energy_mean=np.array(e.mean()),energy_var=np.array(e.var(ddof=1)),
                mz_mean=np.array(m.mean()),mz_var=np.array(m.var(ddof=1)),
                S0_overlap=np.array((x[:,1]*x[:,0]).sum(-1).mean()),
                spatial=np.array([(x[:,1,:-lag]*x[:,1,lag:]).sum(-1).mean() for lag in (1,2,4)]))


def compare_first(g,r,repeats=100):
    g=np.asarray(g)[:,:2];r=np.asarray(r)[:,:2];a=statistics(g);b=statistics(r)
    rows={k:dict(generated=a[k].tolist(),reference=b[k].tolist(),difference=(a[k]-b[k]).tolist(),
                 rmse=float(np.sqrt(np.mean((a[k]-b[k])**2)))) for k in a}
    rng=np.random.default_rng(93977);boot={k:[] for k in a}
    for _ in range(repeats):
        ga=statistics(g[rng.integers(len(g),size=len(g))]);rb=statistics(r[rng.integers(len(r),size=len(r))])
        for k in a:boot[k].append(ga[k]-rb[k])
    if repeats:
        for k in a:rows[k]['pointwise_ci95']=np.quantile(boot[k],[.025,.975],axis=0).tolist()
    return rows

"""Fixed-parent cross-site fluctuation tensors; no sites-as-replicates."""
import numpy as np


def spatial_moments(paths):
    a=np.asarray(paths,dtype=np.float64)
    if a.ndim!=4 or a.shape[1]!=2 or a.shape[-1]!=3 or len(a)<2:raise ValueError('two-frame independent branches required')
    d=a[:,1]-a[:,0];mean=d.mean(0);flat=(d-mean).reshape(len(a),-1)
    covariance=(flat.T@flat)/(len(a)-1)
    return mean,covariance.reshape(d.shape[1],3,d.shape[1],3).transpose(0,2,1,3)


def compare_spatial(generated,reference):
    g=np.asarray(generated);r=np.asarray(reference)
    if g.shape[1:]!=r.shape[1:] or min(len(g),len(r))<4 or len(g)%2 or len(r)%2:raise ValueError('matching grid and even independent branch ensembles >=4 required')
    for x in (g,r):
        if not np.allclose(x[:,0],r[:1,0],atol=1e-6,rtol=0):raise ValueError('different actual initial states')
    mg,cg=spatial_moments(g);mr,cr=spatial_moments(r);n=g.shape[2];rows=[]
    ca=spatial_moments(g[:len(g)//2])[1]-spatial_moments(r[:len(r)//2])[1]
    cb=spatial_moments(g[len(g)//2:])[1]-spatial_moments(r[len(r)//2:])[1]
    for lag in (0,1,2,4,8,16):
        if lag>=n:continue
        i=np.arange(n-lag);j=i+lag;diff=cg[i,j]-cr[i,j]
        rows.append(dict(lag=lag,site_pairs=len(i),covariance_rmse=float(np.sqrt(np.mean(diff**2))),
            covariance_frobenius_rms=float(np.sqrt(np.mean(np.sum(diff**2,axis=(-1,-2))))),
            covariance_squared_bias_crossfold=float(np.mean(np.sum(ca[i,j]*cb[i,j],axis=(-1,-2)))),
            generated_trace=float(np.trace(cg[i,j],axis1=-2,axis2=-1).mean()),
            reference_trace=float(np.trace(cr[i,j],axis1=-2,axis2=-1).mean())))
    difference=mg-mr
    return dict(lags=rows,generated_branches=len(g),reference_branches=len(r),
        fold_branches=[len(g)//2,len(r)//2],site_mean_rmse=np.sqrt(np.mean(difference**2,axis=-1)).tolist(),
        endpoint_mean_rmse=float(np.sqrt(np.mean(difference[[0,1,-2,-1]]**2))),
        interior_mean_rmse=float(np.sqrt(np.mean(difference[2:-2]**2))),
        covariance_squared_bias_crossfold_full=float(np.mean(np.sum(ca*cb,axis=(-1,-2)))),
        covariance_frobenius_error=np.sqrt(np.sum((cg-cr)**2,axis=(-1,-2))).tolist(),
        scope='Exploratory fixed-parent two independent folds, unbiased squared covariance mismatch under iid branches; may be negative. No CI/acceptance gate. Sites/pairs are components, not independent replicates; reused DEV not final.')
