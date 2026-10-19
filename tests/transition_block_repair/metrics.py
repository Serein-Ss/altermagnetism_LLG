"""Two-frame statistics: never evaluate unavailable lag2/lag5 observables."""
import numpy as np
from tests.bauer_p2_20260923.common import energy


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
