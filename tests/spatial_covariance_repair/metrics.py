"""Fixed-parent cross-site fluctuation tensors; no sites-as-replicates."""
import numpy as np


def moments(paths):
    a=np.asarray(paths,dtype=np.float64)
    if a.ndim!=4 or a.shape[1]!=2 or a.shape[-1]!=3 or len(a)<2:raise ValueError('two-frame independent branches required')
    d=a[:,1]-a[:,0];mean=d.mean(0);flat=(d-mean).reshape(len(a),-1)
    covariance=(flat.T@flat)/(len(a)-1)
    return mean,covariance.reshape(d.shape[1],3,d.shape[1],3).transpose(0,2,1,3)


def compare(generated,reference):
    g=np.asarray(generated);r=np.asarray(reference)
    if g.shape[1:]!=r.shape[1:] or min(len(g),len(r))<4 or len(g)%2 or len(r)%2:raise ValueError('matching grid and even independent branch ensembles >=4 required')
    for x in (g,r):
        if not np.allclose(x[:,0],r[:1,0],atol=1e-6,rtol=0):raise ValueError('different actual initial states')
    mg,cg=moments(g);mr,cr=moments(r);n=g.shape[2];rows=[]
    ca=moments(g[:len(g)//2])[1]-moments(r[:len(r)//2])[1]
    cb=moments(g[len(g)//2:])[1]-moments(r[len(r)//2:])[1]
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
