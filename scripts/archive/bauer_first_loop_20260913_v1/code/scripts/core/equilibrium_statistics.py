"""Conservative multichain diagnostics for stationary, equally spaced samples."""
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata


def integrated_time(values):
    x=np.asarray(values,dtype=float)
    x=x-x.mean()
    n=len(x)
    if n<8 or np.dot(x,x)==0:
        return float("inf")
    f=np.fft.rfft(x,n=2*n)
    ac=np.fft.irfft(f*f.conj())[:n]/np.arange(n,0,-1)
    ac/=ac[0]
    total=0.
    previous=float("inf")
    for k in range(1,n-1,2):
        pair=min(previous,float(ac[k]+ac[k+1]))
        if pair<=0:
            break
        total+=pair
        previous=pair
    return max(1.,1+2*total)


def diagnostics(chains):
    """Input [chain,draw]; rank/folded split Rhat and conservative ESS."""
    x=np.asarray(chains,dtype=float)
    if x.ndim!=2 or x.shape[0]<4 or x.shape[1]<16 or not np.isfinite(x).all():
        raise ValueError("need >=4 finite chains and >=16 draws per chain")
    half=x.shape[1]//2
    split=np.concatenate((x[:,:half],x[:,-half:]),axis=0)
    def rhat(values):
        ranks=rankdata(values).reshape(values.shape)
        z=ndtri((ranks-.375)/(values.size+.25))
        within=z.var(axis=1,ddof=1).mean()
        if within<=0:
            return float("inf")
        between=half*z.mean(axis=1).var(ddof=1)
        return float(np.sqrt(((half-1)*within/half+between/half)/within))
    rh=max(rhat(split),rhat(abs(split-np.median(split))))
    tau=np.array([integrated_time(row) for row in x])
    ess=x.shape[1]/tau
    return {"rhat":rh,"tau_max_saved_samples":float(tau.max()),
            "ess_per_chain":ess.tolist(),"ess_total":float(ess.sum()),
            "passed":bool(rh<1.05 and ess.min()>=200 and ess.sum()>=1000)}
