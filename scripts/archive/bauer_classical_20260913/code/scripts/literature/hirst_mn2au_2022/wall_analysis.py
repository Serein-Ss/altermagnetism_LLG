"""Published tanh width delta0; Neel wall width is pi*delta0, not delta0."""
import h5py
import numpy as np
from scipy.optimize import curve_fit
from scripts.literature.workflow import analysis_arguments,require_complete,save_json


def fit_wall(spins):
    # Input [batch,x,y,z,basis,xyz]; average transverse cells and interpolate
    # corner/body-offset Mn to common physical x/a coordinates.
    profiles=np.asarray(spins).mean((2,3)); nx=profiles.shape[1]
    x=np.arange(nx,dtype=float); common=x[2:-2]; output=[]
    if len(common)<8: raise ValueError('at least 12 x cells required for wall fitting')
    offsets=(0.,0.,.5,.5)
    for profile in profiles:
        sx=np.stack([np.interp(common,x+offset,profile[:,basis,0]) for basis,offset in enumerate(offsets)],-1)
        neel=(sx[:,0]+sx[:,3]-sx[:,1]-sx[:,2])/4
        fit,cov=curve_fit(lambda x,me,q,w:-me*np.tanh((x-q)/w),common,neel,
                         p0=[max(np.max(abs(neel)),.1),common[len(common)//2],min(30.,nx/8)],
                         bounds=([0.,common[0],.01],[1.1,common[-1],nx]),maxfev=10000)
        me,q,w=fit
        output.append({'me':float(me),'position_over_a':float(q),'delta0_over_a':float(w),
                       'neel_width_over_a':float(np.pi*w),
                       'fit_covariance':cov.tolist() if np.isfinite(cov).all() else None})
    return output


if __name__=='__main__':
    args=analysis_arguments()
    with h5py.File(args.input) as h:
        require_complete(h)
        result={'final_time':float(h['time'][-1]),'fits':fit_wall(h['spins'][-1]),
                'certified':False,'width_definition':'delta=pi*delta0; all lengths in a, no SI in fitting'}
    save_json(args.output,result)
