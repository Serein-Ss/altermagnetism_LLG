"""Reduced sublattice equilibrium observables and damped AFMR fits."""
import h5py
import numpy as np
from scipy.optimize import curve_fit
from scripts.literature.workflow import analysis_arguments,require_complete,save_json


def damped_afmr(time,signal):
    time=np.asarray(time)-time[0]; signal=np.asarray(signal)
    if len(time)<12 or np.ptp(signal)<1e-10:
        return {'resolved':False,'reason':'insufficient temporal samples or amplitude'}
    freq=np.fft.rfftfreq(len(time),np.median(np.diff(time)))
    peak=1+np.argmax(np.abs(np.fft.rfft(signal-signal.mean()))[1:])
    def model(t,a,omega,tau,phase,offset): return a*np.exp(-t/tau)*np.cos(omega*t+phase)+offset
    try:
        values,cov=curve_fit(model,time,signal,p0=[np.ptp(signal)/2,2*np.pi*freq[peak],time[-1]/2,0,signal.mean()],
                             bounds=([-np.inf,0,1e-12,-2*np.pi,-np.inf],
                                     [np.inf,np.pi/np.min(np.diff(time)),np.inf,2*np.pi,np.inf]),maxfev=30000)
    except (RuntimeError,ValueError) as exc:
        return {'resolved':False,'reason':str(exc)}
    return {'resolved':True,'parameters_order':['amplitude','omega','tau','phase','offset'],
            'parameters':values.tolist(),'fit_covariance':cov.tolist() if np.isfinite(cov).all() else None,
            'residual_rmse':float(np.sqrt(np.mean((signal-model(time,*values))**2))),
            'uncertainty_warning':'Fit covariance is conditional; independent replicate/block uncertainty still required.'}


def main():
    args=analysis_arguments()
    with h5py.File(args.input) as h:
        require_complete(h); t=h['time'][:]; a=h['m_a'][:]; b=h['m_b'][:]
        if h.attrs.get('model_kind','')=='AFM_LLB':
            result={'afmr':[damped_afmr(t,a[:,i,2]) for i in range(a.shape[1])],
                    'model':'AFM_LLB','certified':False,
                    'equilibrium_input':'published_fit_or_explicit_me_not_new_ASD_fit'}
        else:
            start=len(t)//2; nsub=int(np.prod(h['spins'].shape[2:5])*2)
            result={'measurement_start_reduced':float(t[start]),'sites_per_sublattice':nsub,
                    'sublattice_a_mean_length':np.linalg.norm(a[start:],axis=-1).mean(0).tolist(),
                    'sublattice_b_mean_length':np.linalg.norm(b[start:],axis=-1).mean(0).tolist(),
                    'theta_times_longitudinal_susceptibility':(nsub*a[start:,:,:2].var(axis=0,ddof=1).mean(-1)).tolist(),
                    'susceptibility_definition':'N_sub [Var(m_a,x)+Var(m_a,y)]/(2 theta); reported numerator, published in-plane average',
                    'afmr':[damped_afmr(t,a[:,i,2]) for i in range(a.shape[1])],
                    'max_norm_error':float(h['max_norm_error'][:].max()),
                    'certified':False,'remaining':'Independent seeds, equilibration/time/size convergence and reference comparison.'}
    save_json(args.output,result)


if __name__=='__main__': main()


