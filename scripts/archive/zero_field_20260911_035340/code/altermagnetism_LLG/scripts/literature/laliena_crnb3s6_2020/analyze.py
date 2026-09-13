"""Fold evidence and periodic soliton position/width/topological diagnostics."""
import json
import h5py
import numpy as np
from scripts.literature.workflow import analysis_arguments,require_complete,save_json


def texture_observables(spins,dx):
    s=np.asarray(spins); phi=np.arctan2(-s[...,0],s[...,1])
    delta=np.angle(np.exp(1j*(np.roll(phi,-1,axis=-1)-phi)))
    winding=delta.sum(-1)/(2*np.pi)
    n=s.shape[-2]; x=np.arange(n)*dx; length=n*dx
    # Circular center handles a soliton crossing the periodic seam.
    phiprime=(delta+np.roll(delta,1,axis=-1))/(2*dx)
    weight=phiprime**2; moment=(weight*np.exp(2j*np.pi*x/length)).sum(-1)
    center=(np.angle(moment)%(2*np.pi))*length/(2*np.pi)
    distance=(x-center[...,None]+length/2)%length-length/2
    denom=weight.sum(-1)
    width=np.sqrt((weight*distance**2).sum(-1)/np.maximum(denom,1e-30))
    width=np.where(denom>1e-12,width,np.nan)
    return {'winding':winding,'center':center,'phase_gradient_rms_width':width,
            'max_abs_nz':np.abs(s[...,2]).max(-1),'mean_wavevector':2*np.pi*winding/length}


def main():
    args=analysis_arguments()
    if args.input.suffix=='.npz':
        with np.load(args.input) as data:
            if 'gamma' in data and data['gamma'].ndim==1:
                gamma=data['gamma']; i=int(np.argmax(gamma))
                fold=0<i<len(gamma)-1 and gamma[i]>gamma[i-1] and gamma[i]>gamma[i+1]
                result={'fold_bracketed':bool(fold),'sampled_gamma_max':float(gamma[i]),
                        'max_BVP_residual':float(data['residual'].max()),
                        'critical_gamma_certified':False,
                        'remaining':'Converge arclength step, extent, collocation mesh and compare corrected reference.'}
            else:
                result={'gamma':float(data['gamma']),'max_BVP_residual':float(data['residual']),
                        'max_norm_error':float(np.max(np.abs(np.linalg.norm(data['spins'],axis=-1)-1))),
                        'certified':False}
    else:
        with h5py.File(args.input) as h:
            require_complete(h); p=json.loads(h.attrs['protocol_json']); t=h['time'][:]
            rows=[texture_observables(s,p['dx']) for s in h['spins']]
            result={key:np.array([r[key] for r in rows]).tolist() for key in rows[0]}
            result['time']=t.tolist(); result['certified']=False
            length=h['spins'].shape[2]*p['dx']
            center=np.unwrap(np.array(result['center'])*2*np.pi/length,axis=0)*length/(2*np.pi)
            result['fitted_velocity']=[float(np.polyfit(t,center[:,b],1)[0]) for b in range(center.shape[1])]
            result['caution']='Center/width meaningful only for a single surviving soliton, not a disordered helix.'
            # Undefined widths stay explicit null, never a fabricated zero.
            result['phase_gradient_rms_width']=[[None if not np.isfinite(v) else v for v in row] for row in result['phase_gradient_rms_width']]
    save_json(args.output,result)


if __name__=='__main__': main()

