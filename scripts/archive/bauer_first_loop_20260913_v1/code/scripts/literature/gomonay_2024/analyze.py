"""Space-time FFT on multiple BZ paths, or fitted wall position/phase/width."""
import json
import h5py
import numpy as np
from scipy.optimize import curve_fit
from scripts.literature.workflow import analysis_arguments,require_complete,save_json


def spectral_paths(h):
    t=h['time'][:]; nx,ny=h['spins'].shape[2:4]
    if len(t)<4 or not np.allclose(np.diff(t),np.diff(t)[0]):
        raise ValueError('uniform sampling with >=4 frames required for space-time FFT')
    # Every resolvable lattice wavevector on axes, diagonals and BZ boundary.
    path={(i,0) for i in range(nx)}|{(0,j) for j in range(ny)}
    path|={(i,round(i*ny/nx)%ny) for i in range(nx)}
    path|={(i,(-round(i*ny/nx))%ny) for i in range(nx)}
    path|={(i,ny//2) for i in range(nx)}|{(nx//2,j) for j in range(ny)}
    ij=np.array(sorted(path)); series=[]
    for frame in h['spins']:
        xy=frame[...,0]+1j*frame[...,1]
        fft=np.fft.fft2(xy,axes=(1,2))/(nx*ny)
        series.append(fft[:,ij[:,0],ij[:,1],:])
    series=np.asarray(series)
    spectrum=np.fft.fft(series*np.hanning(len(t))[:,None,None,None],axis=0)
    # Retain both signs of omega and both basis channels; summing branches
    # before analysis would conceal circular polarization and signed splitting.
    return {'omega':2*np.pi*np.fft.fftfreq(len(t),np.diff(t)[0]),
            'kx':2*np.pi*np.fft.fftfreq(nx)[ij[:,0]],'ky':2*np.pi*np.fft.fftfreq(ny)[ij[:,1]],
            'power':np.abs(spectrum)**2,'frequency_bin':2*np.pi/(len(t)*np.diff(t)[0])}


def wall_observables(h):
    p=json.loads(h.attrs['protocol_json']); spacing=np.sqrt(2) if p['orientation']=='110' else 1.
    nx=h['spins'].shape[2]; x_full=np.arange(nx)*spacing; x=x_full[2:-2]
    if len(x)<4: raise ValueError('wall fit requires at least eight long-axis cells')
    offsets=np.array([0.,.5]) if p['orientation']=='100' else np.array([0.,1.,1.,2.])/np.sqrt(2)
    rows=[]
    for frame in h['spins']:
        profile=frame.mean(2)
        coincident=np.empty((len(frame),len(x),len(offsets),3))
        for batch in range(len(frame)):
            for basis,offset in enumerate(offsets):
                for c in range(3):
                    coincident[batch,:,basis,c]=np.interp(x,x_full+offset,profile[batch,:,basis,c])
        a=coincident[:,:,0::2].mean(2); b=coincident[:,:,1::2].mean(2)
        n=(a-b)/2; m=(a+b)/2
        for b in range(len(n)):
            try:
                fit,_=curve_fit(lambda x,q,w:-np.tanh((x-q)/w),x,n[b,:,2],
                                 p0=[x[np.argmin(abs(n[b,:,2]))],10.],bounds=([x[0],.01],[x[-1],x[-1]]))
                q,w=fit; px=np.interp(q,x,n[b,:,0]); py=np.interp(q,x,n[b,:,1])
                phase=np.arctan2(py,px)
                rows.append((q,w,phase,float(np.max(np.abs(m[b,:,2])))))
            except (RuntimeError,ValueError):
                raise RuntimeError('wall fit failed; do not interpret failure as Walker breakdown')
    values=np.array(rows).reshape(len(h['time']),h['spins'].shape[1],4)
    phase=np.unwrap(values[...,2],axis=0); t=h['time'][:]
    return {'time':t.tolist(),'observables_order':['position','width','phase','max_abs_coincident_mz'],
            'observables':values.tolist(),
            'velocity':[float(np.polyfit(t,values[:,b,0],1)[0]) for b in range(values.shape[1])],
            'phase_frequency':[float(np.polyfit(t,phase[:,b],1)[0]) for b in range(values.shape[1])],
            'certified':False,'spatial_comparison':'Sublattices linearly interpolated to common long-axis coordinates; two endpoint cells omitted.'}


def main():
    args=analysis_arguments()
    with h5py.File(args.input) as h:
        require_complete(h); p=json.loads(h.attrs['protocol_json'])
        if all(p['periodic']):
            result=spectral_paths(h)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            with args.output.open('xb') as f: np.savez_compressed(f,**result)
        else: save_json(args.output,wall_observables(h))


if __name__=='__main__': main()

