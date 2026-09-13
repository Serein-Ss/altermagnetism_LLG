"""Float64 streaming with atomic checkpoints and trajectory-local RNG state.

Checkpoints are rollback cursors, not a seed restart. Events are postprocessed
from the retained full time grid, so no event accumulator is lost on resume.
Only one writer may own a path. A failed/incompatible checkpoint is preserved.
"""
import json
import os
from pathlib import Path
import signal
import h5py
import numpy as np
import torch


def simulate_resumable(path, s, llg, p, *, identity, step_fn=None,
                       observer=None, stop_after=None):
    path=Path(path); partial=Path(str(path)+'.partial')
    checkpoint=Path(str(path)+'.checkpoint.h5')
    dt=float(p['dt']); steps=int(p['steps']); stride=int(p['save_every'])
    checkpoint_every=int(p.get('checkpoint_every',10000))
    if dt<=0 or min(steps,stride,checkpoint_every)<1 or s.dtype!=torch.float64:
        raise ValueError('positive time settings and float64 state required')
    seeds=p.get('noise_seeds',[int(p['seed'])+i for i in range(len(s))])
    if len(seeds)!=len(s) or len(set(seeds))!=len(seeds):
        raise ValueError('one distinct noise seed per trajectory required')
    identity=json.dumps(dict(scope=identity,protocol=p,shape=list(s.shape),
        torch=torch.__version__,numpy=np.__version__,device=str(s.device),
        device_name=torch.cuda.get_device_name(s.device) if s.is_cuda else 'cpu',
        rng_policy='one_generator_per_trajectory_v1'),sort_keys=True)
    generators=[torch.Generator(device=s.device).manual_seed(int(seed)) for seed in seeds]
    times=np.unique(np.r_[np.arange(0,steps+1,stride,dtype=np.int64),steps])
    if path.exists():
        with h5py.File(path) as h:
            if h.attrs['identity']!=identity or not h.attrs['complete']:
                raise ValueError('existing final file identity/completion mismatch')
            return torch.tensor(h['spins'][-1],device=s.device),True
    frame=0; first_step=0; max_norm=0.; errors=np.zeros(2)
    if partial.exists():
        if not checkpoint.exists():
            raise RuntimeError('partial file without checkpoint: preserve and use a new run')
        with h5py.File(checkpoint) as c:
            if c.attrs['identity']!=identity: raise ValueError('checkpoint identity mismatch')
            s=torch.tensor(c['spins'][:],dtype=s.dtype,device=s.device)
            first_step=int(c.attrs['step']);frame=int(c.attrs['frame'])
            errors=c['errors'][:];max_norm=float(c.attrs['max_norm'])
            for i,g in enumerate(generators): g.set_state(torch.from_numpy(c[f'rng/{i}'][:]))
    elif checkpoint.exists():
        raise RuntimeError('checkpoint without partial data: do not restart silently')
    path.parent.mkdir(parents=True,exist_ok=True)
    stop=[False]
    def request_stop(signum,stack): stop[0]=True
    previous=signal.signal(signal.SIGUSR1,request_stop)
    try:
        with h5py.File(partial,'r+' if partial.exists() else 'x') as h:
            if 'spins' not in h:
                h.attrs.update(identity=identity,complete=False,dt=dt,steps=steps,
                    save_every=stride,method=p['method'],protocol_json=json.dumps(p),
                    event_history='deferred_full_grid_postprocessing_no_event_stopping',
                    diagnostic_semantics='solver_returned_errors; see error_labels')
                h.create_dataset('time',data=times*dt)
                h.create_dataset('integration_step',data=times)
                h.create_dataset('noise_seed',data=np.asarray(seeds,dtype=np.int64))
                h.create_dataset('initial_state',data=s.cpu().numpy())
                h.create_dataset('mask',data=np.ones((len(times),len(s)),dtype=bool))
                h.create_dataset('spins',shape=(len(times),*s.shape),dtype='f8',
                    chunks=(1,1,*[min(n,32) for n in s.shape[1:-1]],3),compression='lzf')
                h.create_dataset('energy',shape=(len(times),len(s)),dtype='f8')
                h.create_dataset('raw_norm_errors',shape=(len(times),2),dtype='f8')
                h.create_dataset('max_norm_error',shape=(len(times),),dtype='f8')
                labels=['predictor_preprojection','final_preprojection']
                if p['method']=='milstein_tretyakov_5_9': labels=['final_preprojection','final_postprojection']
                if step_fn: labels=['unavailable_custom_step','unavailable_custom_step']
                if p['method']=='geometric_midpoint': labels=['implicit_final_norm','implicit_final_norm']
                h.attrs['error_labels']=json.dumps(labels)
            elif h.attrs['identity']!=identity:
                raise ValueError('partial file identity mismatch')
            def save_checkpoint(step):
                h.attrs['written_frames']=frame;h.flush()
                temp=Path(str(checkpoint)+'.tmp')
                # Replacing only this run's checkpoint; raw history is retained.
                with h5py.File(temp,'w') as c:
                    c.attrs.update(identity=identity,step=step,frame=frame,max_norm=max_norm,
                                   physical_time=step*dt,event_history='recompute_from_retained_grid')
                    c.create_dataset('spins',data=s.cpu().numpy())
                    c.create_dataset('errors',data=errors)
                    for i,g in enumerate(generators): c.create_dataset(f'rng/{i}',data=g.get_state().cpu().numpy())
                    c.flush()
                os.replace(temp,checkpoint)
            with torch.no_grad():
                for step in range(first_step,steps+1):
                    if frame<len(times) and step==times[frame]:
                        h['spins'][frame]=s.cpu().numpy()
                        h['energy'][frame]=llg.model.energy(s).cpu().numpy()
                        h['raw_norm_errors'][frame]=errors
                        h['max_norm_error'][frame]=max_norm
                        if observer:
                            for name,value in observer(s).items():
                                value=value.cpu().numpy()
                                if name not in h: h.create_dataset(name,shape=(len(times),*value.shape),dtype='f8')
                                h[name][frame]=value
                        frame+=1
                    if step%checkpoint_every==0 or step==steps or stop[0] or (stop_after is not None and step==stop_after):
                        save_checkpoint(step)
                    if step==steps:
                        h.attrs['complete']=True;h.flush();break
                    if stop[0] or (stop_after is not None and step==stop_after): return s,False
                    dw=torch.cat([llg.noise_increment(s[i:i+1],dt,g) for i,g in enumerate(generators)],0) if llg.theta else torch.zeros_like(s)
                    if step_fn:
                        s=step_fn(s,step*dt,dt,dw);errors[:]=np.nan
                    else:
                        s,err=llg.step(s,dt,dw,method=p['method'])
                        errors=np.maximum(errors,err.cpu().numpy())
                    if not torch.isfinite(s).all(): raise FloatingPointError(f'nonfinite spins step {step+1}')
                    max_norm=max(max_norm,float((s.norm(dim=-1)-1).abs().max()))
        os.replace(partial,path)
        return s,True
    finally:
        signal.signal(signal.SIGUSR1,previous)
