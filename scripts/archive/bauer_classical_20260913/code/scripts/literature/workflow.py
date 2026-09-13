"""Shared file contract for reduced literature simulations (not a job launcher)."""
import argparse
import json
import os
from pathlib import Path
import h5py
import numpy as np
import torch
from scripts.core.literature_config import read_document, load_runtime, sha256

ROOT=Path(__file__).resolve().parents[2]


def arguments(paper, modes):
    parser=argparse.ArgumentParser(description=f'Reduced {paper} simulation')
    parser.add_argument('--config',type=Path,default=ROOT/'conf/literature'/f'{paper}.yaml')
    parser.add_argument('--protocol',required=True,choices=modes)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--validation',action='store_true',help='Run an explicitly configured numerical validation, never certify production')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    return parser.parse_args()


def prepare(args,paper):
    if Path(args.run_id).name != args.run_id or args.run_id in ('.','..'):
        raise ValueError('run-id must be one directory name')
    if args.device=='cuda' and (not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available()):
        raise RuntimeError('CUDA requires an active Slurm GPU allocation')
    document=read_document(args.config)
    if document['paper_id']!=paper:
        raise ValueError('wrong paper configuration')
    runtime=load_runtime(args.config)
    protocol=runtime.numerics['protocols'][args.protocol]
    validation=getattr(args,'validation',False)
    if validation and args.protocol not in runtime.numerics['validation_extension'].get('validation_protocols',[]):
        raise RuntimeError('Validation protocol must be explicitly listed in its frozen configuration')
    if not validation and args.protocol not in ('smoke','weak_rk_smoke','llb_smoke','profile','branch') and not runtime.numerics['validation_extension']['production_enabled']:
        raise RuntimeError('Production disabled: complete numerical/reference gates before enabling this protocol in a frozen YAML')
    output_root=getattr(args,'project_root',ROOT).resolve()
    run_name=args.run_id+'.partial' if validation else args.run_id
    raw=output_root/'data/literature_reproduction'/paper/'raw'/run_name
    if validation and raw.with_name(args.run_id).exists():
        raise FileExistsError(raw.with_name(args.run_id))
    resuming=bool(protocol.get('resumable',False))
    raw.mkdir(parents=True,exist_ok=resuming)
    out=output_root/'output/literature_reproduction'/paper/args.run_id
    out.mkdir(parents=True,exist_ok=resuming)
    if (out/'config.yaml').exists() and (out/'config.yaml').read_bytes()!=args.config.read_bytes():
        raise ValueError('resuming configuration mismatch')
    (out/'config.yaml').write_bytes(args.config.read_bytes())
    manifest={'paper_id':paper,'protocol':args.protocol,'configuration':document,
              'config_sha256':sha256(args.config),'device':args.device,
              'stage':'numerical_validation' if validation else args.protocol,
              'run_id':args.run_id,'status':'running','claim':'numerical_experiment_not_literature_certification',
              'code_sha256':{str(p.relative_to(ROOT)):sha256(p) for folder in
                             (ROOT/'scripts/core',ROOT/'scripts/literature')
                             for p in folder.rglob('*.py')},
              'versions':{'torch':torch.__version__,'numpy':np.__version__,'h5py':h5py.__version__}}
    if resuming:
        protocol=dict(protocol,resume_identity=dict(config=manifest['config_sha256'],code=manifest['code_sha256']))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return runtime.reduced,protocol,raw,out,manifest


def complete(raw,out,manifest):
    if manifest.get('stage')=='numerical_validation':
        raw.rename(raw.with_name(manifest['run_id']))
        raw=raw.with_name(manifest['run_id'])
    manifest['status']='simulation_complete_analysis_not_certified'
    manifest['files_sha256']={p.name:sha256(p) for p in raw.iterdir() if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))


def simulate(path,s,llg,p,*,step_fn=None,observer=None):
    """Stream EVERY trajectory, initial/final states and all scheduled frames.

    No success filtering; exclusive creation prevents overwriting old runs.
    If interrupted, complete=False distinguishes the partial HDF5 file.
    Temporal decimation is recorded explicitly; it is not integration output
    at every step. Raw states stay float64 to permit independent diagnostics.
    """
    if p.get('resumable',False):
        from scripts.core.streaming_llg import simulate_resumable
        state,done=simulate_resumable(path,s,llg,p,identity=p['resume_identity'],step_fn=step_fn,observer=observer)
        if not done:
            raise SystemExit(85)  # saved checkpoint; not scientific completion
        return state
    dt=float(p['dt']); steps=int(p['steps']); stride=int(p['save_every'])
    if dt<=0 or steps<1 or stride<1:
        raise ValueError('positive dt, steps and save_every required')
    generator=torch.Generator(device=s.device).manual_seed(int(p['seed']))
    times=list(range(0,steps+1,stride))
    if times[-1]!=steps: times.append(steps)
    with h5py.File(path,'x') as h:
        h.attrs.update(complete=False,dt=dt,steps=steps,save_every=stride,
                       seed=int(p['seed']),method=p['method'],protocol_json=json.dumps(p))
        h.create_dataset('time',data=np.array(times)*dt)
        chunk=(1,1,*[min(n,32) for n in s.shape[1:-1]],3)
        states=h.create_dataset('spins',shape=(len(times),*s.shape),dtype='f8',
                                chunks=chunk,compression='lzf')
        energy=h.create_dataset('energy',shape=(len(times),s.shape[0]),dtype='f8')
        norm=h.create_dataset('max_norm_error',shape=(len(times),),dtype='f8')
        raw_errors=h.create_dataset('raw_norm_errors',shape=(len(times),2),dtype='f8')
        labels=['predictor_preprojection','final_preprojection']
        if p['method']=='milstein_tretyakov_5_9': labels=['final_preprojection','final_postprojection']
        if step_fn: labels=['unavailable_custom_step','unavailable_custom_step']
        if p['method']=='geometric_midpoint': labels=['implicit_final_norm','implicit_final_norm']
        h.attrs['error_labels']=json.dumps(labels)
        obs={}; frame=0; max_error=0.; pre_errors=np.zeros(2)
        with torch.no_grad():
            for step in range(steps+1):
                if step==times[frame]:
                    states[frame]=s.cpu().numpy()
                    energy[frame]=llg.model.energy(s).cpu().numpy()
                    norm[frame]=max_error
                    raw_errors[frame]=pre_errors
                    if observer:
                        for name,val in observer(s).items():
                            val=val.cpu().numpy()
                            if name not in obs:
                                obs[name]=h.create_dataset(name,shape=(len(times),*val.shape),dtype='f8')
                            obs[name][frame]=val
                    h.attrs['written_frames']=frame+1
                    h.flush(); frame+=1
                if step==steps: break
                dw=llg.noise_increment(s,dt,generator) if llg.theta else torch.zeros_like(s)
                if step_fn:
                    s=step_fn(s,step*dt,dt,dw)
                    pre_errors[:]=np.nan
                else:
                    s,err=llg.step(s,dt,dw,method=p['method'])
                    pre_errors=np.maximum(pre_errors,err.cpu().numpy())
                if not torch.isfinite(s).all():
                    raise FloatingPointError(f'nonfinite state at step {step+1}')
                max_error=max(max_error,float((s.norm(dim=-1)-1).abs().max()))
        h.attrs['complete']=True
    return s


def rk4_step(llg,s,t,dt,field_at=None):
    """Projected deterministic RK4, timed forcing evaluated at RK stages."""
    if llg.theta:
        raise ValueError('RK4 is deterministic, not a stochastic RK integrator')
    def rhs(x,time):
        b=llg.model.field(x)
        if field_at is not None: b=b+field_at(time)
        cross=torch.linalg.cross
        return -(cross(x,b)+llg.alpha*cross(x,cross(x,b)))/(1+llg.alpha**2)
    a=rhs(s,t); b=rhs(s+dt*a/2,t+dt/2); c=rhs(s+dt*b/2,t+dt/2); d=rhs(s+dt*c,t+dt)
    raw=s+dt*(a+2*b+2*c+d)/6
    return raw/raw.norm(dim=-1,keepdim=True)


def analysis_arguments():
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    return parser.parse_args()


def save_json(path,result):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)


def require_complete(h):
    if not h.attrs.get('complete',False):
        raise ValueError('incomplete trajectory cannot enter formal analysis')


def plot_trajectory(input_path,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with h5py.File(input_path) as h:
        require_complete(h)
        t=h['time'][:]; mean=[]; spread=[]
        for frame in h['spins']:
            flat=frame.reshape(frame.shape[0],-1,3)
            mean.append(flat.mean(1)); spread.append(flat.std(1))
        mean=np.array(mean); spread=np.array(spread)
        fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
        for i,label in enumerate(('x','y','z')):
            axes[0].plot(t,mean[:,:,i],alpha=.6,label=label if mean.shape[1]==1 else None)
            axes[1].plot(t,spread[:,:,i],alpha=.6)
        axes[0].set(xlabel='Reduced time',ylabel='Spatial mean spin')
        axes[1].set(xlabel='Reduced time',ylabel='Spatial standard deviation')
        axes[2].plot(t,h['energy'][:]); axes[2].set(xlabel='Reduced time',ylabel='Reduced total energy')
        output.parent.mkdir(parents=True,exist_ok=True)
        if output.exists(): raise FileExistsError(output)
        fig.savefig(output,dpi=160); plt.close(fig)
