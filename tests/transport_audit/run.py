"""Read-only checkpoint diagnosis. No retraining or production sampler change."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from scripts.model.sphere import sphere_exp
from scripts.model.gate_f import path_interpolate
from tests.spin_path_transformer.model import SpinPathTransformer
from tests.spin_path_transformer.run import tasks as original_tasks
from tests.cross_dimension_20260924.model import source
from tests.cross_dimension_20260924.oracle import targets, conditional, STOP
from tests.bauer_comprehensive_20260923.components import synthetic_parents
from tests.bauer_comprehensive_20260923.run import references, require_gpu, diagnostic_comparison
from tests.bauer_p2_20260923.common import dump, comparison
from tests.bauer_repair_20260924.run import mixture_samples

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'tests/spin_path_transformer'


def tasks():
    return [dict(old_index=i,**t) for i,t in enumerate(original_tasks())
            if (t['domain']=='bauer' and t['arm']=='hierarchical') or
            (t['domain']=='toy' and t['modes']==2 and
             (t['arm']=='hierarchical' or t['supervision']=='exact'))]


@torch.no_grad()
def integrate(field,x,start,stop,steps,method):
    if not 0<=start<stop<=1 or steps<1:raise ValueError('transport range')
    if method not in ('euler','midpoint'):raise ValueError(method)
    x=x.clone();initial=x[:,0].clone();dt=(stop-start)/steps
    for k in range(steps):
        tau=x.new_full((len(x),),start+k*dt);v=field(x,tau)
        if method=='euler':x=sphere_exp(x,dt*v)
        else:
            mid=F.normalize(x+.5*dt*v,dim=-1);mid[:,0]=initial
            x=F.normalize(x+dt*field(mid,tau+.5*dt),dim=-1)
        x[:,0]=initial
    return x


def wilson(k,n):
    p=k/n;z=1.95996398454;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(center-half),float(center+half)]


def summarize_toy(x):
    vals=mixture_samples(x.cpu().numpy());k=int(vals['positive_fraction'].sum())
    return dict(**{key:float(v.mean()) for key,v in vals.items()},n=len(x),positive_count=k,
                positive_wilson95=wilson(k,len(x)),target_probability=.5)


def load(task,step):
    net=SpinPathTransformer(task['domain'],task['arm']).cuda()
    path=OLD/f'runs/{task["old_index"]:02d}/step{step}.pt'
    checkpoint=torch.load(path,map_location='cuda',weights_only=True)
    net.load_state_dict(checkpoint['model']);net.eval()
    return net,hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def toy(task,out,smoke):
    net,digest=load(task,12000);times=torch.arange(11,device='cuda')*.1
    parents=synthetic_parents('dev').float().cuda()[:1 if smoke else 4]
    rows=[];errors=[];n=16 if smoke else 256
    with h5py.File(out/'paths.h5','x') as h:
        for j,parent in enumerate(parents):
            initial=parent[None].expand(n,-1,-1);ys=targets(initial,2)
            gen=torch.Generator(device='cuda').manual_seed(925100+j)
            z=source(initial,11,gen,0.)
            labels=torch.randint(2,(n,),device='cuda',generator=gen)
            y=ys[torch.arange(n,device='cuda'),labels]
            field=lambda x,tau:net.velocity(x,tau,initial,times)
            # Teacher-distribution errors by transport time; not physical time.
            for tau in (0.,.01,.03,.1,.25,.5,.75,.9,.94):
                x,_,_=path_interpolate(z,y,z.new_full((n,),tau),gen)
                truth,weights,bad=conditional(x,x.new_full((n,),tau),ys)
                pred=field(x,x.new_full((n,),tau));err=(pred[:,1:]-truth[:,1:]).square().sum(-1)
                errors.append(dict(parent=j,tau=tau,mse=float(err.mean()),
                    error_vector_mean_norm=float((pred[:,1:]-truth[:,1:]).mean((0,1,2)).norm()),
                    unsupported=int(bad.sum()),posterior_positive=float(weights[:,-1].mean())))
            configs=[(0.,16,'euler')] if smoke else [(0.,128,'euler'),(0.,512,'euler'),(0.,256,'midpoint')]
            if not smoke and j==0:configs += [(v,128,'euler') for v in (.03,.1,.25,.5,.75)]
            for start,steps,method in configs:
                x=z if start==0 else path_interpolate(z,y,z.new_full((n,),start),gen)[0]
                torch.cuda.synchronize();began=time.perf_counter()
                chunks=[]
                for b in range(0,n,64):
                    init=initial[b:b+64]
                    chunks.append(integrate(lambda v,t:net.velocity(v,t,init,times),x[b:b+64],start,STOP,steps,method))
                x=torch.cat(chunks);torch.cuda.synchronize()
                name=f'parent{j}_start{start}_steps{steps}_{method}'
                h.create_dataset(name,data=x.cpu().numpy(),compression='lzf')
                rows.append(dict(parent=j,start_tau=start,steps=steps,method=method,
                    velocity_evaluations=steps*(2 if method=='midpoint' else 1),seconds=time.perf_counter()-began,
                    reference_informed_restart=start>0,**summarize_toy(x)))
        h.attrs['complete']=True
    return dict(task=task,checkpoint_sha256=digest,rows=rows,velocity_errors=errors,
        warning='Intermediate-law restarts use known targets for diagnosis, not a deployable generator. Analytic target probability is 0.5, not finite-reference frequency.')


@torch.no_grad()
def bauer(task,out,smoke):
    rows=[];times=torch.arange(11,device='cuda')*.1
    all_refs=list(references('bauer','dev'));selected=[];seen=set()
    for j,(row,raw) in enumerate(all_refs):
        if row['L']!=16 or row['family'] in seen:continue
        seen.add(row['family']);selected.append((j,row,raw))
    if smoke:selected=selected[:1]
    with h5py.File(out/'paths.h5','x') as h:
        for checkpoint_step in ([6000] if smoke else [3000,6000]):
            net,digest=load(task,checkpoint_step)
            for j,row,raw in selected:
                n=2 if smoke else 128
                initial=F.normalize(torch.tensor(raw[:1,0],device='cuda'),dim=-1).expand(n,-1,-1)
                gen=torch.Generator(device='cuda').manual_seed(925200+j);z=source(initial,11,gen,2.)
                for steps,method in ([(16,'euler')] if smoke else [(128,'euler'),(512,'euler'),(256,'midpoint')]):
                    torch.cuda.synchronize();began=time.perf_counter();chunks=[]
                    for b in range(0,n,64):
                        init=initial[b:b+64]
                        chunks.append(integrate(lambda v,t:net.velocity(v,t,init,times),z[b:b+64],0.,1.,steps,method))
                    x=torch.cat(chunks).cpu().numpy();torch.cuda.synchronize();seconds=time.perf_counter()-began
                    name=f'checkpoint{checkpoint_step}_parent{j}_steps{steps}_{method}'
                    group=h.create_group(name);group.create_dataset('generated',data=x,compression='lzf')
                    group.create_dataset('reference',data=raw,compression='lzf')
                    rows.append(dict(checkpoint_step=checkpoint_step,checkpoint_sha256=digest,parent_index=j,
                        family=row['family'],L=16,steps=steps,method=method,seconds=seconds,
                        velocity_evaluations=steps*(2 if method=='midpoint' else 1),generated_branches=n,reference_branches=len(raw),
                        physical=comparison(x,raw,0 if smoke else 100,92520+j),
                        diagnostic=diagnostic_comparison(x,raw,0 if smoke else 100,92530+j)))
        h.attrs['complete']=True
    return dict(task=task,rows=rows,scope='L16 dev first parent per preparation family, 1ps/0.1ps unchanged, no new references')


def prepare():
    if (HERE/'submission.json').exists():raise RuntimeError('already submitted')
    lock=json.loads((OLD/'source_lock.json').read_text())
    for name,digest in lock.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(name)
    protocol=dict(tasks=tasks(),training='none; frozen checkpoints only',
        toy='256 fresh sources per each of first4 fixed dev parents; theory probability0.5; Wilson intervals per parent. Restart only first parent, inference-only diagnostic using true mixture.',
        bauer='128 fresh generated branches, 16 references, L16 first parent per preparation family. Half/final checkpoints and same sources for solvers.',
        numerical='Euler128/512 versus normalized ambient midpoint256 (512 network calls); actual physical time unchanged',
        claims='Exploratory attribution, no equivalence thresholds, no event gates, no scientific certification',
        decisions=['Bias unchanged with independent larger source draws supports model bias, not finite reference imbalance',
                   'Restart-time sensitivity localizes transport accumulation but is not an inference improvement',
                   'Solver disagreement requires numerical investigation before architecture change',
                   'Checkpoint change distinguishes progress/metric trade-offs; no new best-checkpoint claim'])
    dump(HERE/'protocol.json',protocol)
    for p in list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json']:
        lock[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','smoke']);p.add_argument('--index',type=int,default=0)
    p.add_argument('--device',default='cuda');a=p.parse_args()
    if a.action=='prepare':prepare();return
    require_gpu(a.device)
    if a.action=='run':
        for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(name)
    out=HERE/f'{a.action}/{a.index:02d}';out.mkdir(parents=True,exist_ok=False)
    task=tasks()[a.index];began=time.perf_counter()
    result=(toy if task['domain']=='toy' else bauer)(task,out,a.action=='smoke')
    result.update(seconds=time.perf_counter()-began,device=torch.cuda.get_device_name(),scientific_certification=False)
    dump(out/'complete.json',result)


if __name__=='__main__':main()
