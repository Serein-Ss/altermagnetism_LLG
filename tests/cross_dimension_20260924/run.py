"""2D graph training and analytic-mode isolation; all artifacts remain test-only."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from scripts.model.sphere import geodesic_interpolate
from tests.bauer_p2_20260923.common import dump
from tests.bauer_comprehensive_20260923.run import require_gpu
from tests.bauer_comprehensive_20260923.components import synthetic_parents
from tests.bauer_repair_20260924.run import mixture_samples
from tests.literature_path_campaign_20260920.train_miltat import physical_config
from tests.literature_preflight_20260917.models import build
from tests.cross_system_audit_20260920.miltat import DATA, read_window
from .model import GeometryGraph, source, sample
from .oracle import targets, conditional, integrate, make_net, STOP

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def tasks():
    rows=[dict(domain='miltat',arm=a,seed=s,updates=6000) for s in (11,22,33)
          for a in ('shifted','uniform','global')]
    rows += [dict(domain='toy',arm=a,seed=s,modes=m,updates=12000)
             for s in (11,22,33) for m in (1,2) for a in ('fm','global','oracle_supervision')]
    np.random.default_rng(92441).shuffle(rows)
    return rows


def rows_for(split,shape=(64,38)):
    return [r for r in json.loads((DATA/'manifest.json').read_text())['parents']
            if r['split']==split and (shape is None or tuple(r['cell_shape'])==shape)]


def train_update(net,opt,y,times,g,cell=None,task=None):
    shift=0. if cell is None or net.arm=='uniform' else 2.
    z=source(y[:,0],len(times),g,shift)
    tau=torch.rand(len(y),device=y.device,generator=g)*(STOP if cell is None else 1.)
    x,target,cut=path_interpolate(z,y,tau,g)
    if cell is None:
        if task['arm']=='oracle_supervision':
            with torch.no_grad():target,_,bad=conditional(x,tau,targets(y[:,0],task['modes']))
            if bad.any():raise RuntimeError('training sample outside analytic support')
        pred=net.velocity(x,tau,y[:,0],times)
    else:pred=net(x,tau,y[:,0],cell,torch.eye(3,device=y.device),0.,times)
    loss=(pred[:,1:]-target[:,1:]).square().sum(-1).mean()
    opt.zero_grad(set_to_none=True);loss.backward()
    grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
    if not torch.isfinite(loss) or not torch.isfinite(grad):raise RuntimeError('nonfinite optimization')
    opt.step()
    return dict(loss=float(loss),gradient=float(grad),**cut)


@torch.no_grad()
def evaluate_2d(net,out,benchmark=False):
    rows=rows_for('dev') if not benchmark else rows_for('dev')[:1]
    if not benchmark:
        rows += rows_for('train')[:1]
        rows += rows_for('dev',(96,56))[:1]+rows_for('dev',(128,74))[:1]
    records=[];times=torch.arange(11,device='cuda')*.1
    with h5py.File(out/'paths.h5','x') as h:
        h.attrs['complete']=False
        for j,row in enumerate(rows):
            cell,_,_=build(physical_config(),'miltat',shape=row['cell_shape'])
            with h5py.File(DATA/row['file']) as ref:
                raw=ref['spins'][:,:11];pos=ref['positions_nm'][:]
            initial=F.normalize(torch.tensor(raw[:1,0],device='cuda'),dim=-1)
            is_small=tuple(row['cell_shape'])==(64,38)
            n=2 if benchmark else (16 if is_small else 8)
            levels=[16] if benchmark else ([16,64,128,256] if j==0 else [128])
            for steps in levels:
                generated=[];start=time.perf_counter()
                for branch in range(n):
                    g=torch.Generator(device='cuda').manual_seed(924900+j*100+branch)
                    generated.append(sample(net,initial,cell,times,steps,g).cpu().numpy()[0])
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                x=np.stack(generated);group=h.create_group(f'{row["parent_id"]}_steps{steps}')
                group.attrs['split']=row['split'];group.attrs['steps']=steps
                group.attrs['scope']='same_size' if is_small else 'exploratory_size_transfer'
                for name,value in [('generated',x),('reference',raw),('positions_nm',pos),('time_ps',np.arange(11)*.1)]:
                    group.create_dataset(name,data=value,compression='lzf')
                records.append(dict(parent_id=row['parent_id'],steps=steps,branches=n,seconds=elapsed,
                    shape=row['cell_shape'],split=row['split'],scope=group.attrs['scope']))
        h.attrs['complete']=True
    return records


def run_2d(task,out,benchmark=False):
    torch.manual_seed(task['seed']);rng=np.random.default_rng(task['seed'])
    g=torch.Generator(device='cuda').manual_seed(5000+task['seed'])
    net=GeometryGraph(task['arm']).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-5)
    cell,_,_=build(physical_config(),'miltat',shape=(64,38));rows=rows_for('train')
    assert len(rows)==12
    times=torch.arange(11,device='cuda')*.1;history=[];seconds=[]
    for step in range(1,(8 if benchmark else task['updates'])+1):
        start=time.perf_counter();row=rows[int(rng.integers(len(rows)))]
        window=0 if rng.random()<.5 else int(rng.integers(91))
        raw,_=read_window(row,int(rng.integers(8)),window)
        y=F.normalize(torch.tensor(raw[None],device='cuda'),dim=-1)
        result=train_update(net,opt,y,times,g,cell=cell)
        torch.cuda.synchronize();seconds.append(time.perf_counter()-start)
        if step%100==0:print(task,step,result,flush=True)
        if step%500==0:
            history.append(dict(step=step,**result));dump(out/'history.json',history)
        if not benchmark and step in (1500,3000,6000):
            torch.save(dict(model=net.state_dict(),task=task,step=step),out/f'step{step}.pt')
    net.eval();records=evaluate_2d(net,out,benchmark)
    return dict(task=task,device=torch.cuda.get_device_name(),warm_step_seconds=float(np.median(seconds[2:])),
                peak_bytes=torch.cuda.max_memory_allocated(),generation=records,history=history,
                parameters=sum(p.numel() for p in net.parameters()),scientific_certification=False)


def parent_tensor(split):
    return synthetic_parents(split).float().cuda()


@torch.no_grad()
def evaluate_toy(net,task,out,benchmark=False):
    parents=parent_tensor('dev')[:1 if benchmark else 4];rows=[]
    times=torch.linspace(0,1,11,device='cuda')
    with h5py.File(out/'paths.h5','x') as h:
        h.attrs['complete']=False
        for j,parent in enumerate(parents):
            initial=parent[None].expand(16 if benchmark else 64,-1,-1)
            ys=targets(initial,task['modes'])
            g=torch.Generator(device='cuda').manual_seed(93000+j)
            chosen=torch.randint(task['modes'],(len(initial),),device='cuda',generator=g)
            z=source(initial,11,g,0.)
            reference,_=geodesic_interpolate(z,ys[torch.arange(len(initial),device='cuda'),chosen],z.new_full((len(z),),STOP))
            # Conditional-velocity validation on fresh interpolants, not raw pair-target MSE.
            z=source(initial,11,g,0.);tau=torch.rand(len(z),device='cuda',generator=g)*STOP
            x,_=geodesic_interpolate(z,ys[torch.arange(len(z),device='cuda'),chosen],tau)
            truth,_,bad=conditional(x,tau,ys)
            prediction=net.velocity(x,tau,initial,times)
            velocity_error=float((prediction[:,1:]-truth[:,1:]).square().sum(-1).mean())
            levels=[16] if benchmark else ([16,64,128,512] if j==0 else [128])
            for steps in levels:
                x,diagnostic=integrate(initial,ys,steps,torch.Generator(device='cuda').manual_seed(94000+j),net)
                group=h.create_group(f'parent{j}_steps{steps}')
                for k,val in [('generated',x.cpu().numpy()),('reference',reference.cpu().numpy()),('time',times.cpu().numpy())]:
                    group.create_dataset(k,data=val,compression='lzf')
                row=dict(parent=j,steps=steps,conditional_velocity_mse=velocity_error,**diagnostic)
                for label,val in [('generated',x.cpu().numpy()),('reference',reference.cpu().numpy())]:
                    row[label]={k:float(v.mean()) for k,v in mixture_samples(val).items()}
                rows.append(row)
        h.attrs['complete']=True
    return rows


def run_toy(task,out,benchmark=False):
    torch.manual_seed(task['seed']);g=torch.Generator(device='cuda').manual_seed(5100+task['seed'])
    parents=parent_tensor('train');net=make_net(task['arm'],'cuda')
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-5)
    times=torch.linspace(0,1,11,device='cuda');history=[];seconds=[]
    for step in range(1,(8 if benchmark else task['updates'])+1):
        start=time.perf_counter();ix=torch.randint(len(parents),(8,),device='cuda',generator=g)
        ys=targets(parents[ix],task['modes']);choice=torch.randint(task['modes'],(8,),device='cuda',generator=g)
        y=ys[torch.arange(8,device='cuda'),choice]
        result=train_update(net,opt,y,times,g,task=task)
        torch.cuda.synchronize();seconds.append(time.perf_counter()-start)
        if step%1000==0:
            history.append(dict(step=step,**result));dump(out/'history.json',history);print(task,step,result,flush=True)
    torch.save(dict(model=net.state_dict(),task=task,source='uniform_sphere',transport_stop=STOP),out/'last.pt')
    net.eval();rows=evaluate_toy(net,task,out,benchmark)
    return dict(task=task,rows=rows,history=history,warm_step_seconds=float(np.median(seconds[2:])),
                device=torch.cuda.get_device_name(),scientific_certification=False)


def oracle_reference(out):
    rows=[];initial=parent_tensor('dev')[0:1].expand(256,-1,-1)
    with h5py.File(out/'oracle.h5','x') as h:
        for modes in (1,2):
            ys=targets(initial,modes)
            for steps in (16,64,256,1024,4096):
                start=time.perf_counter()
                x,d=integrate(initial,ys,steps,torch.Generator(device='cuda').manual_seed(92831))
                features={k:float(v.mean()) for k,v in mixture_samples(x.cpu().numpy()).items()}
                h.create_dataset(f'modes{modes}_steps{steps}',data=x.cpu().numpy(),compression='lzf')
                rows.append(dict(modes=modes,steps=steps,seconds=time.perf_counter()-start,**d,**features))
    dump(out/'complete.json',dict(rows=rows,scope='analytic transport only; not magnetic dynamics'))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','benchmark','oracle']);p.add_argument('--device',choices=['cuda'],required=True)
    p.add_argument('--index',type=int,default=0);args=p.parse_args();require_gpu(args.device);torch.set_num_threads(1)
    if args.action=='oracle':
        out=HERE/'oracle_reference';out.mkdir(exist_ok=False);oracle_reference(out);return
    task=tasks()[args.index]
    if args.action=='run':
        for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('frozen source changed: '+name)
    if args.action=='benchmark':task=dict(domain='miltat',arm='global',seed=11,updates=8)
    out=HERE/('benchmark' if args.action=='benchmark' else f'runs/{args.index:02d}')
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    result=(run_2d if task['domain']=='miltat' else run_toy)(task,out,args.action=='benchmark')
    result['total_seconds']=time.perf_counter()-started
    dump(out/'complete.json',result)


if __name__=='__main__':main()
