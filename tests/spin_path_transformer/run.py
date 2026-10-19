"""Frozen, paired diagnostic of the new path backbone; no production promotion."""
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
from tests.bauer_p2_20260923.common import cfg, dump, generate
from tests.bauer_comprehensive_20260923.run import require_gpu, real_data, next_batch, references
from tests.cross_dimension_20260924.run import (rows_for, read_window, physical_config, build,
    parent_tensor, evaluate_2d, evaluate_toy)
from tests.cross_dimension_20260924.model import source
from tests.cross_dimension_20260924.oracle import targets, conditional, STOP
from .model import SpinPathTransformer

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def tasks():
    rows=[dict(domain=d,arm=a,seed=s,updates=6000,supervision='fm')
          for d in ('bauer','miltat') for s in (11,22,33) for a in ('local','latent','hierarchical')]
    rows += [dict(domain='toy',arm=a,seed=s,updates=12000,modes=2,supervision='fm')
             for s in (11,22,33) for a in ('local','latent','hierarchical','dense')]
    rows += [dict(domain='toy',arm='dense',seed=s,updates=12000,modes=2,supervision='exact') for s in (11,22,33)]
    rows += [dict(domain='toy',arm='hierarchical',seed=s,updates=12000,modes=1,supervision='fm') for s in (11,22,33)]
    np.random.default_rng(92488).shuffle(rows)
    return rows


def make_net(task,device='cuda'):
    torch.manual_seed(task['seed'])
    baseline=SpinPathTransformer(task['domain'],'local')
    torch.manual_seed(task['seed']+10000)
    net=SpinPathTransformer(task['domain'],task['arm'])
    # Identical shared tensors, despite extra modules changing constructor RNG use.
    common=dict(baseline.named_parameters())
    with torch.no_grad():
        for name,p in net.named_parameters():
            if name in common:p.copy_(common[name])
    return net.to(device)


@torch.no_grad()
def evaluate_bauer(net,out,benchmark=False):
    records=[];times=torch.arange(11,device='cuda')*.1
    with h5py.File(out/'paths.h5','x') as h:
        h.attrs['complete']=False
        for j,(row,raw) in enumerate(references('bauer','dev')):
            if benchmark and j:break
            levels=[16] if benchmark else ([16,64,128,256] if j==0 else [128])
            for steps in levels:
                n=2 if benchmark else 64
                initial=F.normalize(torch.tensor(raw[:1,0],device='cuda'),dim=-1).expand(n,-1,-1)
                g=torch.Generator(device='cuda').manual_seed(924990+j)
                z=source(initial,11,g,2.)
                torch.cuda.synchronize();start=time.perf_counter()
                x=generate(net,initial,times,g,steps,source=z).cpu().numpy()
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                group=h.create_group(f'parent{j:03d}_steps{steps}')
                group.attrs['parent']=json.dumps(row);group.attrs['steps']=steps
                group.create_dataset('generated',data=x,compression='lzf')
                group.create_dataset('reference',data=raw,compression='lzf')
                group.create_dataset('time_ps',data=np.arange(11)*.1)
                records.append(dict(parent=row,steps=steps,branches=n,seconds=elapsed))
        h.attrs['complete']=True
    return records


def run(task,out,benchmark=False):
    net=make_net(task);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-5)
    rng=np.random.default_rng(task['seed']);g=torch.Generator(device='cuda').manual_seed(5200+task['seed'])
    target_rng=torch.Generator(device='cuda').manual_seed(5300+task['seed'])
    times=torch.arange(11,device='cuda')*.1;cell=None;config=cfg()
    if task['domain']=='bauer':data=real_data(config)
    elif task['domain']=='miltat':
        data=rows_for('train');cell,_,_=build(physical_config(),'miltat',shape=(64,38))
    else:data=parent_tensor('train')
    history=[];seconds=[];stream=hashlib.sha256()
    for step in range(1,(8 if benchmark else task['updates'])+1):
        start=time.perf_counter()
        if task['domain']=='bauer':y,times=next_batch(task,config,data,rng,target_rng,'cuda')
        elif task['domain']=='miltat':
            row=data[int(rng.integers(len(data)))];window=0 if rng.random()<.5 else int(rng.integers(91))
            raw,_=read_window(row,int(rng.integers(8)),window)
            y=F.normalize(torch.tensor(raw[None],device='cuda'),dim=-1)
        else:
            ids=torch.randint(len(data),(8,),device='cuda',generator=target_rng)
            ys=targets(data[ids],task['modes']);choice=torch.randint(task['modes'],(8,),device='cuda',generator=target_rng)
            y=ys[torch.arange(8,device='cuda'),choice]
        z=source(y[:,0],11,g,net.source_shift)
        tau=torch.rand(len(y),device='cuda',generator=g)*(STOP if task['domain']=='toy' else 1.)
        x,target,cut=path_interpolate(z,y,tau,g)
        if step<=8:
            for a in (y,z,tau):stream.update(a.detach().cpu().numpy().tobytes())
        if task['supervision']=='exact':
            with torch.no_grad():target,_,bad=conditional(x,tau,targets(y[:,0],task['modes']))
            if bad.any():raise RuntimeError('outside exact support')
        pred=net.velocity(x,tau,y[:,0],times) if cell is None else net(x,tau,y[:,0],cell,torch.eye(3,device='cuda'),0.,times)
        loss=(pred[:,1:]-target[:,1:]).square().sum(-1).mean()
        opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
        if not torch.isfinite(loss) or not torch.isfinite(grad):raise RuntimeError('nonfinite optimizer state')
        opt.step();torch.cuda.synchronize();seconds.append(time.perf_counter()-start)
        if step%500==0:
            history.append(dict(step=step,loss=float(loss),gradient=float(grad),**cut))
            dump(out/'history.json',history);print(task,history[-1],flush=True)
        if not benchmark and step in (task['updates']//2,task['updates']):
            torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),task=task,step=step,
                source_rng=g.get_state(),target_rng=target_rng.get_state(),data_rng=rng.bit_generator.state),out/f'step{step}.pt')
    train_seconds=sum(seconds);net.eval()
    if task['domain']=='miltat':evaluation=evaluate_2d(net,out,benchmark)
    elif task['domain']=='bauer':evaluation=evaluate_bauer(net,out,benchmark)
    else:evaluation=evaluate_toy(net,task,out,benchmark)
    return dict(task=task,metadata=net.get_extra_state(),history=history,evaluation=evaluation,
        warm_step_seconds=float(np.median(seconds[2:])),train_seconds=train_seconds,
        parameters=sum(p.numel() for p in net.parameters()),peak_bytes=torch.cuda.max_memory_allocated(),
        device=torch.cuda.get_device_name(),first8_input_hash=stream.hexdigest(),scientific_certification=False)


def prepare():
    if (HERE/'submission.json').exists():raise RuntimeError('already submitted')
    protocol=dict(tasks=tasks(),purpose='Test local temporal / latent / regional communication, not certify universal magnetism',
        architecture='width64 depth4, per-token LayerNorm, adaLN-zero gates, fixed registered crystal frame; one fine/coarse level only',
        physical_grid='1 ps block, 0.1 ps saved frames; original reference dt unchanged',
        data='Existing parent/source splits only; no new data, no clean future conditions; development not untouched test',
        pairing='Shared tensor initialization, separate identical data/source RNG per seed; scheduling shuffled seed92488',
        optimizer='AdamW lr0.001 weight_decay1e-5 clip10; FM-only except explicit synthetic oracle control',
        budget='6000 physical /12000 synthetic updates; equal updates NOT equal compute; fixed final checkpoint primary',
        physical_scope='Bauer open L16/25/32 at63.824849668K; Miltat open64x38x2 at12.5K, larger diagnostics not certification',
        estimand='Conditional path statistics per independent dev parent; whole branches are replicates, not frames/sites',
        limitations='No new temperature, no 3D physical training, no arbitrary lattice registration, no approved event gates, no long-time claim',
        decision='Need physical joint-statistical improvement, not merely lower FM loss or legal spins; compare each seed separately',
        historical_comparison='Old backbone is descriptive only: new depth and optimizer geometry differ; concurrent local is the causal architecture control')
    dump(HERE/'protocol.json',protocol)
    lock=json.loads((ROOT/'tests/cross_dimension_20260924/source_lock.json').read_text())
    for name,digest in lock.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(name)
    for p in list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json',HERE/'literature.json']:
        lock[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','benchmark'])
    p.add_argument('--index',type=int,default=0);p.add_argument('--device',default='cuda');a=p.parse_args()
    if a.action=='prepare':prepare();return
    require_gpu(a.device)
    task=tasks()[a.index]
    if a.action=='run':
        for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(name)
    out=HERE/f'{"benchmark" if a.action=="benchmark" else "runs"}/{a.index:02d}'
    out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    result=run(task,out,a.action=='benchmark');result['total_seconds']=time.perf_counter()-start
    dump(out/'complete.json',result)


if __name__=='__main__':main()
