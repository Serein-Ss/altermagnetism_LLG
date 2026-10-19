"""Controlled training repair; unchanged backbone and reference physics."""
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
from tests.spin_path_transformer.run import make_net,evaluate_bauer
from tests.bauer_p2_20260923.common import cfg,dump
from tests.bauer_comprehensive_20260923.run import real_data,next_batch,require_gpu
from tests.cross_dimension_20260924.run import parent_tensor
from tests.cross_dimension_20260924.model import source
from tests.cross_dimension_20260924.oracle import targets,integrate,conditional,STOP
from tests.transport_audit.run import summarize_toy
from .objectives import draw_tau,differentiable_sample,joint_loss

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def tasks():
    rows=[dict(domain=d,arm='hierarchical',repair=a,seed=s,updates=6000 if d=='bauer' else 12000,modes=2)
          for d,arms in [('bauer',('baseline','early','joint','early_joint')),('toy',('baseline','importance','early'))]
          for s in (11,22,33) for a in arms]
    np.random.default_rng(92551).shuffle(rows)
    return rows


def training_anchors():
    rows=json.loads((ROOT/'tests/bauer_p2_20260923/manifest_train.json').read_text())
    dev=json.loads((ROOT/'tests/bauer_p2_20260923/manifest_dev.json').read_text())
    assert not {r['parent_id'] for r in rows}&{r['parent_id'] for r in dev}
    arrays=[]
    for row in rows:
        assert row['split']=='train'
        with h5py.File(ROOT/row['path']) as h:raw=h['spins'][:,:11].astype('f4')
        assert np.max(abs(raw[:,0]-raw[:1,0]))<1e-6
        arrays.append(F.normalize(torch.from_numpy(raw),dim=-1))
    return arrays


@torch.no_grad()
def evaluate_toy(net,out,smoke):
    rows=[];times=torch.arange(11,device='cuda')*.1
    n=16 if smoke else 256
    with h5py.File(out/'paths.h5','x') as h:
        for j,parent in enumerate(parent_tensor('dev')[:1 if smoke else 4]):
            initial=parent[None].expand(n,-1,-1);ys=targets(initial,2)
            rng=torch.Generator(device='cuda').manual_seed(925600+j)
            labels=torch.randint(2,(n,),device='cuda',generator=rng)
            z=source(initial,11,rng,0.)
            reference=path_interpolate(z,ys[torch.arange(n,device='cuda'),labels],z.new_full((n,),STOP),rng)[0]
            levels=[16] if smoke else ([128,512] if j==0 else [128])
            for steps in levels:
                start=time.perf_counter();parts=[]
                for b in range(0,n,64):
                    # Paired sources across architectures and step counts.
                    g=torch.Generator(device='cuda').manual_seed(925700+j*100+b)
                    x,_=integrate(initial[b:b+64],ys[b:b+64],steps,g,net);parts.append(x)
                x=torch.cat(parts);torch.cuda.synchronize()
                group=h.create_group(f'parent{j}_steps{steps}')
                group.create_dataset('generated',data=x.cpu().numpy(),compression='lzf')
                group.create_dataset('reference',data=reference.cpu().numpy(),compression='lzf')
                group.create_dataset('time',data=times.cpu().numpy())
                rows.append(dict(parent=j,steps=steps,seconds=time.perf_counter()-start,**summarize_toy(x)))
        h.attrs['complete']=True
    return rows


def run(task,out,smoke):
    torch.set_num_threads(1);net=make_net(task);config=cfg()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-5)
    data=real_data(config) if task['domain']=='bauer' else parent_tensor('train')
    use_aux=task['repair'] in ('joint','early_joint')
    anchors=training_anchors() if use_aux else None
    rng=np.random.default_rng(task['seed']);aux_rng=np.random.default_rng(925800+task['seed'])
    generators={k:torch.Generator(device='cuda').manual_seed(seed+task['seed']) for k,seed in
                [('source',5200),('target',5300),('tau',5400),('aux',5500)]}
    times=torch.arange(11,device='cuda')*.1;history=[];seconds=[];aux_seconds=[]
    input_hash=hashlib.sha256();tau_count=0;draw_count=0
    for step in range(1,(32 if smoke else task['updates'])+1):
        started=time.perf_counter()
        if task['domain']=='bauer':y,times=next_batch(task,config,data,rng,generators['target'],'cuda')
        else:
            ids=torch.randint(len(data),(8,),device='cuda',generator=generators['target'])
            ys=targets(data[ids],2);labels=torch.randint(2,(8,),device='cuda',generator=generators['target'])
            y=ys[torch.arange(8,device='cuda'),labels]
        z=source(y[:,0],11,generators['source'],net.source_shift)
        stop=STOP if task['domain']=='toy' else 1.
        tau,weight=draw_tau(len(y),stop,task['repair'],generators['tau'],'cuda')
        x,v,cut=path_interpolate(z,y,tau,generators['source'])
        if step<=32:
            for val in (y,z):input_hash.update(val.cpu().numpy().tobytes())
        prediction=net.velocity(x,tau,y[:,0],times)
        per_branch=(prediction[:,1:]-v[:,1:]).square().sum(-1).flatten(1).mean(1)
        loss=(weight*per_branch).mean();optimizer.zero_grad(set_to_none=True);loss.backward()
        aux_info={};aux_time=0.
        if use_aux and step%16==0:
            started_aux=time.perf_counter();reference=anchors[int(aux_rng.integers(len(anchors)))].cuda()
            initial=reference[:1,0].expand(8,-1,-1)
            z_aux=source(initial,11,generators['aux'],2.)
            generated=differentiable_sample(net,initial,times,z_aux,32)
            auxiliary,aux_info=joint_loss(generated,reference)
            auxiliary.backward();torch.cuda.synchronize();aux_time=time.perf_counter()-started_aux
            aux_info.update(auxiliary_loss=float(auxiliary.detach()),aux_seconds=aux_time)
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
        if not torch.isfinite(loss) or not torch.isfinite(grad):raise RuntimeError('nonfinite training')
        optimizer.step();torch.cuda.synchronize();seconds.append(time.perf_counter()-started);aux_seconds.append(aux_time)
        tau_count+=int((tau<.1).sum());draw_count+=len(tau)
        if step%500==0 or aux_info or smoke:
            history.append(dict(step=step,fm_loss=float(loss.detach()),gradient=float(grad),**aux_info))
        if step%500==0:dump(out/'history.json',history);print(task,history[-1],flush=True)
        if not smoke and step in (task['updates']//2,task['updates']):
            torch.save(dict(model=net.state_dict(),optimizer=optimizer.state_dict(),task=task,step=step,
                rng={k:g.get_state() for k,g in generators.items()},data_rng=rng.bit_generator.state,
                auxiliary_rng=aux_rng.bit_generator.state),out/f'step{step}.pt')
    train_seconds=sum(seconds);net.eval()
    evaluation=evaluate_bauer(net,out,smoke) if task['domain']=='bauer' else evaluate_toy(net,out,smoke)
    dump(out/'history.json',history)
    return dict(task=task,train_seconds=train_seconds,evaluation=evaluation,history=history,
        first32_data_source_hash=input_hash.hexdigest(),early_tau_fraction=tau_count/draw_count,
        regular_step_seconds=float(np.median([s for s,a in zip(seconds[4:],aux_seconds[4:]) if a==0])),
        mean_aux_seconds=float(np.mean([s for s in aux_seconds if s])) if any(aux_seconds) else 0.,
        peak_bytes=torch.cuda.max_memory_allocated(),device=torch.cuda.get_device_name(),scientific_certification=False)


def prepare():
    if (HERE/'submission.json').exists():raise RuntimeError('frozen after submission')
    protocol=dict(tasks=tasks(),network='unchanged hierarchical SpinPathTransformer width64 depth4',
        physical='Bauer open L16/25/32,63.824849668K,1ps/0.1ps; all reference data/physics unchanged',
        tau='baseline U(0,stop); early mixture 0.5U(0,0.1)+0.5U(0,stop); importance uses density ratio (1/stop)/q to preserve original expected objective; early intentionally changes weighting',
        joint='Every16 updates: add coefficient1 auxiliary gradient through32 spherical Euler steps,8 generated branches versus16 reference branches of one randomly chosen TRAIN anchor parent; squared normalized increment-variance and upper-triangle two-time-correlation mean differences',
        normalization='Reference-only per-parent scales: mean increment variance floored0.05; mean(1-C) floored0.05. No fitted dev statistics.',
        limitations='Auxiliary finite-branch squared moments have sampling bias;32-step discretization can be overfit. Anchor-only auxiliary does not cover all later-window conditional ensembles. No Hamiltonian, energy conservation, event rules or future input.',
        pairing='Same initial parameters/data/source streams per seed; separate tau and auxiliary RNGs. Additional ensemble examples/compute disclosed, NOT compute-matched.',
        evaluation='Existing36 Bauer dev parents and first4 synthetic dev parents; fixed final checkpoint primary. No new holdout. Toy256 branches, theoretical mode prior0.5, first-parent128/512; Bauer64 branches, first-parent16/64/128/256 via frozen evaluator.',
        uncertainty='Whole-branch pointwise intervals; training seeds separate; no equivalence certification or inferred power guarantee',
        decisions='Compare uniform vs importance vs early on toy; Bauer2x2 early weighting and joint-statistics auxiliary. Check unoptimized energy/spatial observables as well as optimized kinetics.',
        skills='Karpathy isolation and tests; Experimental Design paired streams, concurrent controls, blocked seeds')
    dump(HERE/'protocol.json',protocol)
    lock=json.loads((ROOT/'tests/spin_path_transformer/source_lock.json').read_text())
    for name,digest in lock.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(name)
    extra=[ROOT/'tests/transport_audit/run.py']+list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json']
    for file in extra:lock[str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','smoke']);p.add_argument('--index',type=int,default=0);p.add_argument('--device',default='cuda');a=p.parse_args()
    if a.action=='prepare':prepare();return
    require_gpu(a.device)
    if a.action=='run':
        for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(name)
    out=HERE/f'{a.action}/{a.index:02d}';out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    result=run(tasks()[a.index],out,a.action=='smoke');result['total_seconds']=time.perf_counter()-started
    dump(out/'complete.json',result)


if __name__=='__main__':main()
