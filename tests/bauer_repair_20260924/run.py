"""Versioned repair campaign; independent developer evidence, fixed final budgets."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.bauer_p2_20260923.common import ROOT, Model, cfg, dump, generate, comparison
from tests.bauer_comprehensive_20260923.run import (require_gpu, real_data, next_batch,
    references, diagnostic_comparison)
from tests.bauer_comprehensive_20260923.components import synthetic_parents
from .model import ARMS, make_model, draw_source, source_rho

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'bauer_comprehensive_20260923'


def tasks():
    rows=[dict(domain=domain,arm=arm,seed=seed,updates=24000 if domain=='bauer' else 12000)
          for domain in ('bauer','mixture') for seed in (11,22,33) for arm in ARMS]
    np.random.default_rng(92401).shuffle(rows)
    return rows


def prepare():
    if (HERE/'submission.json').exists():raise RuntimeError('already frozen and submitted')
    protocol=dict(tasks=tasks(),controls={
        'baseline':'unchanged local graph; independent source normalize(eps+2*S0)',
        'local_fixed':'active local residual with anchor-containing vector values',
        'temporal_fixed':'same parameters, temporal vector-value mixing',
        'relational_global':'temporal residual plus mean(S0 cross current draft), and global scalar context',
        'temporal_shared':'temporal_fixed with correlated source, unchanged one-site/frame marginal',
        'relational_shared':'relational_global with correlated source, full 2x2 interaction with above'},
        source=dict(shift=2.,rho_independent=0.,rho_shared=.7,
            formula='normalize(2*S0 + sqrt(1-rho^2)*eps + rho*(g cross S0 + h*S0))',
            draws='eps independent per branch/frame/site; g isotropic 3D Gaussian and h scalar Gaussian shared ONLY within each branch across future frames/sites',
            derivation='For unit s: Cov(g cross s+h*s)=(I-ss^T)+ss^T=I. Each marginal is unchanged; rho<1 retains full joint support.',
            semantics='transport latent only, not physical LLG thermal noise; no known mode labels or Hamiltonian are inputs'),
        physical=dict(sizes=[16,25,32],boundary='open',temperature_K=cfg()['temperature_K'],
            block_ps=1.,output_dt_ps=.1,reference_dt_ps=.00025,data='unchanged frozen parent/split library'),
        synthetic='same frozen train/dev parent configurations and analytic mixture law; synthetic coordinate 0..1 has NO ps meaning; dev reused and not a final holdout',
        budgets='Bauer 24000, mixture 12000 updates for EVERY arm, 3 seeds each; parameters/wall time reported separately; equal updates are not equal compute',
        checkpoints='quarter, half, final; final fixed update primary, no retrospective best selection',
        inference='128 steps x64 generated branches per parent; 256 steps shares first16 sources; reference16 Bauer/128 synthetic; train calibration separately',
        uncertainty='100 whole-branch bootstrap draws per fixed parent, independent generated/reference draws, pointwise intervals; seed differences separately; no physics-equivalence threshold',
        diagnostics=['Bauer increment variance/covariance, full two-time correlation, energy, magnetization and spatial statistics',
            'synthetic mixture branch signed rotation score, directional coherence, amplitude, temporal persistence; valid projection thresholds .05/.1/.2, diagnostic only, never Bauer acceptance events',
            'new residual RMS and gradient norms, paired input-stream hashes, 128/256 sampler sensitivity'],
        erratum='20260923 local_extra residual projects to zero and is INVALID as an effective-capacity control. Old files/results remain frozen; other arms remain separately interpretable.',
        limitations='No hierarchy/large-size transfer, no new temperature, no physical event labels; global relational path is a diagnostic candidate, not completed multiscale physics.',
        skill_use='Karpathy Guidelines: surgical versioned repair and regression; Experimental Design: common random numbers, blocked seeds, 2x2 interaction and concurrent controls')
    dump(HERE/'protocol.json',protocol)
    lock=json.loads((OLD/'source_lock.json').read_text())
    for name,digest in lock.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for p in list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json']:
        lock[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock);(HERE/'logs').mkdir(exist_ok=True)


def verify():
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name


def config_for(task):
    config=cfg();config.update(domain=task['domain'],diagnostic_arm=task['arm'],source_shift=2.,source_rho=source_rho(task['arm']))
    if task['domain']!='bauer':config.update(sizes=[16],time_unit='synthetic_coordinate_not_ps')
    else:config['time_unit']='ps'
    return config


def train_step(net,optimizer,y,times,generator,arm):
    z,draws=draw_source(y.shape,y[:,0],generator,arm)
    tau=torch.rand(len(y),device=y.device,generator=generator)
    x,target,cut=path_interpolate(z,y,tau,generator)
    optimizer.zero_grad(set_to_none=True)
    base=Model.velocity(net,x,tau,y[:,0],times)
    correction=net.residual(x,tau,y[:,0],times)
    loss=(base[:,1:]+correction[:,1:]-target[:,1:]).square().sum(-1).mean()
    if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
    loss.backward()
    residual_grad={}
    for name,p in net.named_parameters():
        if name.startswith('residual_') and name.endswith('weight'):
            residual_grad[name]=float(p.grad.norm()) if p.grad is not None else 0.
    grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
    if not torch.isfinite(grad):raise RuntimeError('nonfinite gradient')
    optimizer.step()
    return dict(loss=float(loss),gradient_norm=float(grad),residual_rms=float(correction.detach().square().mean().sqrt()),
        residual_gradient_norms=residual_grad,**cut),draws,tau


def mixture_samples(paths,threshold=.1):
    """Whole-branch features for THIS synthetic law, not magnetic switching rules."""
    s=np.asarray(paths,dtype=np.float64)
    score=np.cross(s[:,:1],s)[...,2]
    projection=np.linalg.norm(s[:,:1,:,:2],axis=-1)*np.linalg.norm(s[...,:2],axis=-1)
    valid=projection[:,-1]>threshold;counts=valid.sum(-1)
    coherence=np.abs((np.sign(score[:,-1])*valid).sum(-1))/np.maximum(counts,1)
    signed=score.mean(-1)
    return dict(positive_fraction=(signed[:,-1]>0).astype(float),
        direction_coherence=coherence,rotation_amplitude=np.abs(signed[:,-1]),
        signed_rotation=signed[:,-1],valid_projection_fraction=valid.mean(-1),
        unresolved_projection=(counts==0).astype(float),
        direction_persistence=(np.sign(signed[:,1:])==np.sign(signed[:,-1:])).mean(-1))


def mixture_comparison(g,r,repeats=100):
    out={};rng=np.random.default_rng(92406)
    for threshold in (.05,.1,.2):
        a,b=mixture_samples(g,threshold),mixture_samples(r,threshold);rows={}
        for k in a:
            row=dict(generated=float(a[k].mean()),reference=float(b[k].mean()))
            row['difference']=row['generated']-row['reference']
            if repeats:
                diffs=[a[k][rng.integers(len(g),size=len(g))].mean()-b[k][rng.integers(len(r),size=len(r))].mean() for _ in range(repeats)]
                row['pointwise_ci95']=np.quantile(diffs,[.025,.975]).tolist()
            rows[k]=row
        out[str(threshold)]=rows
    return out


def evaluate(net,task,out=None,split='dev',steps=128,count=64,repeats=0,limit=None):
    net.eval();rows=[];times=torch.arange(11,device='cuda')*.1
    for j,(row,truth) in enumerate(references(task['domain'],split)):
        if limit is not None and j>=limit:break
        init=F.normalize(torch.tensor(truth[0,0],device='cuda'),dim=-1)[None].expand(64,-1,-1)
        generator=torch.Generator(device='cuda').manual_seed(92404+j*10000)
        source,draws=draw_source((64,11,row['L'],3),init,generator,task['arm'])
        # Fixed full draw shape ensures identical source prefixes at all sample counts.
        source=source[:count];initial=init[:count]
        torch.cuda.synchronize();tick=time.perf_counter()
        pred=generate(net,initial,times,generator,steps,source=source)
        torch.cuda.synchronize();seconds=time.perf_counter()-tick
        generated=pred.cpu().numpy()
        norm=float(np.abs(np.linalg.norm(generated,axis=-1)-1).max());anchor=float(np.abs(generated[:,0]-truth[0,0]).max())
        assert np.isfinite(generated).all() and norm<2e-6 and anchor<2e-6
        rows.append(dict(**row,metrics=comparison(generated,truth,repeats,seed=92405+j),
            diagnostic=diagnostic_comparison(generated,truth,repeats,92405+j),
            mixture=mixture_comparison(generated,truth,repeats) if task['domain']=='mixture' else None,
            generation_seconds=seconds,max_norm_error=norm,anchor_error=anchor,
            source_draw_hash=hashlib.sha256(draws[:count].cpu().numpy().tobytes()).hexdigest()))
        if out:
            out.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out/f'{row["parent_id"]}.npz',generated=generated,reference=truth,initial=truth[0,0],
                time_grid=times.cpu().numpy(),time_unit='ps' if task['domain']=='bauer' else 'synthetic_coordinate',
                flow_steps=steps,source_rho=source_rho(task['arm']))
    return dict(parents=rows,split=split,steps=steps,count=count,
        mean_parent_RMSE={k:float(np.mean([r['metrics'][k]['rmse'] for r in rows])) for k in rows[0]['metrics']},
        status='DEVELOPMENT_REPAIR_NOT_PHYSICS_CERTIFICATION')


def run(index,device):
    require_gpu(device);verify();task=json.loads((HERE/'protocol.json').read_text())['tasks'][index]
    config=config_for(task);torch.manual_seed(task['seed']);net=make_model(config,task['arm']).cuda()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-5)
    rng=np.random.default_rng(task['seed']);generator=torch.Generator(device=device).manual_seed(task['seed']+999)
    target_generator=torch.Generator(device=device).manual_seed(task['seed']+923630)
    out=HERE/'runs'/f'{index:02d}_{task["domain"]}_{task["arm"]}_{task["seed"]}';out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();data=real_data(config) if task['domain']=='bauer' else synthetic_parents('train').cuda()
    history=[];train_seconds=0.;stream=hashlib.sha256()
    for step in range(1,task['updates']+1):
        net.train();torch.cuda.synchronize();tick=time.perf_counter()
        y,times=next_batch(task,config,data,rng,target_generator,device)
        info,draws,tau=train_step(net,optimizer,y,times,generator,task['arm'])
        torch.cuda.synchronize();train_seconds+=time.perf_counter()-tick
        if step<=10 or step%1000==0:
            for value in (y,draws,tau):stream.update(value.detach().cpu().numpy().tobytes())
        history.append(dict(step=step,**info))
        if step%1000==0:print(task,history[-1],flush=True);dump(out/'history.json',history)
        if step in (task['updates']//4,task['updates']//2,task['updates']):
            torch.save(dict(model=net.state_dict(),config=config,task=task,step=step,optimizer=optimizer.state_dict(),
                numpy_rng=rng.bit_generator.state,flow_rng=generator.get_state(),target_rng=target_generator.get_state(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                source_lock=json.loads((HERE/'source_lock.json').read_text())),out/f'step_{step}.pt')
            dump(out/f'dev_step_{step}_64.json',evaluate(net,task,steps=64,count=16))
    final=evaluate(net,task,out/'generated_dev_128',repeats=100)
    dump(out/'dev_final128.json',final)
    dump(out/'dev_final256.json',evaluate(net,task,out/'generated_dev_256',steps=256,count=16))
    dump(out/'train_final128.json',evaluate(net,task,out/'generated_train_128',split='train',count=16))
    dump(out/'history.json',history)
    dump(out/'complete.json',dict(task=task,step=task['updates'],seconds=time.perf_counter()-started,train_seconds=train_seconds,
        gpu=torch.cuda.get_device_name(),peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        parameters=sum(p.numel() for p in net.parameters()),sample_stream_sha256=stream.hexdigest(),
        mean_parent_RMSE=final['mean_parent_RMSE'],status='COMPUTATION_COMPLETE_NOT_PHYSICAL_PASS'))


def benchmark(device):
    require_gpu(device);verify();rows=[];tick=time.perf_counter();data=real_data(cfg());load=time.perf_counter()-tick
    for arm in ARMS:
        task=dict(domain='bauer',arm=arm,seed=11,updates=24000);config=config_for(task)
        torch.manual_seed(11);net=make_model(config,arm).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.001)
        rng=np.random.default_rng(11);g=torch.Generator(device='cuda').manual_seed(1010);tg=torch.Generator(device='cuda').manual_seed(924)
        seconds=[];info=None
        for i in range(60):
            torch.cuda.synchronize();tick=time.perf_counter();y,t=next_batch(task,config,data,rng,tg,'cuda')
            info,_,_=train_step(net,opt,y,t,g,arm);torch.cuda.synchronize()
            if i>=10:seconds.append(time.perf_counter()-tick)
        if arm!='baseline':assert info['residual_rms']>1e-5 and all(v>0 for v in info['residual_gradient_norms'].values())
        tick=time.perf_counter();evaluate(net,task,steps=128,count=64,repeats=100,limit=1)
        rows.append(dict(arm=arm,seconds_per_update=float(np.mean(seconds)),parent_eval_seconds=time.perf_counter()-tick,
            residual_rms=info['residual_rms'],residual_gradient_norms=info['residual_gradient_norms']))
        print(rows[-1],flush=True)
    dump(HERE/'benchmark.json',dict(rows=rows,cache_load_seconds=load,gpu=torch.cuda.get_device_name(),software_smoke_only=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','benchmark','run']);p.add_argument('--device',choices=['cuda']);p.add_argument('--index',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='benchmark':benchmark(a.device)
    else:run(a.index,a.device)
