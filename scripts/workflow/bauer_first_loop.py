"""Bounded exploratory Bauer loop: train, independent reference, baselines, decision.

This is a feasibility screen, not the publication protocol's P3/P5 certificate.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from scripts.core.bauer_fast import ensemble
from scripts.workflow.bauer_campaign import initial_states, graph, observables, event_rows, save, digest
from scripts.model.gate_f import build_model, matching_loss, sample, reference_path
from scripts.model.sphere import sphere_exp
from scripts.analysis.gate_f_metrics import bootstrap_family, fit_training_scale


def ar_loss(net, y, condition, times, g, axis):
    """Vectorization of existing per-frame vMF teacher-forcing objective."""
    b,f=y.shape[:2];previous=y[:,:-1].reshape(b*(f-1),*y.shape[2:])
    target=y[:,1:].reshape_as(previous)
    anchor=previous[:,None].expand(-1,2,*previous.shape[1:])
    c=condition[:,None].expand(-1,f-1,-1).reshape(-1,2)
    t=torch.stack((times[:,:-1],times[:,1:]),-1).reshape(-1,2)
    a=axis[:,None].expand(-1,f-1,-1).reshape(-1,3)
    v=net(anchor,torch.ones(len(previous),device=y.device,dtype=y.dtype),previous,c,t,g,a)[:,1]
    mean=sphere_exp(previous,v);k=net.log_concentration.exp().clamp(1e-3,100.)
    log_sinh=k+torch.log1p(-torch.exp(-2*k))-math.log(2)
    return (log_sinh-torch.log(k)-k*(mean*target).sum(-1)).mean(),{}


def features(paths):
    v=observables(np.asarray(paths,dtype=np.float64))
    idx=np.linspace(0,v.shape[1]-1,11,dtype=int)
    return v[:,idx,:4].reshape(len(v),44),v


def data(config,root):
    """Reuse prior pilot only for training/development; new initial/noise for evaluation."""
    train=[];conditions=[];dev=[];devc=[];resolution=[];start=time.monotonic()
    for ti,theta in enumerate(config['theta']):
        source=Path(config['pilot'])/f'T{theta}.h5'
        with h5py.File(source) as h:
            fine=h['spins'][:,:3201]
        coarse=fine[:,::16].copy() # 0..16000 every 80 reduced time units
        train.append(coarse[:96]);dev.append(coarse[96:])
        conditions.extend([[theta,.1]]*96);devc.extend([[theta,.1]]*32)
        ef,_=event_rows(fine[...,2].mean(-1),np.arange(3201)*5.,.6,160.)
        ec,_=event_rows(coarse[...,2].mean(-1),np.arange(201)*80.,.6,160.)
        resolution.append(dict(theta=theta,fine_probability=float(ef[:,0].mean()),
            coarse_probability=float(ec[:,0].mean()),disagreement_count=int((ef[:,0]!=ec[:,0]).sum()),
            completed_events_fine=int(ef[:,0].sum()),completed_events_coarse=int(ec[:,0].sum()),
            definition='basin +/-0.6, completion after dwell160; fine dt_save5 vs model dt_save80',
            source_sha256=digest(source)))
    np.savez_compressed(root/'development.npz',train=np.concatenate(train),condition=np.asarray(conditions,'f4'),
                         development=np.concatenate(dev),development_condition=np.asarray(devc,'f4'))
    save(root/'resolution.json',resolution)
    # New evaluation families are created independently; same initial for R1 and R2.
    for ti,theta in enumerate(config['theta']):
        initials=initial_states(config['test_initials'],25,f'first_loop_v1/eval/{theta}')
        states=np.repeat(initials,2*config['noise_per_half'],axis=0)
        seeds=400000000+ti*1000000+np.arange(len(states))
        t=time.monotonic()
        paths,raw=ensemble(states,seeds,.02,800000,4000,.1,.1,theta)
        n=config['noise_per_half'];paths=paths.reshape(config['test_initials'],2*n,201,25,3).astype('f4')
        with h5py.File(root/f'reference_T{theta}.h5','w') as h:
            h.create_dataset('spins',data=paths,compression='lzf');h.create_dataset('initial',data=initials)
            h.attrs.update(complete=True,theta=theta,dt=.02,save_dt=80.,duration=16000.,raw_norm_error=float(raw.max()))
        print(json.dumps(dict(stage='reference',theta=theta,paths=len(states),seconds=time.monotonic()-t)),flush=True)
    save(root/'data_manifest.json',dict(status='COMPLETE',wall_seconds=time.monotonic()-start,
        training_paths=192,development_paths=64,evaluation_paths=2*config['test_initials']*2*config['noise_per_half'],
        splits='pilot initial families0..5 train,6..7 development; fresh evaluation initial families, no window crossing',
        leakage_check='Initial IDs and future seed namespaces disjoint; prior pilot used to select window, so exploratory only',
        noise_namespace=[400000000,401000000],source_hashes={p.name:digest(p) for p in root.glob('*.h5')}))


def train(config,root,device):
    with np.load(root/'development.npz') as d:
        y=torch.from_numpy(d['train'][:,:,None,:,None,:]).to(device)
        c=torch.from_numpy(d['condition']).to(device)
    times=torch.linspace(0,1,201,device=device)[None] # registered normalization: physical t/16000
    g=graph(25,device,torch.float32);start=time.monotonic()
    results=[]
    for kind in config['kinds']:
        for training_seed in config['training_seeds']:
            torch.manual_seed(training_seed);torch.cuda.manual_seed_all(training_seed)
            net=build_model(kind,width=config['width'],blocks=config['blocks'],condition_mean=(.12,.1),condition_std=(.01,1.)).to(device)
            # Same small output initialization for all models, before seeing outcomes.
            with torch.no_grad():net.output.weight.mul_(.01);net.output.bias.zero_()
            optimizer=torch.optim.Adam(net.parameters(),lr=config['learning_rate'])
            gen=torch.Generator(device=device).manual_seed(500000000+training_seed)
            losses=[];tick=time.monotonic();maxgrad=0.
            for step in range(config['training_steps']):
                index=torch.randint(len(y),(config['batch_size'],),device=device,generator=gen)
                batch=y[index];cond=c[index];t=times.expand(len(batch),-1)
                axis=batch.new_tensor([0.,0.,1.]).expand(len(batch),-1)
                optimizer.zero_grad(set_to_none=True)
                loss,_=ar_loss(net,batch,cond,t,g,axis) if kind=='autoregressive' else matching_loss(net,batch,cond,t,g,axis,gen,kind=kind)
                if not torch.isfinite(loss):raise ValueError(f'Nonfinite training loss {kind}/{training_seed}/{step}')
                loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                if not torch.isfinite(grad):raise ValueError('Nonfinite gradient')
                optimizer.step();losses.append(float(loss));maxgrad=max(maxgrad,float(grad))
            torch.cuda.synchronize()
            name=f'{kind}_{training_seed}'
            torch.save(dict(model=net.state_dict(),kind=kind,seed=training_seed,config=config),root/f'{name}.pt')
            row=dict(kind=kind,seed=training_seed,steps=len(losses),first50_mean=float(np.mean(losses[:50])),
                     last50_mean=float(np.mean(losses[-50:])),seconds=time.monotonic()-tick,max_gradient=maxgrad,
                     parameters=sum(p.numel() for p in net.parameters()),loss_history=losses,
                     checkpoint_sha256=digest(root/f'{name}.pt'))
            results.append(row);save(root/'training.json',dict(status='RUNNING',models=results))
            print(json.dumps({k:v for k,v in row.items() if k!='loss_history'}),flush=True)
    save(root/'training.json',dict(status='COMPLETE',models=results,wall_seconds=time.monotonic()-start))


def generate(config,root,device):
    g=graph(25,device,torch.float32);stats=[]
    with np.load(root/'development.npz') as d:train_data=d['train'];train_conditions=d['condition']
    for ti,theta in enumerate(config['theta']):
        with h5py.File(root/f'reference_T{theta}.h5') as h:
            initials=h['spins'][:,0,0]
        n=config['noise_per_half'];x=torch.from_numpy(np.repeat(initials,n,axis=0)[:,None,:,None,:]).to(device)
        t=torch.linspace(0,1,201,device=device)[None].expand(len(x),-1)
        cond=x.new_tensor([theta,.1]).expand(len(x),-1);axis=x.new_tensor([0.,0.,1.]).expand(len(x),-1)
        for kind in config['kinds']:
            for training_seed in config['training_seeds']:
                checkpoint=torch.load(root/f'{kind}_{training_seed}.pt',map_location=device,weights_only=False)
                net=build_model(kind,width=config['width'],blocks=config['blocks'],condition_mean=(.12,.1),condition_std=(.01,1.)).to(device)
                net.load_state_dict(checkpoint['model']);net.eval()
                gen=torch.Generator(device=device).manual_seed(600000000+ti*100000+training_seed)
                start=time.monotonic();source=reference_path(x,201,gen) if kind=='rfm' else None
                with torch.no_grad():
                    pred=sample(net,x,cond,t,g,axis,gen,kind=kind,steps=32,source=source)
                    if kind=='rfm':double=sample(net,x,cond,t,g,axis,gen,kind=kind,steps=64,source=source)
                torch.cuda.synchronize();finite=bool(torch.isfinite(pred).all())
                norm=float((pred.norm(dim=-1)-1).abs().max()) if finite else None
                name=f'{kind}_{training_seed}_T{theta}'
                np.save(root/f'{name}.npy',pred[:,:,0,:,0].cpu().numpy().reshape(config['test_initials'],n,201,25,3))
                if kind=='rfm':np.save(root/f'{name}_steps64.npy',double[:,:,0,:,0].cpu().numpy().reshape(config['test_initials'],n,201,25,3))
                row=dict(kind=kind,seed=training_seed,theta=theta,finite=finite,norm_max=norm,
                         initial_error=float((pred[:,0]-x).abs().max()),seconds=time.monotonic()-start,
                         seconds_scope='RFM includes both32/64-step samples; others one sampling run')
                stats.append(row);print(json.dumps(dict(stage='sampling',**row)),flush=True)
        rng=np.random.default_rng(700000000+ti);pool=train_data[np.isclose(train_conditions[:,0],theta)]
        retrieved=pool[rng.integers(len(pool),size=len(x))].copy();retrieved[:,0]=np.repeat(initials,n,axis=0)
        np.save(root/f'retrieval_0_T{theta}.npy',retrieved.reshape(config['test_initials'],n,201,25,3))
    save(root/'sampling.json',dict(status='COMPLETE',models=stats))


def evaluate(config,root):
    with np.load(root/'development.npz') as d:train_features,_=features(d['train'])
    mean,scale=fit_training_scale(train_features,np.full(44,.01),split='train')
    save(root/'feature_scale.json',dict(mean=mean.tolist(),scale=scale.tolist(),floor=.01,source='training_only'))
    names=[f'{k}:{s}' for k in config['kinds'] for s in config['training_seeds']]+['retrieval:0']
    groups=[];diagnostics=[];step_doubling=[];rng=np.random.default_rng(20260913)
    for theta in config['theta']:
        with h5py.File(root/f'reference_T{theta}.h5') as h:real=h['spins'][:]
        i,n=real.shape[:2];half=n//2
        def feat(p):return ((features(p.reshape(-1,201,25,3))[0]-mean)/scale).reshape(i,half,44)
        a,b=feat(real[:,:half]),feat(real[:,half:]);models={}
        for name in names:
            kind,training_seed=name.split(':');path=root/f'{kind}_{training_seed}_T{theta}.npy'
            p=np.load(path)
            if not np.isfinite(p).all():
                diagnostics.append(dict(theta=theta,model=name,status='FAIL_NONFINITE'));continue
            models[name]=feat(p)
            flat=p.reshape(-1,201,25,3);primary,obs=features(flat)
            event,_=event_rows(obs[...,3],np.arange(201)*80.,.6,160.)
            ref=real[:,:half].reshape(-1,201,25,3);_,robs=features(ref)
            revent,_=event_rows(robs[...,3],np.arange(201)*80.,.6,160.)
            differences=event-revent
            # Independent ensemble bootstrap within each independent initial family.
            indices=rng.integers(i,size=(2000,i));ar=event.reshape(i,half,-1);br=revent.reshape(i,half,-1)
            draw=[]
            for idx in indices:
                aa=ar[idx[:,None],rng.integers(half,size=(i,half))].mean((0,1));bb=br[idx[:,None],rng.integers(half,size=(i,half))].mean((0,1))
                draw.append(aa-bb)
            ci=np.quantile(draw,[.025,.975],axis=0)
            diagnostics.append(dict(theta=theta,model=name,status='EXPLORATORY',
                reference_reversal=float(revent[:,0].mean()),generated_reversal=float(event[:,0].mean()),
                event_difference=(event.mean(0)-revent.mean(0)).tolist(),event_pointwise_ci95=ci.tolist(),
                event_features=['reversal','return','RMST/window']+[f'survival_{j}/10' for j in range(1,11)],
                reference_nn_correlation=float(robs[...,4].mean()),generated_nn_correlation=float(obs[...,4].mean()),
                reference_increment2=float(np.diff(robs[...,3],axis=1).var()),generated_increment2=float(np.diff(obs[...,3],axis=1).var()),
                reference_Mz_quantiles=np.quantile(robs[:,-1,3],[.05,.5,.95]).tolist(),
                generated_Mz_quantiles=np.quantile(obs[:,-1,3],[.05,.5,.95]).tolist()))
            if kind=='rfm':
                doubled=np.load(root/f'{kind}_{training_seed}_T{theta}_steps64.npy')
                fd=feat(doubled)-models[name]
                step_doubling.append(dict(theta=theta,seed=int(training_seed),max_absolute_mean_difference=float(abs(fd.mean((0,1))).max()),
                                          threshold=.05,status='point_estimate_only'))
        groups.append(dict(group_id=f'T{theta}',real_a=a,real_b=b,generated=models,
            source_families=[f'eval/{theta}/{j}' for j in range(i)],initial_ids=[f'eval/{theta}/{j}' for j in range(i)]))
    valid=[name for name in names if all(name in g['generated'] for g in groups)]
    metrics=bootstrap_family(groups,valid,repetitions=2000,seed=20260913,confidence=.95)
    save(root/'distribution.json',metrics);save(root/'events_and_paths.json',diagnostics);save(root/'step_doubling.json',step_doubling)
    rfm=[]
    for label,point,ci in zip(metrics['labels'],metrics['estimate'],metrics['simultaneous_ci']):
        if '/rfm:' not in label:continue
        if '/ed_excess' in label:tolerance=.1;passing=ci[1]<=tolerance;failed=ci[0]>tolerance
        elif '/minus/' in label:tolerance=.02;passing=ci[1]<=tolerance;failed=ci[0]>tolerance
        else:tolerance=.1;passing=max(abs(v) for v in ci)<=tolerance;failed=ci[0]>tolerance or ci[1]<-tolerance
        rfm.append(dict(label=label,estimate=point,ci=ci,tolerance=tolerance,
                        status='PASS' if passing else ('FAIL' if failed else 'INCONCLUSIVE')))
    save(root/'decision.json',dict(status='STOP_SCALE_UP' if any(r['status']=='FAIL' for r in rfm) or any(n.startswith('rfm:') and n not in valid for n in names) else 'HOLD_FOR_CONFIRMATION',
        scope='First bounded feasibility experiment completed; not a P3/P5 or novelty certificate',
        accuracy_checks=rfm,valid_models=valid,excluded_nonfinite_models=[n for n in names if n not in valid],
        formal_limitations=['Four evaluation initial families and16 noises per half cannot certify small errors.',
            'Reference dt=.02 at L25 is not fully convergence certified; result applies to this discrete reference.',
            'Event intervals are exploratory pointwise, not the registered confirmatory simultaneous family.',
            'No temperature holdout test or matched-accuracy total-cost advantage claim in this first loop.']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],required=True)
    p.add_argument('--stage',choices=['all','data','train','generate','evaluate'],default='all');args=p.parse_args()
    cfg=json.loads(args.config.read_text());root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    if args.stage in ('all','train','generate') and (args.device!='cuda' or not torch.cuda.is_available() or not os.getenv('SLURM_JOB_ID')):
        raise RuntimeError('Training/inference requires Slurm CUDA allocation')
    started=time.monotonic()
    for stage,fn in [('data',data),('train',train),('generate',generate),('evaluate',evaluate)]:
        if args.stage not in ('all',stage):continue
        if stage in ('train','generate'):fn(cfg,root,args.device)
        else:fn(cfg,root)
    save(root/'completed.json',dict(status='COMPLETE',wall_seconds=time.monotonic()-started,job_id=os.getenv('SLURM_JOB_ID'),
                                   device=args.device,config_sha256=digest(args.config)))


if __name__=='__main__':main()
