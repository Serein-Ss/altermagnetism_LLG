"""Bounded multi-cause campaign. Development evidence, never automatic certification."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.bauer_p2_20260923.common import ROOT, cfg, dump, generate, comparison
from tests.bauer_closed_loop_20260919.run import batch
from .components import (CachedWindows, make_model, source_from_noise,
                         synthetic_parents, synthetic_paths, diagnostic_arrays)

HERE = Path(__file__).resolve().parent
P2 = HERE.parent/'bauer_p2_20260923'
ARMS = ['baseline', 'wide', 'local_extra', 'temporal', 'source', 'temporal_source', 'anchors', 'global_temporal']
SEEDS = [11, 22, 33]


def tasks():
    rows = [dict(domain='bauer', arm=arm, seed=seed, updates=6000) for seed in SEEDS for arm in ARMS]
    rows += [dict(domain='bauer',arm=arm,seed=seed,updates=24000) for seed in SEEDS
             for arm in ('baseline_long','temporal_source_long','global_temporal_long')]
    rows += [dict(domain=case, arm=arm, seed=seed, updates=3000)
             for seed in SEEDS for case in ('deterministic', 'gaussian', 'mixture')
             for arm in ('baseline', 'temporal', 'global_temporal')]
    # Shuffle scheduling, preserving all deterministic task/seed identities.
    np.random.default_rng(923600).shuffle(rows)
    return rows


def prepare():
    if (HERE/'submission.json').exists():
        raise RuntimeError('Campaign frozen after submission')
    protocol = dict(tasks=tasks(), physical=dict(sizes=[16,25,32], boundary='open',
        temperature_K=cfg()['temperature_K'], reference_dt_ps=.00025, output_dt_ps=.1, block_ps=1.),
        controls={
            'baseline_long': 'baseline extended to 24000 updates; paired first 6000-step budget',
            'temporal_source_long': 'temporal+source extended to 24000 updates; budget interaction',
            'global_temporal_long': 'global communication extended to 24000 updates; budget interaction',
            'baseline': 'same P2 width64/depth3 from scratch, same FM-only objective',
            'global_temporal': 'temporal residual plus all-site mean scalar/vector context, preserving vector magnitude; diagnostic global path, not full hierarchical multiscale',
            'wide': 'width128, depth3; isolates capacity within old topology; not equal FLOPs',
            'local_extra': 'parameter-matched new residual with no cross-time vector values',
            'temporal': 'new invariant temporal attention with vector values, plus original graph',
            'source': 'baseline with normalize(N(0,I)+2*S0) full-support source in both training and sampling',
            'temporal_source': 'tests interaction of temporal residual and source',
            'anchors': 'baseline trained only on original first-1ps anchors, tests window distribution mismatch'},
        synthetic='independent train/dev parent configurations; freshly sampled analytic futures; not LLG; deterministic / correlated tangent Gaussian / equal two-mode rotation mixture',
        synthetic_reference_branches=128, generated_branches=64,
        real_reference_branches=16, final_flow_steps=[128,256],
        pairing='identical baseline initialization and explicit data/noise/tau RNG streams within each seed; wider model naturally has different weights; anchor treatment intentionally changes targets',
        selection='fixed final update is primary; half-budget snapshots descriptive; no retrospective best-checkpoint selection',
        uncertainty='100 fixed-parent independent whole-branch bootstrap replicates for final 128-step metrics; pointwise bands, not simultaneous; reference split repetitions are not independent physical experiments',
        finite_sample='32 repeated disjoint 8+8 reference splits, and matched 8+8 model/reference comparisons; not a physical acceptance threshold',
        scope='Bauer supported 1ps window diagnostic, NOT time/size transfer, temperature generalization, or complete multiscale model',
        limitations='not an exhaustive mathematical guarantee; new final holdout and approved physical tolerances remain needed; no event acceptance rules',
        decisions=[
            'synthetic deterministic failure implicates learning/representation/optimization before complex stochastic LLG',
            'deterministic success but stochastic failure isolates stochastic-law difficulty; compare temporal controls',
            'global_temporal versus temporal tests nonlocal communication, especially global mixture coherence; not proof all large-scale physics',
            'temporal versus local_extra isolates vector-value transmission at matched parameter count',
            'wide improvement alone supports capacity limitation, subject to extra compute cost',
            'source improvement supports transport coupling difficulty, not proof uniform source is mathematically invalid',
            'anchors improvement is distribution-allocation evidence, not reason to discard non-anchor data without coverage review',
            'reference-reference spread quantifies sampling limitations, never an automatic pass/fail threshold'],
        skill_use='Experimental Design: blocked seeds, common random numbers, concurrent controls, declared estimands; Karpathy: isolated tests only')
    dump(HERE/'protocol.json', protocol)
    lock = json.loads((P2/'source_lock.json').read_text())
    for name, digest in lock.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    files = list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json',
        P2/'common.py', ROOT/'tests/mdgen_path_refactor_20260918/data.py',
        ROOT/'tests/bauer_closed_loop_20260919/run.py']
    for file in files:
        lock[str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json', lock)
    (HERE/'logs').mkdir(exist_ok=True)


def verify():
    for name, digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name


def require_gpu(device):
    if device != 'cuda' or not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Slurm CUDA allocation required')
    if os.environ.get('SLURM_JOB_PARTITION') != 'rtx4090':
        raise RuntimeError('GPU campaign must use rtx4090 partition')
    torch.set_num_threads(1)


def config_for(task):
    config = cfg()
    config.update(width=128 if task['arm']=='wide' else 64,
                  learning_rate=.001, diagnostic_arm=task['arm'], domain=task['domain'])
    return config


def real_data(config):
    return [CachedWindows([ROOT/config['data']/f'L{n}'], 'train', 1., stride_frames=10) for n in config['sizes']]


def next_batch(task, config, data, rng, target_generator, device):
    if task['domain'] == 'bauer':
        y, times = batch(data, rng, config, device, 'B')
    else:
        ids = rng.integers(len(data), size=config['batch'])
        initial = data[torch.as_tensor(ids, device=device)]
        y = synthetic_paths(initial, task['domain'], target_generator)
        times = torch.arange(11, device=device)*.1
    return y, times


def train_step(net, optimizer, y, times, generator, arm):
    noise = torch.randn(y.shape, device=y.device, generator=generator)
    z = source_from_noise(noise, y[:, 0], arm)
    tau = torch.rand(len(y), device=y.device, generator=generator)
    x, target, cut = path_interpolate(z, y, tau, generator)
    optimizer.zero_grad(set_to_none=True)
    loss = (net.velocity(x, tau, y[:, 0], times)[:, 1:]-target[:, 1:]).square().sum(-1).mean()
    if not torch.isfinite(loss):
        raise RuntimeError('nonfinite FM loss')
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
    if not torch.isfinite(grad):
        raise RuntimeError('nonfinite gradient')
    optimizer.step()
    return dict(loss=float(loss), gradient_norm=float(grad), **cut), noise, tau


def references(domain, split='dev'):
    if domain == 'bauer':
        rows = json.loads((P2/f'manifest_{split}.json').read_text())
        # Training calibration uses one parent per size and family, not selection.
        if split == 'train':
            seen = set(); selected = []
            for row in rows:
                key = (row['L'], row['family'])
                if key not in seen:
                    seen.add(key); selected.append(row)
            rows = selected
        for row in rows:
            with h5py.File(ROOT/row['path']) as h:
                yield row, h['spins'][:, :11].astype('f4')
    else:
        for j, initial in enumerate(synthetic_parents(split)):
            seed = (923640 if split=='dev' else 923641)+j*1000
            truth = synthetic_paths(initial[None].expand(128,-1,-1), domain, torch.Generator().manual_seed(seed))
            yield dict(parent_id=f'{domain}_{split}_{j}', L=16, family=domain), truth.numpy()


def diagnostic_comparison(g, r, repeats, seed):
    a, b = diagnostic_arrays(g), diagnostic_arrays(r)
    result = {k:dict(generated=a[k].tolist(), reference=b[k].tolist(),
                    difference=(a[k]-b[k]).tolist(), rmse=float(np.sqrt(np.mean((a[k]-b[k])**2)))) for k in a}
    if repeats:
        rng = np.random.default_rng(seed)
        boot = {k:[] for k in a}
        for _ in range(repeats):
            ga = diagnostic_arrays(g[rng.integers(len(g),size=len(g))])
            rb = diagnostic_arrays(r[rng.integers(len(r),size=len(r))])
            for k in a:
                boot[k].append(ga[k]-rb[k])
        for k in a:
            result[k]['pointwise_ci95'] = np.quantile(boot[k],[.025,.975],axis=0).tolist()
    return result


def evaluate(net, task, out, split='dev', steps=128, count=64, repeats=0, limit=None):
    net.eval(); results=[]; times=torch.arange(11,device='cuda')*.1
    for j,(row,truth) in enumerate(references(task['domain'],split)):
        if limit is not None and j>=limit:
            break
        initial=F.normalize(torch.as_tensor(truth[0,0],device='cuda'),dim=-1)[None].expand(count,-1,-1)
        g=torch.Generator(device='cuda').manual_seed(923620+j*10000)
        noise=torch.randn((64,11,row['L'],3),device='cuda',generator=g)[:count]
        source=source_from_noise(noise,initial,task['arm'])
        torch.cuda.synchronize(); tick=time.perf_counter()
        pred=generate(net,initial,times,g,steps,source=source)
        torch.cuda.synchronize(); seconds=time.perf_counter()-tick
        generated=pred.cpu().numpy()
        norm=float(np.abs(np.linalg.norm(generated,axis=-1)-1).max())
        anchor=float(np.abs(generated[:,0]-truth[0,0]).max())
        assert np.isfinite(generated).all() and norm<2e-6 and anchor<2e-6
        metrics=comparison(generated,truth,repeats=repeats,seed=j+923)
        diagnostic=diagnostic_comparison(generated,truth,repeats,j+1923)
        # End-point signed z-rotation discriminant diagnoses synthetic mixture weights.
        mode=None
        if task['domain']=='mixture':
            def fraction(s):
                score=np.cross(s[:,0],s[:,-1])[...,2].mean(-1)
                return dict(positive_fraction=float((score>0).mean()), score_quantiles=np.quantile(score,[.05,.25,.5,.75,.95]).tolist())
            mode=dict(generated=fraction(generated),reference=fraction(truth), analytic_mode_probability=.5,
                      note='score is diagnostic, not an LLG event classifier')
        results.append(dict(**row,metrics=metrics,diagnostic=diagnostic,mixture=mode,
                            generation_seconds=seconds,max_norm_error=norm,anchor_error=anchor))
        if out:
            out.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out/f'{row["parent_id"]}.npz',generated=generated,reference=truth,
                                initial=truth[0,0],time_ps=times.cpu().numpy(),flow_steps=steps)
    return dict(parents=results,split=split,flow_steps=steps,generated_branches=count,
                mean_parent_RMSE={k:float(np.mean([r['metrics'][k]['rmse'] for r in results])) for k in results[0]['metrics']},
                status='DEVELOPMENT_DIAGNOSTIC_NOT_CERTIFIED')


def run(index, device):
    require_gpu(device);verify()
    task=json.loads((HERE/'protocol.json').read_text())['tasks'][index]
    config=config_for(task);seed=task['seed'];torch.manual_seed(seed)
    net=make_model(config,task['arm']).cuda()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-5)
    rng=np.random.default_rng(seed)
    generator=torch.Generator(device=device).manual_seed(seed+999)
    target_generator=torch.Generator(device=device).manual_seed(seed+923630)
    out=HERE/'runs'/f'{index:02d}_{task["domain"]}_{task["arm"]}_{seed}'
    out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    data=real_data(config) if task['domain']=='bauer' else synthetic_parents('train').cuda()
    if task['arm']=='anchors':
        for ds in data:
            ds.records=[(p,r,range(1),s) for p,r,_,s in ds.records]
            ds.ends=np.cumsum([r['branches'] for _,r,_,_ in ds.records]).tolist()
    history=[];stream=hashlib.sha256();train_seconds=0.
    for step in range(1,task['updates']+1):
        net.train();torch.cuda.synchronize();tick=time.perf_counter()
        y,times=next_batch(task,config,data,rng,target_generator,device)
        info,noise,tau=train_step(net,optimizer,y,times,generator,task['arm'])
        torch.cuda.synchronize();train_seconds+=time.perf_counter()-tick
        if step<=10 or step%1000==0:
            for v in (y,noise,tau):
                stream.update(v.detach().cpu().numpy().tobytes())
        history.append(dict(step=step,**info))
        if step==3000:
            common_stream=stream.hexdigest()
        if step%500==0:
            dump(out/'history.json',history);print(task,history[-1],flush=True)
        if step in ({task['updates']//2,task['updates'],6000} if task['updates']>6000 else {task['updates']//2,task['updates']}):
            torch.save(dict(model=net.state_dict(),config=config,task=task,step=step,
                optimizer=optimizer.state_dict(),numpy_rng=rng.bit_generator.state,
                flow_rng=generator.get_state(),target_rng=target_generator.get_state(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                source_lock=json.loads((HERE/'source_lock.json').read_text())),out/f'step_{step}.pt')
            metrics=evaluate(net,task,None,steps=64,count=16)
            dump(out/f'dev_step_{step}_64.json',metrics)
    # Fixed final model. Same first 16 latent draws in 128/256 evaluations.
    final=evaluate(net,task,out/'generated_dev_128',steps=128,count=64,repeats=100)
    dump(out/'dev_final128.json',final)
    dump(out/'dev_final256.json',evaluate(net,task,out/'generated_dev_256',steps=256,count=16))
    dump(out/'train_final128.json',evaluate(net,task,out/'generated_train_128',split='train',steps=128,count=16))
    dump(out/'history.json',history)
    dump(out/'complete.json',dict(task=task,step=task['updates'],seconds=time.perf_counter()-start,
        train_seconds=train_seconds,parameters=sum(p.numel() for p in net.parameters()),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(),
        sample_stream_sha256=stream.hexdigest(),common_stream_sha256=common_stream,mean_parent_RMSE=final['mean_parent_RMSE'],
        status='COMPUTATION_COMPLETE_NOT_PHYSICAL_CERTIFICATION'))


def benchmark(device):
    require_gpu(device);verify();start=time.perf_counter();rows=[]
    config=cfg();data=real_data(config);load_seconds=time.perf_counter()-start
    for arm in ARMS:
        task=dict(domain='bauer',arm=arm,seed=11,updates=6000);config=config_for(task)
        torch.manual_seed(11);net=make_model(config,arm).cuda()
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-5)
        rng=np.random.default_rng(11);g=torch.Generator(device='cuda').manual_seed(1010)
        tg=torch.Generator(device='cuda').manual_seed(924);durations=[]
        for step in range(50):
            torch.cuda.synchronize();tick=time.perf_counter()
            y,t=next_batch(task,config,data,rng,tg,'cuda')
            train_step(net,optimizer,y,t,g,arm)
            torch.cuda.synchronize()
            if step>=10:durations.append(time.perf_counter()-tick)
        tick=time.perf_counter()
        evaluate(net,task,None,steps=128,count=64,repeats=100,limit=1)
        eval_seconds=time.perf_counter()-tick
        rows.append(dict(arm=arm,seconds_per_update=float(np.mean(durations)),
                         per_parent_evaluation_seconds=eval_seconds,parameters=sum(p.numel() for p in net.parameters())))
        print(rows[-1],flush=True)
    dump(HERE/'benchmark.json',dict(rows=rows,cache_load_seconds=load_seconds,
         elapsed_seconds=time.perf_counter()-start,gpu=torch.cuda.get_device_name(),
         peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','benchmark','run'])
    p.add_argument('--device',choices=['cuda']);p.add_argument('--index',type=int);a=p.parse_args()
    if a.mode=='run':run(a.index,a.device)
    elif a.mode=='benchmark':benchmark(a.device)
    else:prepare()
