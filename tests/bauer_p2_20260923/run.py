"""Three-seed FM-only local graph: selection on dev, historical test diagnostic."""
import argparse
import hashlib
import json
import os
import time
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.mdgen_path_refactor_20260918.data import PathWindows
from tests.bauer_closed_loop_20260919.run import batch
from .common import HERE, ROOT, cfg, dump, Model, generate, comparison


def evaluate(net, config, split, steps, seed, out=None, repeats=0, limit=None):
    rows = json.loads((HERE/f'manifest_{split}.json').read_text())
    if limit:
        rows=rows[:limit]
    result=[]
    net.eval()
    for j,row in enumerate(rows):
        with h5py.File(ROOT/row['path']) as h:
            truth=h['spins'][:,:round(config['duration_ps']/config['save_ps'])+1].astype('f8')
            initial=h['initial'][:]
        init=F.normalize(torch.tensor(initial,device='cuda',dtype=torch.float32),dim=-1)[None].expand(config['branches'],-1,-1)
        times=torch.arange(truth.shape[1],device='cuda')*config['save_ps']
        torch.cuda.synchronize(); started=time.perf_counter()
        pred=generate(net,init,times,torch.Generator(device='cuda').manual_seed(seed+j*100000),steps)
        torch.cuda.synchronize(); seconds=time.perf_counter()-started
        generated=pred.cpu().numpy()
        norm=float(abs(np.linalg.norm(generated,axis=-1)-1).max())
        anchor=float(abs(generated[:,0]-initial).max())
        assert np.isfinite(generated).all() and norm<2e-6 and anchor<2e-6
        metrics=comparison(generated.astype('f8'),truth,repeats,seed=j+53)
        result.append(dict(parent_id=row['parent_id'],L=row['L'],family=row['family'],metrics=metrics,
                           generation_seconds=seconds,max_norm_error=norm,anchor_error=anchor))
        if out:
            out.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out/f'{row["parent_id"]}.npz',generated=generated,reference=truth.astype('f4'),
                time_ps=times.cpu().numpy(),initial=initial,seed=seed+j*100000,flow_steps=steps)
    return dict(split=split,steps=steps,parents=result,
        mean_parent_RMSE={k:float(np.mean([r['metrics'][k]['rmse'] for r in result])) for k in result[0]['metrics']},
        inference='fixed-parent per-parent whole-branch bootstrap, pointwise scalar CIs; no population CI or automatic equivalence',
        status='DIAGNOSTIC_NOT_PHYSICAL_CERTIFICATION')


def train(index, benchmark=False):
    config=cfg(); seed=config['seeds'][index]
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('requires Slurm CUDA allocation in rtx4090 partition')
    if not json.loads((HERE/'audit.json').read_text())['integrity_pass']:
        raise RuntimeError('P0 integrity audit failed')
    if not benchmark:
        for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError(f'frozen source/config/manifest changed: {name}')
    torch.set_num_threads(1); torch.manual_seed(seed)
    rng=np.random.default_rng(seed); generator=torch.Generator(device='cuda').manual_seed(seed+999)
    net=Model(config).cuda()
    optimizer=torch.optim.AdamW(net.parameters(),lr=config['learning_rate'],weight_decay=1e-5)
    data=[PathWindows([ROOT/config['data']/f'L{n}'],'train',config['duration_ps'],stride_frames=10) for n in config['sizes']]
    out=HERE/('benchmark' if benchmark else f'runs/seed_{seed}')
    out.mkdir(parents=True,exist_ok=False)
    history=[]; selection=[]; best=None; train_seconds=0.
    started=time.perf_counter(); steps=30 if benchmark else config['updates']
    for step in range(1,steps+1):
        net.train(); torch.cuda.synchronize(); tick=time.perf_counter()
        y,times=batch(data,rng,config,'cuda','B')
        z=F.normalize(torch.randn(y.shape,device='cuda',generator=generator),dim=-1); z[:,0]=y[:,0]
        tau=torch.rand(len(y),device='cuda',generator=generator)
        x,target,cut=path_interpolate(z,y,tau,generator)
        optimizer.zero_grad(set_to_none=True)
        loss=(net.velocity(x,tau,y[:,0],times)[:,1:]-target[:,1:]).square().sum(-1).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite FM loss')
        loss.backward(); grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
        if not torch.isfinite(grad):
            raise RuntimeError('nonfinite gradient')
        optimizer.step(); torch.cuda.synchronize(); train_seconds+=time.perf_counter()-tick
        history.append(dict(step=step,loss=float(loss),**cut))
        if step%100==0:
            print(json.dumps(history[-1]),flush=True)
        if benchmark or (step%config['validate_every'] and step!=steps):
            continue
        dev=evaluate(net,config,'dev',config['dev_flow_steps'],73000)
        dump(out/f'dev_{step}.json',dev)
        key=(dev['mean_parent_RMSE']['increment_cov'],dev['mean_parent_RMSE']['energy_mean'])
        selected=best is None or key<best
        selection.append(dict(step=step,criterion=list(key),selected_at_time=selected))
        checkpoint=dict(model=net.state_dict(),config=config,seed=seed,step=step,
            criterion=key,optimizer=optimizer.state_dict(),numpy_rng=rng.bit_generator.state,
            flow_rng=generator.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
            source_lock=json.loads((HERE/'source_lock.json').read_text()),
            manifest_hashes=json.loads((HERE/'audit.json').read_text())['sources'])
        torch.save(checkpoint,out/'last.pt')
        if selected:
            best=key; torch.save(checkpoint,out/'best.pt')
        dump(out/'history.json',history); dump(out/'selection_history.json',selection)
        print(json.dumps(dict(seed=seed,step=step,dev_selection=key)),flush=True)
    if benchmark:
        tick=time.perf_counter()
        smoke=evaluate(net,config,'dev',64,73000,out/'generated',repeats=20,limit=3)
        eval_seconds=time.perf_counter()-tick
        result=dict(train_seconds=train_seconds,steps=steps,seconds_per_update=train_seconds/steps,
            evaluation_seconds=eval_seconds,parents=3,gpu=torch.cuda.get_device_name(),
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),loss_initial=history[0]['loss'],loss_last=history[-1]['loss'],
            estimated_train_eval_seconds=(train_seconds/steps*config['updates']+eval_seconds/3*36*(6+2*240/64))*1.5,
            status='SOFTWARE_SMOKE_ONLY_NOT_TRAINED_DYNAMICS')
        dump(out/'smoke_metrics.json',smoke); dump(out/'benchmark.json',result)
        torch.save(dict(model=net.state_dict(),config=config),out/'smoke.pt')
        print(json.dumps(result),flush=True); return
    checkpoint=torch.load(out/'best.pt',weights_only=False,map_location='cuda')
    net.load_state_dict(checkpoint['model'])
    dump(out/'selection.json',dict(selected_step=checkpoint['step'],criterion=list(best),
        rule=config['selection'],gpu=torch.cuda.get_device_name(),train_seconds=train_seconds,
        parameters=sum(p.numel() for p in net.parameters()),peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    for split in ('dev','diagnostic'):
        for n in config['flow_steps']:
            report=evaluate(net,config,split,n,83000,out/'generated'/f'{split}_{n}',
                            repeats=config['bootstrap_repeats'] if n==128 else 0)
            dump(out/f'{split}_{n}.json',report)
    dump(out/'complete.json',dict(status='COMPUTATION_COMPLETE_NOT_AUTOMATIC_PHYSICS_PASS',
        seconds=time.perf_counter()-started,seed=seed))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--device',choices=['cuda'],required=True); parser.add_argument('--benchmark',action='store_true')
    args=parser.parse_args(); train(args.index,args.benchmark)


if __name__=='__main__':
    main()
