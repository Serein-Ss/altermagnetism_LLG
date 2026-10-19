"""Compare teacher interpolation marginals with actual sampled flow marginals."""
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
from scripts.model.sphere import sphere_exp
from tests.bauer_p2_20260923.common import ROOT, Model, dump, generate, observable_samples
from tests.bauer_p2_diagnostics_20260923.run import curves,table
from tests.bauer_flow_probe_20260923.run import checkpoints

HERE=Path(__file__).resolve().parent
TAUS=[0.,.125,.25,.5,.75,.875,1.]


@torch.no_grad()
def trace(net,source,initial,times,steps):
    x=source.clone();x[:,0]=initial;states={0:x.clone()}
    saved={round(t*steps) for t in TAUS}
    for i in range(steps):
        v=net.velocity(x,x.new_full((len(x),),i/steps),initial,times)
        x=sphere_exp(x,v/steps);x[:,0]=initial
        if i+1 in saved:states[i+1]=x.clone()
    return torch.stack([states[round(t*steps)] for t in TAUS])


def prepare():
    previous=HERE.parent/'bauer_flow_probe_20260923'
    old=json.loads((previous/'protocol.json').read_text())
    protocol=dict(parents=old['parents'],tau=TAUS,flow_steps=[128,256],branches=8,seeds=[11,22,33],
        estimate_minutes_per_job=[4,9],
        scope='Original and low-lr frozen checkpoints; same source per branch and steps; teacher interpolation uses reference future ONLY for diagnostic, never as network conditioning.',
        interpretation='Compare ensembles, not trajectory reconstruction. Intermediate energies are descriptive, NOT required physical distributions. Feature mismatch is evidence of marginal mismatch, not proof of support violation or unique cause.',
        uncertainty='8 reference branches per parent. Store whole-branch samples. Tables descriptive; no pass/fail threshold or independence claim for sites, frames, or tau.')
    dump(HERE/'protocol.json',protocol)
    lock=json.loads((previous/'source_lock.json').read_text())
    for name,digest in lock.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for p in list(HERE.glob('*.py'))+[HERE/'protocol.json']:
        lock[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock);(HERE/'logs').mkdir(exist_ok=True)


@torch.no_grad()
def run(seed,device):
    if device!='cuda' or not os.getenv('SLURM_JOB_ID'):raise RuntimeError('Slurm CUDA required')
    torch.set_num_threads(1)
    protocol=json.loads((HERE/'protocol.json').read_text())
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    nets={}
    for name,p in checkpoints(seed).items():
        ck=torch.load(p,map_location=device,weights_only=False)
        net=Model(ck['config']).to(device);net.load_state_dict(ck['model']);net.eval();nets[name]=net
    out=HERE/f'runs/seed_{seed}';out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();rows=[];times=torch.arange(11,device=device)*.1
    for j,row in enumerate(protocol['parents']):
        with h5py.File(ROOT/row['path']) as h:
            ids=np.sort(np.random.default_rng(92340+j).choice(len(h['spins']),8,replace=False))
            y=F.normalize(torch.tensor(h['spins'][ids,:11],device=device),dim=-1)
        gen=torch.Generator(device=device).manual_seed(92350+j)
        z=F.normalize(torch.randn(y.shape,device=device,generator=gen),dim=-1);z[:,0]=y[:,0]
        teacher=torch.stack([path_interpolate(z,y,y.new_full((8,),t),torch.Generator(device=device).manual_seed(92351+j))[0] for t in TAUS]).cpu().numpy()
        ref=[curves(x) for x in teacher]
        arrays=dict(teacher=teacher,source=z.cpu().numpy(),reference=y.cpu().numpy(),tau=np.array(TAUS),time_ps=times.cpu().numpy(),branch_indices=ids)
        models={}
        for name,net in nets.items():
            models[name]={}
            for steps in protocol['flow_steps']:
                s=trace(net,z,y[:,0],times,steps).cpu().numpy();arrays[f'{name}_{steps}']=s
                assert np.isfinite(s).all() and abs(np.linalg.norm(s,axis=-1)-1).max()<2e-6
                assert abs(s[:,:,0]-y[:,0].cpu().numpy()).max()<2e-6
                obs=[curves(x) for x in s]
                models[name][str(steps)]={k:dict(generated=np.stack([o[k] for o in obs]).tolist(),
                    teacher=np.stack([o[k] for o in ref]).tolist(),
                    difference=np.stack([o[k]-r[k] for o,r in zip(obs,ref)]).tolist()) for k in ref[0]}
        np.savez_compressed(out/f'{row["parent_id"]}.npz',**arrays)
        rows.append(dict(parent_id=row['parent_id'],split=row['split'],L=row['L'],family=row['family'],models=models))
        if j%12==0:print(seed,j,flush=True)
    dump(out/'results.json',dict(seed=seed,parents=rows,seconds=time.perf_counter()-started,status='DIAGNOSTIC_ONLY'))


def oracle():
    """Known joint distribution via deterministic rotation pushforward; no training."""
    torch.set_num_threads(1);gen=torch.Generator().manual_seed(92352)
    initial=F.normalize(torch.randn(32,8,3,generator=gen,dtype=torch.float64),dim=-1)
    source=F.normalize(initial[:,None]+.3*torch.randn(32,11,8,3,generator=gen,dtype=torch.float64),dim=-1)
    source[:,0]=initial;times=torch.arange(11,dtype=torch.float64)*.1
    class Rotation:
        def velocity(self,x,tau,initial,times):
            axis=torch.zeros_like(x);axis[...,2]=times[None,:,None]
            v=torch.linalg.cross(axis,x);v[:,0]=0;return v
    angle=times[None,:,None];exact=source.clone()
    exact[...,0]=source[...,0]*angle.cos()-source[...,1]*angle.sin()
    exact[...,1]=source[...,0]*angle.sin()+source[...,1]*angle.cos()
    rows=[]
    for steps in (16,32,64,128,256):
        got=generate(Rotation(),initial,times,torch.Generator().manual_seed(1),steps,source=source)
        error=float((got-exact).square().mean().sqrt())
        rows.append(dict(steps=steps,rmse=error,norm_error=float((got.norm(dim=-1)-1).abs().max()),
            anchor_error=float((got[:,0]-initial).abs().max())))
    assert all(rows[i+1]['rmse']<rows[i]['rmse'] for i in range(4))
    assert rows[-1]['rmse']<.001 and rows[-1]['anchor_error']==0
    dump(HERE/'oracle.json',dict(results=rows,scope='Analytic rotational pushforward on a correlated path ensemble. Checks sampler and time indexing only, NOT trained flow recovery or LLG.'))
    print(rows)


def report():
    parts=['<h1>P2 训练插值与实际运输状态对照</h1><p>本页比较同一flow时间的两个集合：随机源到真实参考路径的训练插值，以及冻结模型的实际采样状态。每父状态8分支，共36训练+36开发父状态。两个模型/两个步数使用相同随机源。不要求中间草稿满足真实LLG能量分布，不以随机路径逐条重合为目标。</p>']
    if (HERE/'oracle.json').exists():
        o=json.loads((HERE/'oracle.json').read_text());parts.append('<h2>已知旋转运输采样器测试（不是训练验证）</h2>'+table(['步数','RMSE','模长误差','首帧误差'],[[r[k] for k in ('steps','rmse','norm_error','anchor_error')] for r in o['results']]))
    completed=[]
    for seed in (11,22,33):
        p=HERE/f'runs/seed_{seed}/results.json'
        if not p.exists():parts.append(f'<p>Seed {seed} 暂无完成结果。</p>');continue
        rows=json.loads(p.read_text())['parents'];completed.append(seed)
        for split in ('train','dev'):
            group=[r for r in rows if r['split']==split]
            for name in ('original','low_lr'):
                for steps in ('128','256'):
                    parts.append(f'<h2>seed {seed} / {split} / {name} / {steps}步</h2>')
                    entries=[]
                    for i,t in enumerate(TAUS):
                        entries.append([t]+[f'{np.mean([np.sqrt(np.mean(np.square(r["models"][name][steps][k]["difference"][i]))) for r in group]):.6f}' for k in ('energy_mean','energy_variance','overlap','increment_variance','increment_memory')])
                    parts.append(table(['flow tau','能量均值RMSE','能量方差RMSE','初态重叠RMSE','增量方差RMSE','相邻增量点积RMSE'],entries))
    parts.append('<p>表为固定父状态上的描述平均；参考分支有限，没有物理通过阈值或总体置信区间。差异不是唯一原因证明，更不等于这些状态完全不在训练支持域。完整逐flow/逐物理帧状态保存在各run NPZ，可按整条分支进一步核查不确定性。tau=1才对应最终物理路径；tau不是物理时间。</p>')
    out=HERE/'assets';out.mkdir(exist_ok=True);body=''.join(parts)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>运输状态诊断</title><style>body{font:15px/1.6 sans-serif;margin:24px}table{display:block;overflow:auto;border-collapse:collapse}td,th{padding:6px;border:1px solid #bbc}</style>'+body)
    dump(out/'summary.json',dict(completed_seeds=completed,status='DIAGNOSTIC_ONLY'))
    from tests.bauer_flow_probe_20260923.run import report as previous
    previous();page=ROOT/'tests/index.html';text=page.read_text()
    page.write_text(text.replace('</main>','<section id="p2-transport"><details><summary>P2最新：训练插值与实际运输状态对照</summary>'+body+'</details></section></main>',1))
    path=ROOT/'tests/progress_dashboard/snapshot.json';snapshot=json.loads(path.read_text())
    snapshot['sources'][str((out/'index.html').relative_to(ROOT))]=hashlib.sha256((out/'index.html').read_bytes()).hexdigest();dump(path,snapshot)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','oracle','report']);p.add_argument('--seed',type=int);p.add_argument('--device',choices=['cuda']);a=p.parse_args()
    if a.mode=='run':run(a.seed,a.device)
    else:globals()[a.mode]()
