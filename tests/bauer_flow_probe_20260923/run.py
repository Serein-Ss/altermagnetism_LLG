"""Frozen-checkpoint flow/frame error and draft-frame sensitivity diagnostics."""
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
from scripts.model.sphere import sphere_exp, tangent_project
from tests.bauer_p2_20260923.common import ROOT, Model, dump
from tests.bauer_p2_diagnostics_20260923.run import table

HERE=Path(__file__).resolve().parent
P2=HERE.parent/'bauer_p2_20260923'


def checkpoints(seed):
    return {'original':P2/f'runs/seed_{seed}/best.pt',
            'low_lr':HERE.parent/f'bauer_optimizer_20260923/runs/low_lr_seed_{seed}/step_3000.pt'}


def prepare():
    rng=np.random.default_rng(92340);rows=[]
    train=json.loads((P2/'manifest_train.json').read_text())
    for size in (16,25,32):
        for family in sorted({r['family'] for r in train}):
            group=sorted([r for r in train if r['L']==size and r['family']==family],key=lambda r:r['parent_id'])
            rows.extend(group[i] for i in rng.choice(len(group),2,replace=False))
    rows+=json.loads((P2/'manifest_dev.json').read_text())
    dump(HERE/'protocol.json',dict(parents=rows,seeds=[11,22,33],
        tau=[0.,.05,.1,.25,.5,.75,.9,.95,.99],repeats=4,branches=8,
        physical_times_ps=[i*.1 for i in range(11)],selection_seed=92340,
        sensitivity=dict(tau=[.1,.5,.9],target_frames=[1,5],remote_frame=9,epsilon=.001),
        interpretation='Teacher interpolation diagnostics, not generated-path validation. FM target has conditional variability; nonzero error is not necessarily model bias. Sensitivity is computational dependence, not evidence of correct physics.',
        uncertainty='Per-parent branch error arrays retained; latent repeats/sites/frames are not independent physical replicates; aggregate tables descriptive, no equivalence gate.',
        estimate_minutes_per_job=[3,8]))
    lock=json.loads((P2/'source_lock.json').read_text())
    for name,digest in lock.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    paths=list(HERE.glob('*.py'))+[HERE/'protocol.json']
    paths += [p for seed in (11,22,33) for p in checkpoints(seed).values()]
    for p in paths:lock[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock)
    (HERE/'logs').mkdir(exist_ok=True)


def branch_stats(pred,target):
    return {'mse':(pred-target).square().sum(-1).mean(-1),
            'prediction_power':pred.square().sum(-1).mean(-1),
            'target_power':target.square().sum(-1).mean(-1),
            'cross':(pred*target).sum(-1).mean(-1)}


@torch.no_grad()
def sensitivity(net,x,tau,initial,times,target_frame,changed_frame,noise,epsilon):
    delta=tangent_project(x[:,changed_frame],noise)
    delta=F.normalize(delta,dim=-1)
    plus=x.clone();minus=x.clone()
    plus[:,changed_frame]=sphere_exp(x[:,changed_frame],epsilon*delta)
    minus[:,changed_frame]=sphere_exp(x[:,changed_frame],-epsilon*delta)
    dv=(net.velocity(plus,tau,initial,times)[:,target_frame]-net.velocity(minus,tau,initial,times)[:,target_frame])/(2*epsilon)
    return dv.square().sum(-1).mean(-1).sqrt().cpu().numpy()


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
    out=HERE/'runs'/f'seed_{seed}';out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();records=[];times=torch.arange(11,device=device)*.1
    for j,row in enumerate(protocol['parents']):
        with h5py.File(ROOT/row['path']) as h:
            # Fixed seeded branch subset; no outcome-based selection.
            idx=np.sort(np.random.default_rng(92340+j).choice(len(h['spins']),8,replace=False))
            y=F.normalize(torch.tensor(h['spins'][idx,:11],device=device),dim=-1)
        accum={name:{key:[] for key in ('mse','prediction_power','target_power','cross')} for name in nets}
        sens={name:[] for name in nets}
        for rep in range(protocol['repeats']):
            gen=torch.Generator(device=device).manual_seed(92341+j*100+rep)
            z=F.normalize(torch.randn(y.shape,device=device,generator=gen),dim=-1);z[:,0]=y[:,0]
            repeat={name:{key:[] for key in accum[name]} for name in nets}
            for t in protocol['tau']:
                tau=y.new_full((len(y),),t)
                x,target,_=path_interpolate(z,y,tau,torch.Generator(device=device).manual_seed(92342+j*100+rep))
                for name,net in nets.items():
                    pred=net.velocity(x,tau,y[:,0],times)
                    for key,value in branch_stats(pred[:,1:],target[:,1:]).items():
                        repeat[name][key].append(value.cpu().numpy())
                    if rep==0 and t in protocol['sensitivity']['tau']:
                        noise=torch.randn(y[:,0].shape,device=device,generator=torch.Generator(device=device).manual_seed(92343+j))
                        for frame in (1,5):
                            for label,changed in [('local',frame+1),('remote',9)]:
                                values=sensitivity(net,x,tau,y[:,0],times,frame,changed,noise,.001)
                                sens[name].append(dict(tau=t,target_frame=frame,changed_frame=changed,kind=label,branch_rms=values.tolist()))
                for name in nets:assert torch.isfinite(pred).all()
            for name in nets:
                for key in accum[name]:accum[name][key].append(np.stack(repeat[name][key]))
        records.append(dict(parent_id=row['parent_id'],split=row['split'],L=row['L'],family=row['family'],branch_indices=idx.tolist(),
            models={name:dict(branch_statistics={k:np.mean(v,axis=0).tolist() for k,v in accum[name].items()},sensitivity=sens[name]) for name in nets}))
        if j%12==0:print(seed,j,flush=True)
    dump(out/'results.json',dict(seed=seed,parents=records,seconds=time.perf_counter()-started,
        axes='branch_statistics=[flow_tau, branch, physical_future_frame]; averaged over 4 latent draws, not 4 physical replicates',status='DIAGNOSTIC_ONLY'))


def report():
    protocol=json.loads((HERE/'protocol.json').read_text())
    parts=['<h1>P2：flow阶段、物理帧误差与远帧敏感性</h1><p>冻结模型，不重训。每个尺寸/初态族抽取2个训练父状态，共36个；另用36个开发父状态。每父状态8条参考分支，4次潜变量抽样；重复潜变量不增加独立物理样本数。</p>',
        '<p>表格列为物理时间0.1至1.0 ps，行为辅助flow时间tau。MSE是运输速度误差，不是LLG力矩误差。FM目标存在条件随机性，MSE不必为零；本页不能替代实际生成路径统计。</p>']
    completed=[]
    for seed in (11,22,33):
        path=HERE/f'runs/seed_{seed}/results.json'
        if not path.exists():parts.append(f'<p>Seed {seed} 暂无完成结果。</p>');continue
        result=json.loads(path.read_text());completed.append(seed)
        for split in ('train','dev'):
            rows=[r for r in result['parents'] if r['split']==split]
            for model in ('original','low_lr'):
                parts.append(f'<h2>Seed {seed} / {split} / {model}</h2>')
                for key in ('mse','prediction_power','target_power'):
                    values=np.mean([np.mean(r['models'][model]['branch_statistics'][key],axis=1) for r in rows],axis=0)
                    parts.append(f'<h3>{key}</h3>'+table(['tau']+[f'{t:.1f}ps' for t in protocol['physical_times_ps'][1:]],
                        [[t]+[f'{v:.5f}' for v in values[i]] for i,t in enumerate(protocol['tau'])]))
                entries=[]
                for tau in (.1,.5,.9):
                    for frame in (1,5):
                        entries.append([tau,frame]+[f'{np.mean([np.mean(s["branch_rms"]) for r in rows for s in r["models"][model]["sensitivity"] if s["tau"]==tau and s["target_frame"]==frame and s["kind"]==kind]):.5f}' for kind in ('local','remote')])
                parts.append(table(['tau','输出帧','相邻帧灵敏度','远帧9灵敏度'],entries))
    parts.append('<p>灵敏度为草稿某帧球面小扰动引起的输出速度变化/扰动角；正值只说明存在计算依赖，不证明学对方向信息，零/小值也不是唯一病因。固定实际初态，未输入额外干净未来。表为固定父状态描述平均，不给虚假的总体置信区间。逐分支数值留在JSON。下一项结构对照须依据本页与已有路径统计共同决定。</p>')
    out=HERE/'assets';out.mkdir(exist_ok=True);body=''.join(parts)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Flow诊断</title><style>body{font:15px/1.6 sans-serif;margin:24px}table{display:block;overflow:auto;border-collapse:collapse}td,th{padding:6px;border:1px solid #bbc}</style>'+body)
    dump(out/'summary.json',dict(completed_seeds=completed,status='DIAGNOSTIC_ONLY'))
    from tests.bauer_optimizer_20260923.run import report as previous_report
    previous_report()
    page=ROOT/'tests/index.html';text=page.read_text()
    page.write_text(text.replace('</main>','<section id="p2-flow-probe"><details><summary>P2最新：逐帧flow误差与远帧敏感性</summary>'+body+'</details></section></main>',1))
    path=ROOT/'tests/progress_dashboard/snapshot.json';snapshot=json.loads(path.read_text())
    snapshot['sources'][str((out/'index.html').relative_to(ROOT))]=hashlib.sha256((out/'index.html').read_bytes()).hexdigest();dump(path,snapshot)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','report']);p.add_argument('--seed',type=int);p.add_argument('--device',choices=['cuda']);a=p.parse_args()
    if a.mode=='run':run(a.seed,a.device)
    else:globals()[a.mode]()
