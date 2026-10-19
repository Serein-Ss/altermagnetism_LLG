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
from tests.bauer_p2_20260923.common import ROOT, Model, cfg, dump, generate, comparison
from tests.bauer_p2_diagnostics_20260923.run import table

HERE=Path(__file__).resolve().parent
P2=HERE.parent/'bauer_p2_20260923'


def decompose(s):
    d=np.diff(np.asarray(s,dtype=np.float64),axis=1)
    mean=d.mean(0);c=d-mean
    variance=(c*c).sum(-1).mean((0,2))
    cov=(c[:,1:]*c[:,:-1]).sum(-1).mean((0,2))
    drift=(mean[1:]*mean[:-1]).sum(-1).mean(-1)
    raw=(d[:,1:]*d[:,:-1]).sum(-1).mean((0,2))
    denom=np.sqrt(variance[1:]*variance[:-1])
    corr=np.divide(cov,denom,out=np.zeros_like(cov),where=denom>1e-14)
    lengths=np.linalg.norm(d,axis=-1);pair=lengths[:,1:]*lengths[:,:-1]
    valid=pair>1e-14
    cosine=np.divide((d[:,1:]*d[:,:-1]).sum(-1),pair,out=np.zeros_like(pair),where=valid)
    counts=valid.sum((0,2))
    direction=np.divide(cosine.sum((0,2)),counts,out=np.zeros_like(cov),where=counts>0)
    assert np.allclose(raw,cov+drift,atol=1e-12)
    return dict(variance=variance,raw_dot=raw,drift_dot=drift,centered_cov=cov,
        normalized_cov=corr,direction_cosine=direction,
        normalized_cov_valid=(denom>1e-14).astype(float),direction_valid_fraction=valid.mean((0,2)))


def paired(g,r,seed=91):
    a,b=decompose(g),decompose(r);rng=np.random.default_rng(seed)
    boot={k:[] for k in a}
    for _ in range(200):
        x=decompose(g[rng.integers(len(g),size=len(g))])
        y=decompose(r[rng.integers(len(r),size=len(r))])
        for k in boot:boot[k].append(x[k]-y[k])
    return {k:dict(generated=a[k].tolist(),reference=b[k].tolist(),difference=(a[k]-b[k]).tolist(),
                   ci95=np.quantile(boot[k],[.025,.975],axis=0).tolist()) for k in a}


def prepare():
    rows=json.loads((P2/'manifest_train.json').read_text());rng=np.random.default_rng(92326)
    selected=[]
    for family in sorted({r['family'] for r in rows}):
        group=sorted([r for r in rows if r['L']==16 and r['family']==family],key=lambda r:r['parent_id'])
        selected.append(group[int(rng.integers(len(group)))])
    dev=[r for r in json.loads((P2/'manifest_dev.json').read_text()) if r['L']==16]
    config=cfg();config.update(sizes=[16],updates=2000,validate_every=500,
        training_windows='fixed_first_1ps_6_train_parents_8_branches',selection='fixed_snapshots_no_selection')
    dump(HERE/'protocol.json',dict(config=config,train=selected,dev=dev,selection_seed=92326,
        seeds=[11,22,33],snapshots=[0,500,1000,1500,2000],estimate_minutes_per_gpu=[10,18],
        scope='Empirical fitting diagnostic, not certification; frozen P2 descriptive control is not compute matched.',
        skill_reference='Kassis et al. 2026 Scientific Agent Skills doi:10.48550/arXiv.2609.00065'))
    lock=json.loads((P2/'source_lock.json').read_text())
    for name,digest in lock.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for p in [HERE/'run.py',HERE/'protocol.json']:
        lock[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock)


def load_paths(rows):
    paths=[]
    for row in rows:
        with h5py.File(ROOT/row['path']) as h:paths.append(h['spins'][:,:11].astype('f4'))
    return paths


def probe(net,data,times):
    gen=torch.Generator(device='cuda').manual_seed(92327);losses=[]
    with torch.no_grad():
        for y in data:
            for _ in range(4):
                z=F.normalize(torch.randn(y.shape,device='cuda',generator=gen),dim=-1);z[:,0]=y[:,0]
                tau=torch.rand(len(y),device='cuda',generator=gen)
                x,target,_=path_interpolate(z,y,tau,gen)
                losses.append(float((net.velocity(x,tau,y[:,0],times)[:,1:]-target[:,1:]).square().sum(-1).mean()))
    return float(np.mean(losses))


def fit(seed,device):
    if device!='cuda' or not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm CUDA required')
    protocol=json.loads((HERE/'protocol.json').read_text())
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    config=protocol['config'];torch.set_num_threads(1);torch.manual_seed(seed)
    gen=torch.Generator(device=device).manual_seed(seed+999);rng=np.random.default_rng(seed)
    data={s:[torch.tensor(x,device=device) for x in load_paths(protocol[s])] for s in ('train','dev')}
    net=Model(config).to(device);optimizer=torch.optim.AdamW(net.parameters(),lr=config['learning_rate'],weight_decay=1e-5)
    times=torch.arange(11,device=device)*.1
    out=HERE/'runs'/f'seed_{seed}';out.mkdir(parents=True,exist_ok=False)
    history=[];evaluations=[];start=time.perf_counter()
    for step in range(config['updates']+1):
        if step:
            net.train();parent=int(rng.integers(6));indices=torch.tensor(rng.integers(8,size=8),device=device)
            y=data['train'][parent][indices]
            z=F.normalize(torch.randn(y.shape,device=device,generator=gen),dim=-1);z[:,0]=y[:,0]
            tau=torch.rand(8,device=device,generator=gen);x,target,_=path_interpolate(z,y,tau,gen)
            optimizer.zero_grad(set_to_none=True)
            loss=(net.velocity(x,tau,y[:,0],times)[:,1:]-target[:,1:]).square().sum(-1).mean()
            assert torch.isfinite(loss)
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(grad)
            optimizer.step();history.append(dict(step=step,loss=float(loss)))
        if step not in protocol['snapshots']:continue
        net.eval();rows=[];probes={s:probe(net,data[s],times) for s in data}
        for split in data:
            for j,(row,y) in enumerate(zip(protocol[split],data[split])):
                initial=F.normalize(y[0,0],dim=-1)[None].expand(16,-1,-1)
                g=generate(net,initial,times,torch.Generator(device=device).manual_seed(92328+j*1000),128).cpu().numpy()
                r=y.cpu().numpy()
                assert np.isfinite(g).all() and abs(np.linalg.norm(g,axis=-1)-1).max()<2e-6
                assert abs(g[:,0]-r[0,0]).max()<2e-6
                dest=out/f'generated_{step}';dest.mkdir(exist_ok=True)
                np.savez_compressed(dest/f'{split}_{row["parent_id"]}.npz',generated=g,reference=r,time_ps=np.arange(11)*.1)
                rows.append(dict(split=split,parent_id=row['parent_id'],family=row['family'],metrics=comparison(g,r),decomposition=paired(g,r)))
        evaluations.append(dict(step=step,probe_FM=probes,parents=rows))
        torch.save(dict(model=net.state_dict(),config=config,step=step,seed=seed),out/f'step_{step}.pt')
        dump(out/'results.json',dict(seed=seed,completed_step=step,seconds=time.perf_counter()-start,
            gpu=torch.cuda.get_device_name(),evaluations=evaluations,history=history,status='DIAGNOSTIC_ONLY'))
        print(seed,step,probes,flush=True)
    dump(out/'complete.json',dict(seconds=time.perf_counter()-start,step=step))


def analyze():
    rows=[];manifest=json.loads((P2/'manifest_dev.json').read_text())
    for seed in (11,22,33):
        for j,row in enumerate(manifest):
            with np.load(P2/'runs'/f'seed_{seed}'/'generated/dev_128'/f'{row["parent_id"]}.npz') as z:
                rows.append(dict(seed=seed,**row,curves=paired(z['generated'],z['reference'],91+j)))
    dump(HERE/'decomposition.json',dict(parents=rows,contract='Fixed-parent whole-branch 200-bootstrap pointwise CI; 1/B covariance for exact raw=drift+cov identity; zero denominators masked.'))


def report():
    parts=['<h1>P2 幅度/时间相关分解与固定小样本拟合</h1><p>开发诊断，不是物理认证。6个L16训练父状态、每个8条1 ps路径；独立开发12父状态。网络和FM不变，2000次更新，三个seed；不改原P2。</p>',
        '<p>增量d=S(t+0.1ps)-S(t)。raw_dot=drift_dot+centered_cov，分别为相邻增量点积、均值增量点积、去均值涨落点积。normalized_cov用涨落RMS归一化；direction_cosine先归一化各增量再比较方向。零分母通过valid标记，不作物理解读。variance从0.1 ps起，其余配对量从0.2 ps起，步距0.1 ps。</p>']
    path=HERE/'decomposition.json'
    if path.exists():
        results=json.loads(path.read_text())['parents']
        for seed in (11,22,33):
            group=[r for r in results if r['seed']==seed];entries=[]
            for k in group[0]['curves']:
                a=np.mean([r['curves'][k]['generated'] for r in group],axis=0)
                b=np.mean([r['curves'][k]['reference'] for r in group],axis=0)
                entries.append([k,', '.join(f'{v:.4f}' for v in a),', '.join(f'{v:.4f}' for v in b)])
            parts.append(f'<h2>原P2 seed {seed} 固定开发父状态平均曲线</h2>'+table(['指标','生成','参考'],entries))
    for seed in (11,22,33):
        path=HERE/'runs'/f'seed_{seed}'/'results.json'
        if not path.exists():parts.append(f'<h2>小样本seed {seed}：暂无结果</h2>');continue
        result=json.loads(path.read_text());entries=[]
        for e in result['evaluations']:
            for split in ('train','dev'):
                rows=[r for r in e['parents'] if r['split']==split];keys=list(rows[0]['metrics'])
                entries.append([e['step'],split,f'{e["probe_FM"][split]:.5f}']+[f'{np.mean([r["metrics"][k]["rmse"] for r in rows]):.5f}' for k in keys])
        parts.append(f'<h2>小样本seed {seed} 已记录step {result["completed_step"]}</h2>'+table(['step','split','固定随机probe FM']+keys,entries))
    parts.append('<p>逐父状态200次完整分支bootstrap点态区间在JSON，不是同时区间或总体区间。训练参考8条、开发16条，不直接比较二者RMSE判断过拟合；看同一split随训练的变化。小样本成功只是有限经验分布可拟合，不是物理泛化；失败不能唯一归因容量。原P2不是等预算对照。没有新的自动通过阈值。</p>')
    out=HERE/'assets';out.mkdir(exist_ok=True)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>P2拟合诊断</title><style>body{font:15px/1.6 sans-serif;margin:24px}table{border-collapse:collapse;display:block;overflow:auto}td,th{border:1px solid #bbc;padding:8px}</style>'+''.join(parts),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','fit','analyze','report']);p.add_argument('--seed',type=int);p.add_argument('--device',choices=['cuda']);a=p.parse_args()
    if a.mode=='fit':fit(a.seed,a.device)
    else:globals()[a.mode]()
