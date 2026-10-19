import argparse
import hashlib
import html
import json
import os
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from tests.bauer_p2_20260923.common import Model, generate, comparison, statistics, energy, dump, ROOT

HERE=Path(__file__).resolve().parent
P2=HERE.parent/'bauer_p2_20260923'


def curves(s):
    s=np.asarray(s,dtype=np.float64)
    d=np.diff(s,axis=1); e=energy(s)
    return dict(energy_mean=e.mean(0),energy_variance=e.var(0,ddof=1),
        mz_mean=s[...,2].mean((0,2)),
        overlap=(s*s[:,:1]).sum(-1).mean((0,2)),
        increment_variance=d.var(0,ddof=1).sum(-1).mean(-1),
        adjacent_overlap=(s[:,1:]*s[:,:-1]).sum(-1).mean((0,2)),
        increment_memory=(d[:,1:]*d[:,:-1]).sum(-1).mean((0,2)))


def diagnose(g,r,seed):
    # Match finite-ensemble bias: original reference train=8, dev=16.
    rng=np.random.default_rng(seed)
    matched=[]
    for _ in range(16):
        gi=rng.choice(len(g),8,replace=False); ri=rng.choice(len(r),8,replace=False)
        matched.append(comparison(g[gi],r[ri]))
    a,b=curves(g),curves(r)
    boot={k:[] for k in a}
    for _ in range(200):
        ga=curves(g[rng.integers(len(g),size=len(g))])
        rb=curves(r[rng.integers(len(r),size=len(r))])
        for k in a: boot[k].append(ga[k]-rb[k])
    return dict(full=comparison(g,r),
        matched8={k:float(np.mean([x[k]['rmse'] for x in matched])) for k in matched[0]},
        curves={k:dict(generated=a[k].tolist(),reference=b[k].tolist(),difference=(a[k]-b[k]).tolist(),
            ci95=np.quantile(boot[k],[.025,.975],axis=0).tolist()) for k in a})


def run(seed,device):
    if device!='cuda' or not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('Slurm CUDA allocation required')
    torch.set_num_threads(1)
    checkpoint=P2/'runs'/f'seed_{seed}'/'best.pt'
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    ck=torch.load(checkpoint,map_location=device,weights_only=False)
    net=Model(ck['config']).to(device); net.load_state_dict(ck['model']); net.eval()
    out=HERE/'runs'/f'seed_{seed}'; out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter(); rows=[]
    for split in ('train','dev'):
        manifest=json.loads((P2/f'manifest_{split}.json').read_text())
        for j,row in enumerate(manifest):
            if split=='train':
                with h5py.File(ROOT/row['path']) as h:
                    r=h['spins'][:,:11].astype('f8'); initial=h['initial'][:]
                init=F.normalize(torch.tensor(initial,device=device,dtype=torch.float32),dim=-1)[None].expand(16,-1,-1)
                times=torch.arange(11,device=device)*.1
                latent_seed=923000+j*100000
                g=generate(net,init,times,torch.Generator(device=device).manual_seed(latent_seed),128).cpu().numpy()
                dest=out/'generated'; dest.mkdir(exist_ok=True)
                np.savez_compressed(dest/f'{row["parent_id"]}.npz',generated=g,time_ps=np.arange(11)*.1,
                    initial=initial,latent_seed=latent_seed,reference_path=row['path'])
            else:
                with np.load(P2/'runs'/f'seed_{seed}'/'generated/dev_128'/f'{row["parent_id"]}.npz') as z:
                    g=z['generated'];r=z['reference'].astype('f8');initial=z['initial']
            assert np.isfinite(g).all() and np.max(abs(np.linalg.norm(g,axis=-1)-1))<2e-6
            assert np.max(abs(g[:,0]-initial))<2e-6
            rows.append(dict(split=split,L=row['L'],family=row['family'],parent_id=row['parent_id'],
                             **diagnose(g.astype('f8'),r,j+29)))
            if j%24==0: print(seed,split,j,flush=True)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    dump(out/'results.json',dict(seed=seed,checkpoint_sha256=digest,checkpoint_step=ck['step'],
        seconds=time.perf_counter()-started,parents=rows,
        contract='Frozen checkpoint. Fixed-parent pointwise whole-branch bootstrap, 200 repeats. Matched8 repeats are Monte Carlo subsampling, NOT independent physical replicates. No equivalence gate. Train/dev preparation coverage differs.'))


def table(headers,rows):
    return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'


def report():
    parts=['<h1>P2 固定模型原因诊断</h1><p>三个模型不重新训练。训练参考8分支、开发参考16分支；统一8对8比较取16次不放回子抽样的平均误差。这些重复不是新物理样本。开发路径复用原128步结果；训练父状态补生成16分支。</p>',
        '<p>按固定父状态对完整分支重采样200次，时间曲线区间为点态95%，非同时区间。训练/开发父状态的准备覆盖不同，误差差异不能唯一证明过拟合或容量不足。本页为开发诊断，不自动判定通过。</p>']
    all_results=[]
    for seed in (11,22,33):
        path=HERE/'runs'/f'seed_{seed}'/'results.json'
        if not path.exists():
            parts.append(f'<h2>Seed {seed} 尚无完整报告</h2>');continue
        result=json.loads(path.read_text());all_results.append(result)
        parts.append(f'<h2>Seed {seed}</h2>')
        rows=result['parents']; keys=list(rows[0]['matched8'])
        summary=[]
        for split in ('train','dev'):
            for size in (16,25,32):
                for family in ['all']+sorted({r['family'] for r in rows}):
                    subset=[r for r in rows if r['split']==split and r['L']==size and (family=='all' or r['family']==family)]
                    summary.append([split,size,family,len(subset)]+[f'{np.mean([r["matched8"][k] for r in subset]):.5f}' for k in keys])
        parts.append('<p>逐尺寸、初态族的等样本量RMSE；all表示这些父状态的均值，不是自然占据权重。</p>'+table(['split','L','family','parents']+keys,summary))
        parts.append('<h3>逐父状态、逐时间诊断</h3><p>能量单位H/(NJ)；增量方差是逐格点分支协方差迹的空间平均。increment_memory是相邻增量点积，不是归一化相关系数。每个父状态可展开。</p>')
        for r in rows:
            detail=[]
            for key,c in r['curves'].items():
                start=2 if key=='increment_memory' else 1 if key in ('increment_variance','adjacent_overlap') else 0
                for i,(a,b,d) in enumerate(zip(c['generated'],c['reference'],c['difference'])):
                    detail.append([key,f'{(i+start)*.1:.1f}',f'{a:.5f}',f'{b:.5f}',f'{d:.5f}',f'[{c["ci95"][0][i]:.5f}, {c["ci95"][1][i]:.5f}]'])
            parts.append(f'<details><summary>{r["split"]} / {r["parent_id"]} / {r["family"]}</summary>'+table(['量','t (ps)','生成','参考','差值','点态95%区间'],detail)+'</details>')
    parts.append('<h2>判读顺序</h2><p>训练和开发同样差：优先考虑共同的拟合、目标或表示问题；训练明显更好：考虑覆盖/泛化，并核对父状态差异。比较first increment与后续时刻定位初帧附近偏差；结合能量方差、增量方差和增量记忆，区分局部抖动与累计演化。任何单项都不是唯一病因证明。</p>')
    body=''.join(parts); out=HERE/'assets';out.mkdir(exist_ok=True)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>P2诊断</title><style>body{font:15px/1.6 sans-serif;padding:24px;color:#20313e}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ccd}details{margin:12px 0;overflow:auto}</style>'+body,encoding='utf-8')
    dump(out/'summary.json',dict(completed_seeds=[r['seed'] for r in all_results],status='DIAGNOSTIC_ONLY'))
    from tests.progress_dashboard.build import main
    main()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int);p.add_argument('--device',choices=['cuda']);p.add_argument('--report',action='store_true');a=p.parse_args()
    report() if a.report else run(a.seed,a.device)
