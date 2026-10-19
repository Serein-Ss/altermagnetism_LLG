"""Empirical fitting versus independent stochastic population diagnostics."""
import argparse
import json
import re
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tests.transition_block_repair.metrics import statistics,compare_first
from tests.conditional_moment_repair.report import metrics
from tests.transport_audit.report import table
from .run import HERE,ROOT,chosen,tasks,dump


def empirical(g,r):
    a=statistics(g);b=statistics(r)
    # The frozen64-point empirical target is a discrete distribution: ddof0.
    for k in ('increment_cov','increment_variance','energy_var','mz_var'):b[k]*=(len(r)-1)/len(r)
    return {k:dict(generated=a[k].tolist(),reference=b[k].tolist(),difference=(a[k]-b[k]).tolist(),rmse=float(np.sqrt(np.mean((a[k]-b[k])**2)))) for k in a}


def analyze(index,smoke):
    folder=HERE/('smoke' if smoke else 'train')/f'{index:02d}';run=json.loads((folder/'complete.json').read_text());assert run['task']==tasks()[index]
    records=[];solver=[]
    for event in run['evaluation']:
        step=event['step']
        with h5py.File(folder/f'paths{step:05d}.h5') as h:
            assert h.attrs['complete'] and len(h)==3
            for row in chosen():
                sid=row['index'];name=f'state{sid:03d}';g=h[name];refs={}
                for role in ('train','probe','fine'):
                    with h5py.File(ROOT/f"tests/training_restart_repair/reference/{row['parent_index']:02d}/{name}_{role}.h5") as ref:
                        assert ref.attrs['complete'] and ref.attrs['role']==role;refs[role]=ref['spins'][:]
                arrays={k:v[:] for k,v in g.items() if k.startswith('midpoint')}
                for sampler,x in arrays.items():
                    assert np.isfinite(x).all() and abs(np.linalg.norm(x,axis=-1)-1).max()<2e-6
                    assert abs(x[:,0]-refs['train'][:1,0]).max()<2e-7
                    records.append(dict(step=step,state=sid,family=row['parent']['family'],sampler=sampler,
                        empirical=empirical(x,refs['train']),probe=compare_first(x,refs['probe']),fine=compare_first(x,refs['fine']),reference_control=compare_first(refs['probe'],refs['fine'])))
                if len(arrays)==2:
                    a,b=arrays.values();solver.append(dict(step=step,state=sid,paired_rms=float(np.sqrt(np.mean((a-b)**2)))))
    dump(folder/'report.json',dict(run=run,records=records,solver=solver,scientific_certification=False))


def aggregate(smoke):
    results=[];missing=[];errors=[];overview=[]
    for i in range(2 if smoke else 6):
        p=HERE/('smoke' if smoke else 'train')/f'{i:02d}/report.json'
        if not p.exists():missing.append(i);continue
        r=json.loads(p.read_text());results.append(r);t=r['run']['task']
        for x in r['records']:
            for role in ('empirical','fine'):
                overview.append([t['seed'],t['repair'],x['step'],x['state'],x['family'],x['sampler'],role,*metrics(x[role])])
    for seed in (11,) if smoke else (11,22,33):
        pair=[r for r in results if r['run']['task']['seed']==seed]
        if len({r['run']['paired_hash'] for r in pair})>1:errors.append([seed,'unpaired training inputs'])
    assets=HERE/('smoke_assets' if smoke else 'assets');assets.mkdir(exist_ok=True)
    dump(assets/'summary.json',dict(overview=overview,missing=missing,errors=errors,scientific_certification=False,
        reports=[f'../{("smoke" if smoke else "train")}/{i:02d}/report.json' for i in range(2 if smoke else 6)]))
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for r in results:
        t=r['run']['task'];color='#0072B2' if t['repair']=='low_lr' else '#D55E00'
        for ax,role in zip(axes,('empirical','fine')):
            steps=sorted({x['step'] for x in r['records']});ys=[]
            for step in steps:
                v=[x[role]['increment_mean']['rmse'] for x in r['records'] if x['step']==step and x['sampler']==('midpoint8' if smoke else 'midpoint128')]
                ys.append(float(np.mean(v)))
            ax.plot(steps,ys,marker='o',color=color,alpha=.7,label=f"{t['repair']} seed{t['seed']}")
            ax.set(xlabel='Additional fitting updates',ylabel='Increment mean RMSE',title=role+' reference')
    axes[0].legend(fontsize=7);fig.savefig(assets/'fit_curves.png',dpi=200);plt.close(fig)
    body=f'<h1>固定构型拟合与局部状态覆盖诊断</h1><p>拟合完成{len(results)}；缺失{missing}；配对错误{errors}。'+('仅冒烟。' if smoke else '')+'</p>'
    body+='<p>固定3个训练构型：tilted/interior_domain/single_wall各一个0.8ps状态，每构型64训练后继。从同一restart_fm18000模型出发，只改变续训学习率3e-5或3e-4，保留相同Adam状态和随机流。集中训练不代表通用模型修复，不用于宣布尺寸或时间推广。</p><img src="fit_curves.png" style="width:100%">'
    body+='<p>左图对应固定64点经验训练分布，目标协方差按ddof0；右图对应独立细步长LLG分支，按总体ddof1评估。拟合训练分布不等于恢复真实总体，曲线为3构型平均并逐seed展示，无置信带不表示精确。最终6000固定，其他快照仅诊断，不事后挑最好。</p>'
    body+=table(['seed','方案','更新','状态','纹理','采样','参考','均值RMSE','协方差RMSE','方差偏差','能量均值RMSE'],overview)
    body+='<h2>局部状态覆盖</h2><p>重放上一轮6000更新实际原生条件取样，使用训练端标准化的radius1/2不变量与边缘距离。比较加入48状态前后与独立开发状态的局部最近距离。它不是全部历史训练覆盖，也不是完备物理特征；散点关联不证明因果。格点、同父状态不同时间不能视为独立实验。</p>'
    coverage_folder='smoke_coverage' if smoke else 'coverage'
    for i in range(1 if smoke else 3):
        p=HERE/coverage_folder/f'{i:02d}/summary.json'
        if p.exists():body+=f'<p>seed{(11,22,33)[i]}</p><img src="../{coverage_folder}/{i:02d}/coverage.png" style="width:100%"><a href="../{coverage_folder}/{i:02d}/summary.json">状态／格点分项数据</a>'
        else:missing.append(f'coverage{i}');body+=f'<p>覆盖诊断{i}缺失，不能解释为通过。</p>'
    body+='<p>少量构型仍学不准时，优化和表达能力尚未分离；经验分布拟合好而独立分支差，需考虑有限样本过拟合；不能把一次失败当体系不可学习。所有详细区间与采样收敛见各训练任务report.json。<a href="summary.json">轻量汇总</a></p>'
    dump(assets/'summary.json',dict(overview=overview,missing=missing,errors=errors,scientific_certification=False,
        reports=[f'../{("smoke" if smoke else "train")}/{i:02d}/report.json' for i in range(2 if smoke else 6)]))
    (assets/'index.html').write_text('<!doctype html><meta charset="utf-8"><body style="max-width:1500px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body>')
    if not smoke:
        p=ROOT/'tests/index.html';text=p.read_text();start='<!-- fixed-state-fit-repair:start -->';end='<!-- fixed-state-fit-repair:end -->'
        linked=body.replace('src="fit_curves.png"','src="fixed_state_fit_repair/assets/fit_curves.png"').replace('href="summary.json"','href="fixed_state_fit_repair/assets/summary.json"').replace('="../coverage/','="fixed_state_fit_repair/coverage/')
        block=start+'<section id="fixed-state-fit-repair">'+linked+'</section>'+end
        text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,text,flags=re.S) if start in text else (text.replace('</body>',block+'</body>') if '</body>' in text else text+block)
        tmp=p.with_suffix('.html.tmp');tmp.write_text(text);tmp.replace(p)
    if missing or errors:raise RuntimeError('Incomplete or unpaired experiment')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['analyze','aggregate']);p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    if a.action=='analyze':analyze(a.index,a.smoke)
    else:aggregate(a.smoke)
