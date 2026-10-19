"""Per-run temporal checkpoint audit, followed by an independent aggregation job."""
import argparse
import html
import json
import re
import h5py
import numpy as np
import torch
from tests.reference_resolution.report import compare,brief
from tests.full_data_decay.metrics import compare as seen_compare
from tests.moment_bias_repair.report import analyze
from tests.transport_audit.report import table
from .run import HERE,ROOT,tasks,parents,STAGES,dump


def analyze_run(index,smoke):
    out=HERE/('smoke' if smoke else 'train')/f'{index:02d}'
    run=json.loads((out/'complete.json').read_text());task=tasks()[index]
    assert run['task']==task and run['stages']==([16,32] if smoke else list(STAGES))
    overview=[];stages=[];seen=[]
    for step in run['stages']:
        folder=out/f'step{step:05d}';rows=[]
        stage=json.loads((folder/'complete.json').read_text());assert stage['step']==24000+step
        with h5py.File(folder/'expanded.h5') as h:
            assert h.attrs['complete'] and len(h)==(1 if smoke else 18)
            for name,g in h.items():
                j=int(name.removeprefix('parent'));parent=json.loads(g.attrs['parent'])
                with h5py.File(ROOT/f'tests/reference_resolution/reference/{j:02d}/paths.h5') as r:
                    assert r.attrs['complete'] and json.loads(r.attrs['parent'])==parent
                    refs={k:r[f'{k}/spins'][:] for k in ('base','fine')}
                for sampler in ('euler8','midpoint8') if smoke else ('euler128','midpoint128'):
                    x=g[sampler][:];assert len(x)==(4 if smoke else 256)
                    assert np.max(abs(np.linalg.norm(x,axis=-1)-1))<2e-6 and np.max(abs(x[:,0]-refs['base'][:1,0]))<2e-6
                    rows.append(dict(parent=parent,sampler=sampler,**{k:compare(x,v) for k,v in refs.items()}))
        for L in sorted({r['parent']['L'] for r in rows}):
            for sampler in sorted({r['sampler'] for r in rows}):
                selected=[r for r in rows if r['parent']['L']==L and r['sampler']==sampler]
                overview.append([task['seed'],task['repair'],step,L,sampler,*np.mean([brief(r['fine']) for r in selected],axis=0).tolist()])
        dump(folder/'diagnostics.json',rows);stages.append(stage)
    with h5py.File(out/'seen.h5') as h:
        assert h.attrs['complete'] and len(h)==(1 if smoke else 24)
        for j,parent in enumerate(parents()[:1] if smoke else parents()):
            g=h[f'parent{j:02d}'];assert json.loads(g.attrs['parent'])==parent
            with h5py.File(ROOT/f'tests/branch_coverage_repair/reference/{j:02d}/probe.h5') as r:
                assert r.attrs['complete'] and r.attrs['role']=='probe';ref=r['spins'][:]
            for sampler in ('euler8','midpoint8') if smoke else ('euler128','midpoint128'):
                x=g[sampler][:]
                assert np.max(abs(np.linalg.norm(x,axis=-1)-1))<2e-6 and np.max(abs(x[:,0]-ref[:1,0]))<2e-6
                seen.append(dict(parent=parent,sampler=sampler,comparison=seen_compare(x,ref,'population',repeats=2 if smoke else 100)))
    dev=[] if smoke else analyze(out)
    if not smoke:
        assert len([r for r in dev if r['split']=='dev' and r['group'].endswith('steps128')])==36
        assert all(r['norm_error']<2e-6 and r['anchor_error']<2e-6 for r in dev)
    dump(out/'final_diagnostics.json',dict(seen=seen,dev=dev))
    dump(out/'report.json',dict(task=task,run=run,stages=stages,overview=overview,scientific_certification=False))


def aggregate(smoke):
    assets=HERE/('smoke_assets' if smoke else 'assets');assets.mkdir(exist_ok=True)
    results=[];missing=[];errors=[];overview=[];prior=[]
    for i in range(3 if smoke else 9):
        f=HERE/('smoke' if smoke else 'train')/f'{i:02d}/report.json'
        if not f.exists():missing.append(i);continue
        r=json.loads(f.read_text());assert r['task']==tasks()[i];results.append(r);overview.extend(r['overview'])
    for seed in (11,22,33):
        rows=[r for r in results if r['task']['seed']==seed]
        for step in ([16,32] if smoke else STAGES):
            for key in ('schedule_hash','source_tau_hash'):
                values={next(v for v in r['stages'] if v['step']==24000+step)[key] for r in rows}
                if len(values)>1:errors.append(dict(seed=seed,step=step,error='unpaired '+key))
        if not smoke and rows:
            expected=next(r for r in json.loads((ROOT/'tests/sampling_balance_repair/sampling_audit.json').read_text())['native'] if r['seed']==seed)['trace_hash']
            if any(r['run']['schedule_hash']!=expected for r in rows):errors.append(dict(seed=seed,error='sampler drift'))
    if not smoke:
        old=json.loads((ROOT/'tests/condition_time_repair/assets/summary.json').read_text())
        for r in old['results']:
            if r['task']['repair']!='control':continue
            for L in (16,25,32):
                for sampler in ('euler128','midpoint128'):
                    rows=[v for v in r['expanded'] if v['parent']['L']==L and v['sampler']==sampler]
                    prior.append([r['task']['seed'],'frozen_start',0,L,sampler,*np.mean([brief(v['fine']) for v in rows],axis=0).tolist()])
    dump(assets/'summary.json',dict(results=results,missing=missing,errors=errors,overview=overview,prior=prior,scientific_certification=False))
    body=f'<h1>学习率续训诊断：中间checkpoint与物理统计</h1><p>完成{len(results)}；缺失{missing}；错误{html.escape(str(errors))}。</p>'
    body+='<p>同一step24000起点与Adam状态；固定原尺寸/初态/窗口采样及扩充分支。restart_high：3e-4余弦降至3e-6；restart_low：3e-5降至3e-6；continue_floor：保持3e-6。三训练seed，每组6000步。只改变学习率策略；没有新数据、网络或损失。</p>'
    body+='<p>新增500/1500/3000/6000步保存完整checkpoint，并对相同18个开发初态各生成256条1ps路径。每个阶段使用相同潜变量及原/半步长LLG参考；表内为半步长参考。每尺寸6个开发初态，按初态等权；指标越小差异越小，不是自动通过阈值。</p>'
    body+=table(['seed','策略','新增步数','L','采样器','增量方差RMSE','两时关联RMSE','能量均值RMSE'],prior+overview)
    body+='<p>固定6000步结果为主比较；中间最优点仅作开发诊断，不追溯宣称最终测试通过。降低学习率可能只保留原性能而没有改善。完整磁化、能量、时空统计与整分支bootstrap保存在各step目录diagnostics.json；最终已见初态新噪声及原36开发初态诊断在final_diagnostics.json。阶段性评估不消耗训练随机流，训练与评估耗时分开记录。无自动模型推广或科学认证。</p>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1450px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    dashboard=ROOT/'tests/index.html'
    if not smoke and dashboard.exists():
        text=dashboard.read_text();start='<!-- lr-continuation-repair:start -->';end='<!-- lr-continuation-repair:end -->';block=start+body+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,text,flags=re.S)
        else:text=text.replace('</body>',block+'</body>') if '</body>' in text else text+block
        temp=dashboard.with_suffix('.html.tmp');temp.write_text(text);temp.replace(dashboard)
    if missing or errors:raise RuntimeError('Incomplete campaign: see summary')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','aggregate']);p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    if a.action=='run':analyze_run(a.index,a.smoke)
    else:aggregate(a.smoke)


if __name__=='__main__':main()
