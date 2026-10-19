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
from .run import HERE,ROOT,tasks,parents,dump


def one(index,smoke):
    out=HERE/('smoke' if smoke else 'train')/f'{index:02d}';run=json.loads((out/'complete.json').read_text())
    task=tasks()[index];assert run['task']==task;expanded=[];seen=[];view=[];velocity=[]
    with h5py.File(out/'expanded.h5') as h:
        assert h.attrs['complete'] and len(h)==(1 if smoke else 18)
        for name,g in h.items():
            j=int(name.removeprefix('parent'))
            with h5py.File(ROOT/f'tests/reference_resolution/reference/{j:02d}/paths.h5') as r:
                assert r.attrs['complete'] and r.attrs['parent']==g.attrs['parent'];refs={k:r[f'{k}/spins'][:] for k in ('base','fine')}
            for sampler in ('euler8','midpoint8') if smoke else ('euler128','midpoint128'):
                x=g[sampler][:];assert len(x)==(4 if smoke else 256)
                assert np.max(abs(np.linalg.norm(x,axis=-1)-1))<2e-6 and np.max(abs(x[:,0]-refs['base'][:1,0]))<2e-6
                expanded.append(dict(parent=json.loads(g.attrs['parent']),sampler=sampler,**{k:compare(x,v) for k,v in refs.items()}))
    for L in sorted({r['parent']['L'] for r in expanded}):
        for sampler in sorted({r['sampler'] for r in expanded}):
            rows=[r for r in expanded if r['parent']['L']==L and r['sampler']==sampler]
            view.append([task['seed'],task['repair'],L,sampler,*np.mean([brief(r['fine']) for r in rows],axis=0).tolist()])
    with h5py.File(out/'seen.h5') as h:
        assert h.attrs['complete'] and len(h)==(1 if smoke else 24)
        for j,parent in enumerate(parents()[:1] if smoke else parents()):
            with h5py.File(ROOT/f'tests/branch_coverage_repair/reference/{j:02d}/probe.h5') as r:
                assert r.attrs['complete'] and r.attrs['role']=='probe';ref=r['spins'][:]
            for sampler in ('euler8','midpoint8') if smoke else ('euler128','midpoint128'):
                x=h[f'parent{j:02d}/{sampler}'][:]
                seen.append(dict(parent=parent,sampler=sampler,comparison=seen_compare(x,ref,'population',repeats=2 if smoke else 100)))
    for stage,folder in [('before',out/'before'),('after',out)]:
        rows=json.loads((folder/'velocity_diagnostics.json').read_text())['rows']
        for kind in sorted({r['kind'] for r in rows}):
            for tau in sorted({r['tau'] for r in rows}):
                selected=[r for r in rows if r['kind']==kind and r['tau']==tau]
                for window,sl in [('0.1-0.3ps',slice(0,3)),('0.4-1ps',slice(3,10))]:
                    velocity.append([task['seed'],task['repair'],stage,kind,tau,window,*[float(np.mean([r[k][sl] for r in selected])) for k in ('sampled_target_mse','empirical_field_mse','target_noise')]])
    dev=[] if smoke else analyze(out)
    if not smoke:
        assert len([r for r in dev if r['split']=='dev' and r['group'].endswith('steps128')])==36
        assert all(r['norm_error']<2e-6 and r['anchor_error']<2e-6 for r in dev)
    dump(out/'diagnostics.json',dict(expanded=expanded,seen=seen,dev=dev))
    dump(out/'report.json',dict(task=task,run=run,overview=view,velocity=velocity))


def aggregate(smoke):
    assets=HERE/('smoke_assets' if smoke else 'assets');assets.mkdir(exist_ok=True)
    results=[];missing=[];errors=[];view=[];velocity=[];prior=[]
    for i in range(2 if smoke else 6):
        f=HERE/('smoke' if smoke else 'train')/f'{i:02d}/report.json'
        if not f.exists():missing.append(i);continue
        r=json.loads(f.read_text());results.append(r);view.extend(r['overview']);velocity.extend(r['velocity'])
    for seed in (11,22,33):
        rows=[r for r in results if r['task']['seed']==seed]
        for key in ('input_hash','schedule_hash'):
            if len({r['run'][key] for r in rows})>1:errors.append(dict(seed=seed,error='Unpaired '+key))
    if not smoke:
        prior=json.loads((ROOT/'tests/sampling_balance_repair/assets/summary.json').read_text())['prior_dev']
    dump(assets/'summary.json',dict(results=results,missing=missing,errors=errors,overview=view,prior=prior,velocity=velocity,scientific_certification=False))
    body=f'<h1>条件速度拟合与随机监督噪声诊断</h1><p>完成{len(results)}；缺失{missing}；错误{html.escape(str(errors))}。</p>'
    body+='<p>固定原采样比例、训练数据、网络、低学习率余弦3e-5至3e-6。sampled_fm普通随机速度目标；conditional_mean仅在24个训练初态的原始窗口且tau≤0.9时，使用其72条训练路径的解析条件平均速度。dev与probe只用于诊断，绝不参与监督。其他窗口与正常生成接口不变。</p>'
    body+='<h2>独立路径统计：修复前与两组续训</h2>'+table(['seed','方案','L','采样器','增量方差RMSE','两时关联RMSE','能量均值RMSE'],prior+view)
    body+='<h2>按运输时间tau和物理时间分段的速度误差</h2><p>三个指标依次为：对随机配对目标的误差；对有限样本条件平均速度的误差；随机配对目标与其条件均值的差。最后一项是监督随机性，不是LLG热噪声强度。有限样本条件速度不是真实LLG总体速度，不同池大小也影响结果；不能把这张表当作物理认证。</p>'
    body+=table(['seed','方案','阶段','样本池','tau','物理帧区间','随机目标MSE','经验条件速度MSE','目标随机性'],velocity)
    body+='<p>所有tau/物理帧的原始诊断在velocity_diagnostics.json；逐父状态独立路径统计及整分支bootstrap在diagnostics.json。误差减少不等于独立动力学正确，不自动选模、放宽阈值或推广正式模型。新增平均监督有额外计算成本，两组训练输入及随机流配对。三个训练种子分别报告。</p>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1450px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    dashboard=ROOT/'tests/index.html'
    if not smoke and dashboard.exists():
        text=dashboard.read_text();start='<!-- velocity-target-repair:start -->';end='<!-- velocity-target-repair:end -->';block=start+body+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,text,flags=re.S)
        else:text=text.replace('</body>',block+'</body>') if '</body>' in text else text+block
        temp=dashboard.with_suffix('.html.tmp');temp.write_text(text);temp.replace(dashboard)
    if missing or errors:raise RuntimeError('Incomplete campaign')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','aggregate']);p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    if a.action=='run':one(a.index,a.smoke)
    else:aggregate(a.smoke)


if __name__=='__main__':main()
