import argparse
import html
import json
import re
import h5py
import numpy as np
import torch
from tests.full_data_decay.metrics import compare as empirical_compare
from tests.reference_resolution.report import compare,brief
from tests.moment_bias_repair.report import analyze
from tests.transport_audit.report import table
from .run import HERE,ROOT,tasks,parents,dump


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    assets=HERE/('smoke_assets' if a.smoke else 'assets');assets.mkdir(exist_ok=True)
    results=[];missing=[];errors=[];seenview=[];devview=[];counts=[]
    for i in range(2 if a.smoke else 6):
        task=tasks()[i];out=HERE/('smoke' if a.smoke else 'train')/f'{i:02d}'
        if not (out/'complete.json').exists():missing.append(i);continue
        try:
            run=json.loads((out/'complete.json').read_text());seen=[];extra=[];dev=[]
            with h5py.File(out/'seen.h5') as h:
                assert h.attrs['complete'] and len(h)==(1 if a.smoke else 24)
                for j,row in enumerate(parents()[:1] if a.smoke else parents()):
                    g=h[f'parent{j:02d}'];assert json.loads(g.attrs['parent'])==row
                    with h5py.File(ROOT/f'tests/branch_coverage_repair/reference/{j:02d}/probe.h5') as r:
                        assert r.attrs['complete'] and r.attrs['role']=='probe';ref=r['spins'][:]
                    with h5py.File(ROOT/row['path']) as r:old=r['spins'][:,:11]
                    for sampler in ('euler8','midpoint8') if a.smoke else ('euler128','midpoint128'):
                        x=g[sampler][:];assert len(x)==(4 if a.smoke else 128)
                        assert np.max(abs(np.linalg.norm(x,axis=-1)-1))<2e-6 and np.max(abs(x[:,0]-ref[:1,0]))<2e-6
                        comparisons={k:empirical_compare(x,v,kind,repeats=2 if a.smoke else 100) for k,v,kind in [('old8_empirical',old,'empirical'),('unseen_noise',ref,'population')]}
                        seen.append(dict(parent=row,sampler=sampler,comparisons=comparisons))
            for sampler in sorted({v['sampler'] for v in seen}):
                for family in sorted({v['parent']['family'] for v in seen}):
                    for kind in ('old8_empirical','unseen_noise'):
                        values=[v['comparisons'][kind]['metrics'] for v in seen if v['sampler']==sampler and v['parent']['family']==family]
                        seenview.append([task['seed'],task['repair'],sampler,family,kind,*np.mean([[v[k]['rmse'] for k in ('increment_variance','two_time_spin_correlation','energy_mean')] for v in values],axis=0).tolist()])
            if not a.smoke:
                dev=analyze(out);assert len([v for v in dev if v['split']=='dev' and v['group'].endswith('steps128')])==36
                assert all(v['norm_error']<2e-6 and v['anchor_error']<2e-6 for v in dev)
                with h5py.File(out/'expanded.h5') as h:
                    assert h.attrs['complete'] and len(h)==18
                    for name,g in h.items():
                        j=int(name.removeprefix('parent'))
                        with h5py.File(ROOT/f'tests/reference_resolution/reference/{j:02d}/paths.h5') as r:
                            assert r.attrs['complete'];refs={k:r[f'{k}/spins'][:] for k in ('base','fine')}
                        for sampler in ('euler128','midpoint128'):
                            x=g[sampler][:];assert len(x)==256
                            assert np.max(abs(np.linalg.norm(x,axis=-1)-1))<2e-6 and np.max(abs(x[:,0]-refs['base'][:1,0]))<2e-6
                            extra.append(dict(parent=json.loads(g.attrs['parent']),sampler=sampler,**{k:compare(x,v) for k,v in refs.items()}))
                for L in (16,25,32):
                    for sampler in ('euler128','midpoint128'):
                        values=[v for v in extra if v['parent']['L']==L and v['sampler']==sampler]
                        devview.append([task['seed'],task['repair'],L,sampler,*np.mean([brief(v['fine']) for v in values],axis=0).tolist()])
            c=run['counts'];counts.append([task['seed'],task['repair'],*[c.get(k,0)/c['total'] for k in ('L16','L25','L32','anchor','eligible','new_branch')]])
            dump(out/'diagnostics.json',dict(seen=seen,dev=dev,expanded=extra));results.append(dict(index=i,task=task,run=run))
        except Exception as exc:errors.append(dict(index=i,error=repr(exc)))
    for seed in (11,22,33):
        selected=[r for r in results if r['task']['seed']==seed]
        for key in ('schedule_hash','source_tau_hash'):
            if len({r['run'][key] for r in selected})>1:errors.append(dict(seed=seed,error='Unpaired '+key))
        if not a.smoke:
            expected=next(v for v in json.loads((HERE/'sampling_audit.json').read_text())['native'] if v['seed']==seed)['trace_hash']
            if any(r['run']['schedule_hash']!=expected for r in selected):errors.append(dict(seed=seed,error='Native schedule changed'))
    prior=[]
    if not a.smoke:
        old=json.loads((ROOT/'tests/condition_time_repair/assets/summary.json').read_text())
        for result in old['results']:
            if result['task']['repair']!='control':continue
            for L in (16,25,32):
                for sampler in ('euler128','midpoint128'):
                    vals=[v for v in result['expanded'] if v['parent']['L']==L and v['sampler']==sampler]
                    prior.append([result['task']['seed'],'before24000',L,sampler,*np.mean([brief(v['fine']) for v in vals],axis=0).tolist()])
    dump(assets/'summary.json',dict(results=results,missing=missing,errors=errors,seen_overview=seenview,dev_overview=devview,prior_dev=prior,sampling=counts,scientific_certification=False))
    body=f'<h1>恢复原采样比例：随机分支替换修复</h1><p>完成{len(results)}；缺失{missing}；错误{html.escape(str(errors))}。</p>'
    body+='<p>上一轮额外提高L16初态窗口权重；本轮两组均恢复原尺寸、父状态、窗口采样规则。native_old保留旧分支；native_expanded只在抽到对应24个L16原始初态时，从已有72分支抽样替换。相同step24000起点、6000次FM更新、三seed。没有新数据或网络改动。</p>'
    body+='<h2>实际训练暴露比例</h2>'+table(['seed','方案','L16','L25','L32','初态窗口','可替换窗口','新分支'],counts)
    body+='<h2>独立开发：修复前与修复后，按尺寸分开</h2>'+table(['seed','方案','L','采样器','增量方差RMSE','两时关联RMSE','能量均值RMSE'],prior+devview)
    body+='<h2>已见初态、新噪声与原8条经验分布</h2>'+table(['seed','方案','采样器','初态族','参考类型','增量方差RMSE','两时关联RMSE','能量均值RMSE'],seenview)
    body+='<p>新分支暴露次数较上一轮减少；恢复分布与降低额外曝光剂量不可完全分开。旧结果是历史背景，不是本轮同期对照。已有开发父状态反复使用，不是最终留出集。逐父状态磁化、能量、空间/时间统计及整分支bootstrap在各任务diagnostics.json；区间含零不是等价证明。若某尺寸改善、其他尺寸恶化，仍须报告权衡，不宣布修复成功。</p>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1450px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    dashboard=ROOT/'tests/index.html'
    if not a.smoke and dashboard.exists():
        text=dashboard.read_text();start='<!-- sampling-balance-repair:start -->';end='<!-- sampling-balance-repair:end -->';block=start+body+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,text,flags=re.S)
        else:text=text.replace('</body>',block+'</body>') if '</body>' in text else text+block
        temp=dashboard.with_suffix('.html.tmp');temp.write_text(text);temp.replace(dashboard)
    if missing or errors:raise RuntimeError('Incomplete experiment: see summary')


if __name__=='__main__':main()
