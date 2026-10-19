"""Descriptive paired physical diagnostics; no automatic equivalence gate."""
import argparse
import html
import json
import re
import h5py
import numpy as np
import torch
from tests.full_data_decay.metrics import compare as explicit_compare
from tests.reference_resolution.report import compare, brief
from tests.moment_bias_repair.report import analyze
from tests.transport_audit.report import table
from .run import HERE, ROOT, tasks, parents, dump


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);assets=HERE/('smoke_assets' if a.smoke else 'assets');assets.mkdir(exist_ok=True)
    results=[];missing=[];errors=[];view=[];devview=[]
    indices=[0,3] if a.smoke else range(6)
    for i in indices:
        task=tasks()[i];out=HERE/('smoke' if a.smoke else 'train')/f'{i:02d}'
        if not (out/'complete.json').exists():missing.append(i);continue
        try:
            run=json.loads((out/'complete.json').read_text());records=[]
            for stage,folder in (('before',out/'before'),('after',out)):
                with h5py.File(folder/'seen.h5') as h:
                    assert h.attrs['complete'] and len(h)==(1 if a.smoke else 24)
                    for j,row in enumerate(parents()[:1] if a.smoke else parents()):
                        g=h[f'parent{j:02d}'];assert json.loads(g.attrs['parent'])==row
                        with h5py.File(ROOT/row['path']) as r:old=r['spins'][:,:11]
                        f=HERE/('smoke_reference' if a.smoke else 'reference')/f'{j:02d}/probe.h5'
                        with h5py.File(f) as r:
                            assert r.attrs['complete'] and r.attrs['role']=='probe' and json.loads(r.attrs['parent'])==row
                            ref=r['spins'][:]
                        for sampler in ('euler8','midpoint8') if a.smoke else ('euler128','midpoint128'):
                            x=g[sampler][:]
                            assert len(x)==(4 if a.smoke else 128)
                            assert np.max(abs(np.linalg.norm(x,axis=-1)-1))<2e-6 and np.max(abs(x[:,0]-ref[:1,0]))<2e-6
                            comparisons={k:explicit_compare(x,v,kind,repeats=2 if a.smoke else 100)
                                         for k,v,kind in [('old8_empirical',old,'empirical'),('unseen_noise',ref,'population')]}
                            records.append(dict(stage=stage,parent=row,sampler=sampler,comparisons=comparisons))
            for stage in ('before','after'):
                for sampler in ('euler8','midpoint8') if a.smoke else ('euler128','midpoint128'):
                    for kind in ('old8_empirical','unseen_noise'):
                        for family in sorted({r['parent']['family'] for r in records}):
                            chosen=[r['comparisons'][kind]['metrics'] for r in records if r['stage']==stage and r['sampler']==sampler and r['parent']['family']==family]
                            view.append([task['seed'],task['repair'],stage,sampler,kind,family,
                                *np.mean([[r[k]['rmse'] for k in ('increment_variance','two_time_spin_correlation','energy_mean')] for r in chosen],axis=0).tolist()])
            dev=[];expanded=[]
            if not a.smoke:
                dev=analyze(out)
                chosen=[r for r in dev if r['split']=='dev' and r['group'].endswith('steps128')]
                assert len(chosen)==36
                assert all(r['norm_error']<2e-6 and r['anchor_error']<2e-6 for r in dev)
                devview.append([task['seed'],task['repair'],'original36dev','euler128',*np.mean([brief(r) for r in chosen],axis=0).tolist()])
                with h5py.File(out/'expanded.h5') as h:
                    assert h.attrs['complete'] and len(h)==18
                    for name,g in h.items():
                        j=int(name.removeprefix('parent'))
                        with h5py.File(ROOT/f'tests/reference_resolution/reference/{j:02d}/paths.h5') as r:
                            assert r.attrs['complete'];refs={k:r[f'{k}/spins'][:] for k in ('base','fine')}
                        for sampler in ('euler128','midpoint128'):
                            x=g[sampler][:];assert len(x)==256
                            assert np.max(abs(np.linalg.norm(x,axis=-1)-1))<2e-6 and np.max(abs(x[:,0]-refs['base'][:1,0]))<2e-6
                            expanded.append(dict(parent=json.loads(g.attrs['parent']),sampler=sampler,
                                **{k:compare(x,v) for k,v in refs.items()}))
                for sampler in ('euler128','midpoint128'):
                    devview.append([task['seed'],task['repair'],'expanded18dev fine128',sampler,
                        *np.mean([brief(r['fine']) for r in expanded if r['sampler']==sampler],axis=0).tolist()])
            dump(out/'diagnostics.json',dict(seen=records,dev=dev,expanded=expanded))
            results.append(dict(index=i,task=task,run=run))
        except Exception as exc:errors.append(dict(index=i,error=repr(exc)))
    for seed in (11,22,33):
        chosen=[r for r in results if r['task']['seed']==seed]
        for key in ('schedule_hash','source_tau_hash'):
            if len({r['run'][key] for r in chosen})>1:errors.append(dict(seed=seed,error='Unpaired '+key))
    dump(assets/'summary.json',dict(results=results,missing=missing,errors=errors,seen_overview=view,dev_overview=devview,scientific_certification=False))
    body=f'<h1>随机分支覆盖修复：原8分支与扩充72分支</h1><p>完成{len(results)}/{len(indices)}；缺失{missing}；错误{html.escape(str(errors))}。</p>'
    body+='<p>24个L16训练初态（倾斜、内部反向畴、单畴壁各8个），每个另生64条训练分支和128条独立诊断未来。后者仍属于训练父状态，只检验已见初态的新噪声泛化，不是独立父状态验证集。开放边界、原参数；1ps跨度、0.1ps输出。三训练seed配对，对照与修复训练步数和父状态频率一致。</p>'
    body+='<h2>同初态检验：修复前后</h2><p>old8_empirical是固定8条经验分布；unseen_noise是未用于训练的新128条未来，二者不能混为同一物理判据。越小表示相应统计差异越小，不是自动通过分数。</p>'
    body+=table(['seed','方案','阶段','采样器','参考类型','初态族','增量方差RMSE','两时关联RMSE','能量均值RMSE'],view)
    body+='<h2>独立开发初态：防止只记住训练父状态</h2>'+table(['seed','方案','参考','采样器','增量方差RMSE','两时关联RMSE','能量均值RMSE'],devview)
    body+='<p>各任务diagnostics.json包含逐父状态的磁化、Hamiltonian能量、空间关联、增量和两时统计及整分支bootstrap区间。父状态/训练seed不可被大量格点或帧替代；区间含零不证明等价。重复使用的dev不称为最终测试；本轮没有自动宣告物理准确或尺寸泛化。</p>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1450px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    dashboard=ROOT/'tests/index.html'
    if not a.smoke and dashboard.exists():
        text=dashboard.read_text();start='<!-- branch-coverage-repair:start -->';end='<!-- branch-coverage-repair:end -->';block=start+body+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,text,flags=re.S)
        else:text=text.replace('</body>',block+'</body>') if '</body>' in text else text+block
        temp=dashboard.with_suffix('.html.tmp');temp.write_text(text);temp.replace(dashboard)
    if missing or errors:raise RuntimeError('Incomplete experiment: see summary')


if __name__=='__main__':main()
