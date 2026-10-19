"""Common-state diagnostics; independent branch uncertainty, no tuning on dev futures."""
import argparse
import html
import json
import re
import h5py
import numpy as np
import torch
from tests.transition_block_repair.metrics import compare_first
from tests.transport_audit.report import table
from .run import HERE,ROOT,selected,tasks,dump,seeds


def brief(value):
    return [value['increment_mean']['rmse'],value['increment_cov']['rmse'],
            value['increment_variance']['difference'],value['energy_mean']['rmse']]


def analyze(index,smoke):
    out=HERE/('smoke_gpu' if smoke else 'gpu')/f'{index:02d}';run=json.loads((out/'complete.json').read_text())
    rows=[];solver=[];numerical=[]
    with h5py.File(out/'paths.h5') as h:
        assert h.attrs['complete'] and len(h)==(1 if smoke else 216)
        for row in selected(smoke=smoke):
            sid=row['index'];key=f'state{sid:03d}';g=h[key]
            assert json.loads(g.attrs['metadata'])==row
            path=HERE/('smoke_reference' if smoke else 'reference')/f"{row['parent_index']:02d}/paths.h5"
            with h5py.File(path) as ref:
                assert ref.attrs['complete'] and ref.attrs['role']=='development_diagnostic_never_train'
                # Numerics.duration_ps metadata in reference is explicitly updated to restart span.
                metadata=json.loads(ref[key].attrs['metadata'])
                for field in ('index','parent','origin','branch','frame','state_sha256'):assert metadata[field]==row[field],field
                assert metadata['config']['physics']==row['config']['physics']
                base=ref[f'{key}/base/spins'][:];fine=ref[f'{key}/fine/spins'][:]
                for name,flag in (('base',False),('fine',True)):
                    assert np.array_equal(ref[f'{key}/{name}/seeds'][:],seeds(sid,flag,len(base)))
                assert np.array_equal(base[:,0],np.repeat(ref[f'{key}/initial'][:][None],len(base),0))
                assert np.array_equal(fine[:,0],base[:,0])
                assert np.allclose(ref[f'{key}/time_ps'][:],[0.,.1])
                assert np.max(abs(g['model_initial'][:]-ref[f'{key}/initial'][:]))<2e-7
            numerical.append(dict(state=row,base_vs_fine=compare_first(base,fine)))
            arrays={}
            for steps in ((8,16) if smoke else (128,512)):
                x=g[f'midpoint{steps}'][:];arrays[steps]=x
                assert x.shape[1:]==(2,row['parent']['L'],3) and np.isfinite(x).all()
                rows.append(dict(state=row,steps=steps,base=compare_first(x,base),fine=compare_first(x,fine)))
            lo,hi=(8,16) if smoke else (128,512)
            solver.append(dict(state=sid,paired_path_rms=float(np.sqrt(np.mean((arrays[lo]-arrays[hi])**2)))))
    dump(out/'report.json',dict(task=tasks()[index],run=run,rows=rows,solver=solver,numerical=numerical,scientific_certification=False))


def aggregate(smoke):
    results=[];missing=[];overview=[]
    for i in ([0] if smoke else range(6)):
        p=HERE/('smoke_gpu' if smoke else 'gpu')/f'{i:02d}/report.json'
        if not p.exists():missing.append(i);continue
        r=json.loads(p.read_text());results.append(r)
        for origin in sorted({v['state']['origin'] for v in r['rows']}):
            for L in sorted({v['state']['parent']['L'] for v in r['rows']}):
                for frame in (3,8):
                    for steps in ((8,16) if smoke else (128,512)):
                        ss=[v for v in r['rows'] if v['state']['origin']==origin and v['state']['parent']['L']==L and v['state']['frame']==frame and v['steps']==steps]
                        if ss:overview.append([r['task']['seed'],r['task']['repair'],origin,L,frame*.1,steps,len(ss),*np.mean([brief(v['fine']) for v in ss],0).tolist()])
    assets=HERE/('smoke_assets' if smoke else 'assets');assets.mkdir(exist_ok=True)
    dump(assets/'summary.json',dict(results=results,missing=missing,overview=overview,scientific_certification=False))
    body=f'<h1>真实中间态与生成中间态：同构型独立未来核验</h1><p>模型完成{len(results)}；缺失{missing}。'+('仅冒烟测试。' if smoke else '')+'</p>'
    body+='<p>冻结prefix/adjacent各3seed的step12000。18个开发父状态，每个选真实LLG、prefix seed11、adjacent seed11三类轨迹的固定分支0/1，在0.3/0.8ps取状态。共216个固定构型，每个重新启动128条0.25fs和128条0.125fs LLG未来，跨度0.1ps；模型从同构型独立生成128条。没有真实未来条件，没有训练。</p>'
    body+='<p>所有6个模型共享状态集合：用于隔离条件转移差异，而不是声称所有模型都在自己的典型状态上评估。生成状态来自seed11，两条源分支只是有限诊断切片；同父状态的来源/时间相关，不是216个独立父状态。新参考严格开发诊断用，不入训练库。历史其他初态中间态参考不能替代这些确切状态。</p>'
    body+=table(['模型seed','模型方案','中间态来源','L','来源物理时间ps','flow步数','固定状态数','增量均值RMSE','增量协方差RMSE','增量方差有符号偏差','能量均值RMSE'],overview)
    body+='<p>若真实中间态上仍失败，单步条件转移误差不能归咎于自由接续本身；若主要在生成态上失败，则检查状态覆盖和偏移，但来源构型分布不同不能单独证明唯一因果。两个LLG步长对照含独立抽样误差，不是自动容差。100次整未来分支bootstrap，逐点区间不是总体等价认证。所有状态源路径、分支、时间、参考/生成轨迹、初态舍入误差与计时保存。</p><p><a href="summary.json">完整指标、参考数值对照和置信区间</a></p>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1550px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    if not smoke:
        p=ROOT/'tests/index.html';text=p.read_text();start='<!-- intermediate-restart-repair:start -->';end='<!-- intermediate-restart-repair:end -->'
        block=start+'<section id="intermediate-restart-repair">'+body.replace('href="summary.json"','href="intermediate_restart_repair/assets/summary.json"')+'</section>'+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,text,flags=re.S)
        else:text=text.replace('</body>',block+'</body>') if '</body>' in text else text+block
        tmp=p.with_suffix('.html.tmp');tmp.write_text(text);tmp.replace(p)
    if missing:raise RuntimeError('Incomplete diagnostic, see summary')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['analyze','aggregate']);p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    if a.action=='analyze':analyze(a.index,a.smoke)
    else:aggregate(a.smoke)


if __name__=='__main__':main()
