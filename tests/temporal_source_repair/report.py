"""Source-law paired audit; independent physical metrics, not FM loss selection."""
import argparse
import html
import json
import re
import h5py
import numpy as np
import torch
from tests.reference_resolution.report import compare,brief
from tests.transport_audit.report import table
from .run import HERE,ROOT,tasks,dump


def analyze(index,smoke):
    out=HERE/('smoke' if smoke else 'train')/f'{index:02d}';run=json.loads((out/'complete.json').read_text())
    assert run['task']==tasks()[index];rows=[];solver=[]
    for step in run['stages']:
        folder=out/f'step{step:05d}'
        with h5py.File(folder/'paths.h5') as h:
            assert h.attrs['complete']
            for role in ('dev','seen'):
                assert len(h[role])==(1 if smoke else (18 if role=='dev' else 24))
                for name,g in h[role].items():
                    i=int(name.removeprefix('parent'));parent=json.loads(g.attrs['parent'])
                    path=ROOT/(f'tests/reference_resolution/reference/{i:02d}/paths.h5' if role=='dev' else f'tests/branch_coverage_repair/reference/{i:02d}/probe.h5')
                    with h5py.File(path) as ref:
                        assert ref.attrs['complete'] and json.loads(ref.attrs['parent'])==parent
                        refs={k:ref[f'{k}/spins'][:] for k in ('base','fine')} if role=='dev' else dict(probe=ref['spins'][:])
                    arrays={k:v[:] for k,v in g.items() if k.startswith('midpoint')}
                    for sampler,x in arrays.items():
                        ref=next(iter(refs.values()))
                        norm=float(abs(np.linalg.norm(x,axis=-1)-1).max());anchor=float(abs(x[:,0]-ref[:1,0]).max())
                        assert norm<2e-6 and anchor<2e-6
                        rows.append(dict(step=step,role=role,parent=parent,sampler=sampler,norm_error=norm,anchor_error=anchor,
                                         comparison={k:compare(x,v) for k,v in refs.items()}))
                    if role=='dev':
                        lo,hi=('midpoint8','midpoint16') if smoke else ('midpoint128','midpoint512')
                        solver.append(dict(step=step,parent=parent,same_latent_rms=float(np.sqrt(np.mean((arrays[lo]-arrays[hi])**2)))))
    dump(out/'report.json',dict(task=tasks()[index],run=run,rows=rows,solver=solver,scientific_certification=False))


def aggregate(smoke):
    results=[];errors=[];missing=[];view=[]
    for i in range(2 if smoke else 6):
        p=HERE/('smoke' if smoke else 'train')/f'{i:02d}/report.json'
        if not p.exists():missing.append(i);continue
        r=json.loads(p.read_text());results.append(r)
        for step in r['run']['stages']:
            for role in ('dev','seen'):
                selected=[v for v in r['rows'] if v['step']==step and v['role']==role]
                for L in sorted({v['parent']['L'] for v in selected}):
                    for sampler in sorted({v['sampler'] for v in selected}):
                        ss=[v for v in selected if v['parent']['L']==L and v['sampler']==sampler]
                        values=[brief(v['comparison']['fine' if role=='dev' else 'probe']) for v in ss]
                        view.append([r['task']['seed'],r['task']['rho'],step,role,L,sampler,*np.mean(values,axis=0).tolist()])
    for seed in (11,) if smoke else (11,22,33):
        pair=[r for r in results if r['task']['seed']==seed]
        if len(pair)==2:
            for key in ('initial_hash','paired_input_hash','schedule_hash'):
                if pair[0]['run'][key]!=pair[1]['run'][key]:errors.append(dict(seed=seed,error='unpaired '+key))
    assets=HERE/('smoke_assets' if smoke else 'assets');assets.mkdir(exist_ok=True)
    dump(assets/'summary.json',dict(results=results,missing=missing,errors=errors,overview=view,scientific_certification=False))
    body=f'<h1>时间相关潜变量：配对修复验证</h1><p>完成{len(results)}；缺失{missing}；错误{html.escape(str(errors))}。'+('仅冒烟，不是科学结果。' if smoke else '')+'</p>'
    body+='<p>同一当前骨干、同一训练数据和初始化，从头训练12000步。rho=0逐帧独立；rho=0.8每格点平稳AR(1)，不同格点独立，单帧边缘分布相同。仅改变生成潜变量时序相关，非修改LLG热噪声。没有重新标记物理时间，没有使用真实未来作为推理条件。</p>'
    body+='<p>Bauer开放链L16/25/32，1ps、0.1ps网格。6000步仅过程诊断，12000步为预定主比较；两组均新训练，不可与旧24000步模型当作等预算因果对照。既有旧骨干全局共享噪声实验不被覆盖。本轮不是首次提出相关噪声。</p>'
    body+=table(['seed','rho','训练步数','评估类','L','采样器','增量方差RMSE','两时关联RMSE','能量均值RMSE'],view)
    body+='<p>dev：独立开发父状态，128生成对128fine参考；seen：训练父状态64生成对未用于训练的128probe未来，不是独立父状态泛化。100次整条分支bootstrap及全部分项指标保存在JSON；格点/帧不是独立样本，逐点区间不是整体等价认证。rho改变运输目标，FM loss不可直接作为物理准确性比较。没有自动通过或正式晋升。</p>'
    body+='<p><a href="summary.json">完整分项指标、区间与计时</a></p>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1500px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    if not smoke:
        p=ROOT/'tests/index.html';text=p.read_text();start='<!-- temporal-source-repair:start -->';end='<!-- temporal-source-repair:end -->'
        block=start+'<section id="temporal-source-repair">'+body.replace('href="summary.json"','href="temporal_source_repair/assets/summary.json"')+'</section>'+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,text,flags=re.S)
        else:text=text.replace('</body>',block+'</body>') if '</body>' in text else text+block
        tmp=p.with_suffix('.html.tmp');tmp.write_text(text);tmp.replace(p)
    if errors or missing:raise RuntimeError('Incomplete or unpaired campaign; see summary')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['analyze','aggregate']);p.add_argument('--smoke',action='store_true');p.add_argument('--index',type=int,default=0);a=p.parse_args();torch.set_num_threads(1)
    if a.action=='analyze':analyze(a.index,a.smoke)
    else:aggregate(a.smoke)


if __name__=='__main__':main()
