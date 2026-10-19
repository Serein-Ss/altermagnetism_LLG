"""Report train-fit and independent development evidence separately."""
import html
import json
import re
import h5py
import numpy as np
import torch
from tests.bauer_p2_20260923.common import comparison
from tests.bauer_comprehensive_20260923.run import diagnostic_comparison
from tests.transport_audit.report import table
from .run import HERE,ROOT,tasks,dump


def analyze(out):
    parents=[]
    with h5py.File(out/'paths.h5') as h:
        assert h.attrs['complete']
        for name,g in h.items():
            x=g['generated'][:];r=g['reference'][:]
            parents.append(dict(group=name,split=g.attrs['split'],fit_parent=bool(g.attrs['fit_parent']),
                parent=json.loads(g.attrs['parent']),generated_branches=len(x),reference_branches=len(r),
                physical=comparison(x,r,100,92877),diagnostic=diagnostic_comparison(x,r,100,92877),
                norm_error=float(abs(np.linalg.norm(x,axis=-1)-1).max()),
                anchor_error=float(abs(x[:,0]-r[:1,0]).max())))
    return parents


def main():
    torch.set_num_threads(1);assets=HERE/'assets';assets.mkdir(exist_ok=True)
    results=[];missing=[];errors=[];overview=[]
    for i,task in enumerate(tasks()):
        out=HERE/f'train/{i:02d}'
        if not (out/'complete.json').exists():missing.append(i);continue
        try:
            info=json.loads((out/'complete.json').read_text());parents=analyze(out)
            assert sum(p['split']=='dev' and p['group'].endswith('steps128') for p in parents)==36
            dump(out/'diagnostics.json',parents)
            for split in ('train','fit','dev'):
                ps=[p for p in parents if p['group'].endswith('steps128') and
                    (p['fit_parent'] if split=='fit' else p['split']==split)]
                if not ps:continue
                vals=[np.mean([p['diagnostic'][k]['rmse'] for p in ps]) for k in ('increment_variance','two_time_spin_correlation')]
                vals += [np.mean([p['physical'][k]['rmse'] for p in ps]) for k in ('energy_mean','spatial')]
                overview.append([i,task['seed'],task['repair'],split,len(ps),*[round(float(v),6) for v in vals]])
            results.append(dict(index=i,task=task,run=info,parents=parents))
        except Exception as exc:errors.append(dict(index=i,error=repr(exc)))
    pairing={}
    for seed in (11,22,33):
        hashes=[r['run']['first32_data_source_hash'] for r in results if r['task']['seed']==seed and r['task']['repair'] not in ('fit1','fit6')]
        pairing[str(seed)]=dict(count=len(hashes),identical=len(set(hashes))==1 if hashes else None)
        if len(set(hashes))>1:errors.append(dict(seed=seed,error='unpaired full-data FM streams'))
    dump(assets/'summary.json',dict(results=results,missing=missing,errors=errors,pairing=pairing,scientific_certification=False))
    body='<h1>训练内拟合、辅助项拆分与有限生成样本偏差诊断</h1>'
    body+=f'<p>完成 {len(results)}/21；缺失 {missing}；错误 {html.escape(str(errors))}。这是开发诊断，不是物理通过结论。</p>'
    body+='<p>同一seed从原FM第6000步继续3000步。fm/variance/correlation/joint/cross分别是纯FM、方差项、相关项、原联合平方差、独立生成子集交叉估计。固定相同弱权重、8条生成、32步可微采样；不改变参考物理、网络或输出网格。</p>'
    body+='<p>cross去掉生成统计估计量方差在平方损失期望中的额外项；只相对于固定经验参考及其归一化成立。训练参考仍只有8条，分成4+4可能增加梯度噪声；单批loss允许负值。不是完全无偏的物理分布损失。</p>'
    body+='<p>fit1/fit6只对固定1/6个L16训练父状态的8条已有路径做FM续训。fit是这些重复训练初态，train是18个训练校准初态，dev是36个独立开发初态。fit成功只能说明经验样本拟合，不能说明新热噪声路径已正确；本预算下失败也不能证明架构不可能拟合。</p>'
    body+=table(['索引','seed','方案','集合','父状态数','增量方差RMSE','两时关联RMSE','能量均值RMSE','空间RMSE'],overview)
    body+='<p>每父状态64条生成；训练参考8条、开发参考16条；评估128flow步，首个训练/开发父状态另比较32/256步。100次整分支bootstrap，逐点区间不等于等价性判定。比较未优化的磁化、能量、空间指标；没有自动综合通过分数。</p>'
    body+='<details><summary>完整分项结果、约束误差与固定插值探针</summary><pre>'+html.escape(json.dumps(results,ensure_ascii=False,indent=2))+'</pre></details>'
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1300px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>'
    (assets/'index.html').write_text(page)
    dashboard=ROOT/'tests/index.html'
    if dashboard.exists():
        text=dashboard.read_text();start='<!-- moment-bias-repair:start -->';end='<!-- moment-bias-repair:end -->'
        block=start+'<section id="moment-bias-repair">'+body+'</section>'+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,text,flags=re.S)
        elif '</body>' in text:text=text.replace('</body>',block+'</body>')
        else:text+=block
        temp=dashboard.with_suffix('.html.tmp');temp.write_text(text);temp.replace(dashboard)
    if errors or missing:raise RuntimeError('incomplete experiment; see summary.json')


if __name__=='__main__':main()
