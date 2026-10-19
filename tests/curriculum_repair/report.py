"""Whole-branch physical evaluation; missing runs are never counted as passes."""
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


def main():
    torch.set_num_threads(1);assets=HERE/'assets';assets.mkdir(exist_ok=True)
    rows=[];results=[];missing=[];errors=[]
    for i,task in enumerate(tasks()):
        out=HERE/f'train/{i:02d}'
        if not (out/'complete.json').exists():missing.append(i);continue
        try:
            info=json.loads((out/'complete.json').read_text());parents=[]
            with h5py.File(out/'paths.h5') as h:
                assert h.attrs['complete']
                for name,g in h.items():
                    x=g['generated'][:];r=g['reference'][:]
                    parents.append(dict(group=name,parent=json.loads(g.attrs['parent']),
                        physical=comparison(x,r,repeats=100,seed=92640+i),
                        diagnostic=diagnostic_comparison(x,r,100,92640+i),
                        norm_error=float(np.max(abs(np.linalg.norm(x,axis=-1)-1))),
                        anchor_error=float(np.max(abs(x[:,0]-r[:1,0])))))
            selected=[p for p in parents if p['group'].endswith('steps128')]
            assert len(selected)==36
            means={k:float(np.mean([p['diagnostic'][k]['rmse'] for p in selected])) for k in ('increment_variance','two_time_spin_correlation')}
            rows.append([i,task['seed'],task['repair'],round(info['train_seconds'],1),*[round(v,6) for v in means.values()]])
            results.append(dict(index=i,task=task,run=info,parents=parents,mean_diagnostic_rmse=means))
            dump(out/'diagnostics.json',parents)
        except Exception as exc:errors.append(dict(index=i,error=repr(exc)))
    pairing={}
    for seed in (11,22,33):
        hashes=[r['run']['first32_data_source_hash'] for r in results if r['task']['seed']==seed]
        pairing[str(seed)]=dict(count=len(hashes),identical=len(set(hashes))==1 if hashes else None)
        if len(set(hashes))>1:errors.append(dict(seed=seed,error='FM streams mismatch'))
    gradients=[]
    for i in range(3):
        path=HERE/f'diagnose/{i:02d}/gradients.json'
        if path.exists():gradients.append(dict(seed=(11,22,33)[i],rows=json.loads(path.read_text())))
    summary=dict(results=results,missing=missing,errors=errors,pairing=pairing,gradients=gradients,scientific_certification=False)
    dump(assets/'summary.json',summary)
    body='<h1>弱统计约束与渐进训练：受控续训</h1><p>从相同seed的纯FM第6000步恢复模型、优化器和随机流，再训练3000步。开放Bauer，1ps，0.1ps网格；未改变参考物理或网络。</p>'
    body+='<p>fm：纯FM；fixed32：恒定弱辅助；ramp32：先500步FM，再1000步升至0.1；ramp64：相同调度，辅助采样64步。固定与渐进的累计权重匹配，但算力和实际梯度不等价。</p>'
    body+=f'<p>完成 {len(results)}/12；缺失 {missing}；错误 {html.escape(str(errors))}。这些是开发诊断，不是物理认证。</p>'
    body+=table(['索引','seed','方案','训练秒','增量方差RMSE','完整两时相关RMSE'],rows)
    body+='<p>下方完整数据包括逐父状态能量、磁化、空间统计、增量、时间关联及100次独立整分支bootstrap；区间为逐点区间，不是等价性判定。六个训练L16父状态上的梯度诊断不代表完整训练总体，也不用于伪装独立留出验证。</p>'
    for label,data in [('随机流配对检查',pairing),('梯度诊断',gradients),('完整模型结果',results)]:
        body+='<details><summary>'+label+'</summary><pre>'+html.escape(json.dumps(data,ensure_ascii=False,indent=2))+'</pre></details>'
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1300px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>'
    (assets/'index.html').write_text(page)
    dashboard=ROOT/'tests/index.html'
    if dashboard.exists():
        text=dashboard.read_text();start='<!-- curriculum-repair:start -->';end='<!-- curriculum-repair:end -->'
        block=start+'<section id="curriculum-repair">'+body+'</section>'+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,text,flags=re.S)
        elif '</body>' in text:text=text.replace('</body>',block+'</body>')
        else:text+=block
        temp=dashboard.with_suffix('.html.tmp');temp.write_text(text);temp.replace(dashboard)
    if errors or missing:raise RuntimeError('incomplete experiment; see summary.json')


if __name__=='__main__':main()
