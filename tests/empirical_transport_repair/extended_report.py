"""Supplementary supervised-field and independent-latent checks."""
import html
import json
import re
from .run import HERE,ROOT,dump
from .report import main as base_report
from .supervision import tasks
from tests.transport_audit.report import table


def main():
    errors=[]
    try:base_report()
    except Exception as exc:errors.append(repr(exc))
    results=[];missing=[];overview=[];replicates=[]
    for i,t in enumerate(tasks()):
        p=HERE/f'supervision_train/{i:02d}/complete.json'
        if not p.exists():missing.append(f'supervision{i}');continue
        r=json.loads(p.read_text());results.append(dict(index=i,**r))
        for x in r['rows']:
            if x['start_tau']==0 and x['steps']==128 and x['stop_tau']==.95:
                overview.append([t['seed'],t['repair'],round(x['nearest_path_mse'],6),round(x['diagnostic']['increment_variance']['rmse'],6),round(x['diagnostic']['two_time_spin_correlation']['rmse'],6)])
    for seed in (11,22,33):
        hashes=[r['first32_input_hash'] for r in results if r['task']['seed']==seed]
        if len(set(hashes))>1:errors.append(f'seed{seed} supervision inputs mismatch')
    for i in range(3):
        p=HERE/f'resample/{i:02d}/complete.json'
        if not p.exists():missing.append(f'resample{i}');continue
        r=json.loads(p.read_text());replicates.append(r)
        if r['missing']:errors.append(f'resample{i} missing checkpoints')
    assets=HERE/'assets';assets.mkdir(exist_ok=True)
    dump(assets/'extended_summary.json',dict(supervision=results,resampling=replicates,missing=missing,errors=errors,scientific_certification=False))
    body='<h1>补充：监督噪声与潜变量重复采样</h1><p>paired_cfm和exact_velocity均从头训练，相同输入和余弦学习率，运输训练范围均为[0,0.95)。区别仅在随机配对速度目标与8条经验路径的精确条件平均速度。不得直接把该组与原[0,1)训练的差异全归因于监督。</p>'
    body+=f'<p>监督完成 {len(results)}/6；重复采样完成 {len(replicates)}/3；缺失 {missing}；错误 {html.escape(str(errors))}。</p>'
    body+=table(['seed','监督','最近路径MSE','增量方差RMSE','两时关联RMSE'],overview)
    body+='<p>补充训练输出中tau1属于超出训练运输范围的探索，不作为成功判据。每个模型另用3组独立潜变量，各512条，检查抽样波动，并在第一组比较128/512步。它们仍共享同一组8条LLG经验参考，不是新增物理参考数据。</p>'
    for title,data in [('监督对照详情',results),('独立潜变量重复与采样收敛',replicates)]:
        body+='<details><summary>'+title+'</summary><pre>'+html.escape(json.dumps(data,ensure_ascii=False,indent=2))+'</pre></details>'
    (assets/'extended.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body>'+body+'</body></html>')
    p=ROOT/'tests/index.html'
    if p.exists():
        text=p.read_text();start='<!-- empirical-transport-extended:start -->';end='<!-- empirical-transport-extended:end -->'
        block=start+'<section id="empirical-transport-extended">'+body+'</section>'+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,text,flags=re.S)
        elif '</body>' in text:text=text.replace('</body>',block+'</body>')
        else:text+=block
        temp=p.with_suffix('.html.tmp');temp.write_text(text);temp.replace(p)
    if missing or errors:raise RuntimeError('incomplete extended experiment')


if __name__=='__main__':main()
