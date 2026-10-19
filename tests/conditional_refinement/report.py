"""Family/size resolved diagnostic, preserving all historical evidence."""
import html
import json
import re
import numpy as np
import torch
from tests.moment_bias_repair.report import analyze
from tests.transport_audit.report import table
from .run import HERE,ROOT,tasks,dump

KEYS=[('diagnostic','increment_variance'),('diagnostic','two_time_spin_correlation'),
      ('diagnostic','increment_cross_covariance'),('physical','energy_mean'),('physical','mz_mean'),('physical','spatial')]


def metrics(parents):
    return [round(float(np.mean([p[typ][key]['rmse'] for p in parents])),6) for typ,key in KEYS]


def main():
    torch.set_num_threads(1);assets=HERE/'assets';assets.mkdir(exist_ok=True)
    results=[];missing=[];errors=[];overview=[];families=[];sizes=[];probe_rows=[];signs=[]
    for i,task in enumerate(tasks()):
        out=HERE/f'train/{i:02d}'
        if not (out/'complete.json').exists():missing.append(i);continue
        try:
            info=json.loads((out/'complete.json').read_text());parents=analyze(out)
            dev=[p for p in parents if p['split']=='dev' and p['group'].endswith('steps128')]
            assert len(dev)==36
            assert all(p['norm_error']<2e-6 and p['anchor_error']<2e-6 for p in parents)
            dump(out/'diagnostics.json',dict(reference_kind='population',parents=parents))
            for split in ('train','dev'):
                ps=[p for p in parents if p['split']==split and p['group'].endswith('steps128')]
                overview.append([task['seed'],task['repair'],split,*metrics(ps)])
            for family in sorted({p['parent']['family'] for p in dev}):
                families.append([task['seed'],task['repair'],family,*metrics([p for p in dev if p['parent']['family']==family])])
            for size in (16,25,32):
                sizes.append([task['seed'],task['repair'],size,*metrics([p for p in dev if p['parent']['L']==size])])
            var=np.array([p['diagnostic']['increment_variance']['difference'] for p in dev])
            corr=np.array([p['diagnostic']['two_time_spin_correlation']['difference'] for p in dev])
            signs.append([task['seed'],task['repair'],float(var[:,0].mean()),float(var[:,-1].mean()),float(corr[:,0,-1].mean())])
            probes=json.loads((out/'transport_probes.json').read_text())
            for split in ('train','dev'):
                for family in sorted({p['parent']['family'] for p in probes['before'] if p['split']==split}):
                    for tau in (.02,.1,.25,.5,.9,.98):
                        vals=[]
                        for stage in ('before','after'):
                            selected=[p for p in probes[stage] if p['split']==split and p['parent']['family']==family and p['tau']==tau]
                            vals.append(float(np.mean([p['mse_by_physical_frame'] for p in selected])))
                        probe_rows.append([task['seed'],task['repair'],split,family,tau,*vals])
            results.append(dict(index=i,task=task,run=info,parents=parents))
        except Exception as exc:errors.append(dict(index=i,error=repr(exc)))
    for seed in (11,22,33):
        pair=[r['run']['prefix_data_source_tau_hash'] for r in results if r['task']['seed']==seed and r['task']['repair'] in ('control','accum4')]
        if len(set(pair))>1:errors.append(dict(seed=seed,error='Control/accumulation microbatch prefix mismatch'))
    previous=json.loads((ROOT/'tests/full_data_decay/assets/summary.json').read_text())['results']
    for r in previous:
        if r['task']['repair']=='decay':
            dev=[p for p in r['parents'] if p['split']=='dev' and p['group'].endswith('steps128')]
            overview.append([r['task']['seed'],'起点step12000','dev',*metrics(dev)])
    dump(assets/'summary.json',dict(results=results,missing=missing,errors=errors,overview=overview,
        family=families,size=sizes,transport_probes=probe_rows,temporal_signed_bias=signs,scientific_certification=False))
    body=f'<h1>父状态起点覆盖与梯度平均修复对照</h1><p>完成{len(results)}/9；缺失{missing}；错误{html.escape(str(errors))}。这是开发诊断，不是统计认证。</p>'
    body+='<p>三seed分别从已完成的衰减方案step12000恢复，续训6000次更新，统一余弦学习率1e-4→3e-6。control保持原抽样；anchor90将强制起点抽样概率从0.5增至0.9；accum4平均四个微批梯度，训练曝光与计算约四倍，不能将效果单独归因于梯度噪声或称为等成本比较。网络、物理条件、1ps时间窗和0.1ps网格不变。</p>'
    headers=['seed','方案','集合/分类','增量方差RMSE','两时关联RMSE','增量交叉协方差RMSE','能量均值RMSE','磁化均值RMSE','空间RMSE']
    body+='<h2>总体与起点模型比较</h2>'+table(headers,overview)
    body+='<h2>开发父状态类别</h2>'+table(headers,families)
    body+='<h2>开发尺寸分项（均为训练支持尺寸，不是尺寸外推）</h2>'+table(headers,sizes)
    body+='<h2>时间结构的有符号偏差：生成减参考</h2>'+table(['seed','方案','首段增量方差差','末段增量方差差','初末态关联差'],signs)
    body+='<p>以下CFM残差对照使用相同参考路径、潜变量及运输时刻；包含随机监督目标的不可约波动，不能称为未知总体真实速度场的精确误差。运输时间tau不等于物理时间，未使用真实未来作为额外生成条件。</p>'
    body+='<details><summary>按类别与运输阶段的训练前后残差</summary>'+table(['seed','方案','集合','类别','tau','续训前CFM残差','续训后CFM残差'],probe_rows)+'</details>'
    audit=json.loads((HERE/'data_audit.json').read_text())
    body+='<details><summary>训练父状态与窗口曝光审计（数量不证明充分覆盖）</summary><pre>'+html.escape(json.dumps(audit,ensure_ascii=False,indent=2))+'</pre></details>'
    body+='<p>各父状态64生成分支、16开发参考分支；总体统计使用N−1方差估计；100次整分支bootstrap在完整结果内。首个训练/开发父状态含32/128/256步对照。参考有限、开发集反复使用，区间含零不证明等价；没有自动通过阈值或综合得分。</p>'
    body+='<details><summary>完整物理指标、区间、采样步数和约束检查</summary><pre>'+html.escape(json.dumps(results,ensure_ascii=False,indent=2))+'</pre></details>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1400px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    dashboard=ROOT/'tests/index.html'
    if dashboard.exists():
        text=dashboard.read_text();start='<!-- conditional-refinement:start -->';end='<!-- conditional-refinement:end -->'
        block=start+'<section id="conditional-refinement">'+body+'</section>'+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,text,flags=re.S)
        else:text=text.replace('</body>',block+'</body>') if '</body>' in text else text+block
        temp=dashboard.with_suffix('.html.tmp');temp.write_text(text);temp.replace(dashboard)
    if errors or missing:raise RuntimeError('Incomplete campaign; see summary.json')


if __name__=='__main__':main()
