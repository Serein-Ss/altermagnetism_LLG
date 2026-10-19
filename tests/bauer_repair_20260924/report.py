"""Versioned erratum, repair comparisons and self-contained PNG dashboard."""
import argparse
import base64
import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
import numpy as np
from .run import HERE, ROOT, dump, mixture_samples
from .model import ARMS
from tests.bauer_comprehensive_20260923.report import finite_sample
from tests.bauer_p2_20260923.common import comparison
from tests.bauer_p2_diagnostics_20260923.run import table

ERRATUM='上一轮 local_extra 的值向量平行于当前磁矩；其叉积和切向投影理论上为零。因此它不是有效容量对照，相关归因无效。37项旧软件测试未覆盖分支非退化性。旧记录保留，本轮用 local_fixed 替代；这不解释原模型此前全部物理偏差。'


def directory(index,task):
    return HERE/'runs'/f'{index:02d}_{task["domain"]}_{task["arm"]}_{task["seed"]}'


def figures(runs):
    if not runs:return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    assets=HERE/'assets';paths=[];colors=['#0072B2','#D55E00','#009E73']
    for domain in ('bauer','mixture'):
        subset=[r for r in runs if r['task']['domain']==domain]
        if not subset:continue
        fig,axs=plt.subplots(2,3,figsize=(13,7),constrained_layout=True,sharex=True,sharey=True)
        for ax,arm in zip(axs.flat,ARMS):
            arm_runs=sorted([r for r in subset if r['task']['arm']==arm],key=lambda r:r['task']['seed'])
            ax.set_title(arm)
            if not arm_runs:ax.text(.5,.5,'No completed result',ha='center',transform=ax.transAxes);continue
            for i,row in enumerate(arm_runs):
                out=directory(row['index'],row['task'])
                if domain=='bauer':
                    parents=json.loads((out/'dev_final128.json').read_text())['parents']
                    gen=np.mean([p['diagnostic']['increment_variance']['generated'] for p in parents],0)
                    ref=np.mean([p['diagnostic']['increment_variance']['reference'] for p in parents],0)
                    t=np.arange(1,11)*.1
                    if i==0:ax.plot(t,ref,color='black',lw=2.5,label='LLG reference')
                    ax.plot(t,gen,color=colors[i],marker='o',ms=3,label=f'seed {row["task"]["seed"]}')
                    ax.set_xlabel('Interval end time (ps)');ax.set_ylabel('Conditional increment variance')
                else:
                    gs=[];rs=[]
                    for file in sorted((out/'generated_dev_128').glob('*.npz')):
                        with np.load(file) as z:
                            gs.extend(mixture_samples(z['generated'])['signed_rotation'])
                            rs.extend(mixture_samples(z['reference'])['signed_rotation'])
                    edges=np.linspace(-1,1,41)
                    if i==0:ax.hist(rs,bins=edges,density=True,color='#b9b9b9',alpha=.6,label='Analytic-law reference')
                    hist,_=np.histogram(gs,bins=edges,density=True)
                    ax.stairs(hist,edges,color=colors[i],lw=1.7,label=f'seed {row["task"]["seed"]}')
                    ax.set_xlabel('Mean signed rotation score');ax.set_ylabel('Probability density')
                ax.legend(fontsize=8)
            ax.grid(alpha=.15)
        fig.suptitle('Bauer: fixed-parent averages, separate training seeds' if domain=='bauer' else 'Synthetic mixture: endpoint signed-score distributions')
        path=assets/f'{domain}_repair.png';fig.savefig(path,dpi=180,facecolor='white');plt.close(fig);paths.append(path)
    return paths


def report():
    protocol=json.loads((HERE/'protocol.json').read_text());assets=HERE/'assets';assets.mkdir(exist_ok=True)
    parts=['<h1>2026-09-24：有效残差与共同随机模式修复验证</h1>',
        '<aside style="border:2px solid #b23a31;padding:16px;background:#fff3ed"><strong>旧对照勘误</strong><p>'+ERRATUM+'</p></aside>',
        '<p>沿用相同Bauer开放边界、单温度、L16/25/32、1 ps与0.1 ps网格。新方案只改变网络残差和运输潜变量相关性，不改变参考LLG、损失、初态制备和数据划分。合成双峰路径不是LLG，也没有ps单位。当前全部为开发验证，不能直接声称大尺寸迁移或物理通过。</p>',
        table(['方案','受控变化'],[[a,html.escape(v)] for a,v in protocol['controls'].items()]),
        '<p>所有方案都用normalize(2*S0+noise)的相同单点潜变量分布。共享方案只改变分支内部时空相关性，各独立分支的随机流不同；它不是LLG热噪声，也不是学习式初态生成器。候选均不输入Hamiltonian或已知双峰标签。</p>']
    tests=HERE/'software_tests.xml'
    if tests.exists():
        suites=ET.parse(tests).getroot().findall('.//testsuite')
        parts.append('<p>软件检查：'+str(sum(int(s.get('tests',0)) for s in suites))+'项；失败/错误数：'+str(sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites))+'。另需训练后的物理统计检查。</p>')
    benchmark=HERE/'benchmark.json'
    if benchmark.exists():
        b=json.loads(benchmark.read_text())
        parts.append('<h2>GPU非退化性预检</h2><p>下列为60次预检更新后的量，不是训练完成结果。</p>'+table(['方案','残差RMS','每更新秒'],[[r['arm'],f'{r["residual_rms"]:.5f}',f'{r["seconds_per_update"]:.4f}'] for r in b['rows']]))
    runs=[];missing=[];keys=['energy_mean','energy_var','temporal','temporal_lag','spatial','increment_cov']
    for index,task in enumerate(protocol['tasks']):
        out=directory(index,task)
        if not (out/'complete.json').exists():missing.append(dict(index=index,**task));continue
        complete=json.loads((out/'complete.json').read_text());final=json.loads((out/'dev_final128.json').read_text())
        history=json.loads((out/'history.json').read_text());tail=history[-500:]
        calibration=[];convergence=[]
        for j,file in enumerate(sorted((out/'generated_dev_128').glob('*.npz'))):
            with np.load(file) as z:g=z['generated'];r=z['reference']
            calibration.append(dict(parent_id=file.stem,**finite_sample(g,r,92410+j)))
            with np.load(out/'generated_dev_256'/file.name) as z:hi=z['generated']
            convergence.append(dict(parent_id=file.stem,paired_path_rmse=float(np.sqrt(np.mean((g[:16]-hi)**2))),statistics_change=comparison(g[:16],hi)))
        dump(out/'matched_sample_calibration.json',calibration);dump(out/'sampler_common_source.json',convergence)
        mixture=None
        if task['domain']=='mixture':
            mixture={k:{field:float(np.mean([r['mixture']['0.1'][k][field] for r in final['parents']]))
                        for field in ('generated','reference','difference')} for k in final['parents'][0]['mixture']['0.1']}
        row=dict(index=index,task=task,complete=complete,metrics=final['mean_parent_RMSE'],mixture=mixture,
            residual_rms_last500=float(np.mean([q['residual_rms'] for q in tail])),
            first_increment_variance={v:float(np.mean([p['diagnostic']['increment_variance'][v][0] for p in final['parents']])) for v in ('generated','reference')})
        runs.append(row)
    parts.append(f'<h2>完成状态：{len(runs)}/{len(protocol["tasks"])}</h2><p>缺少完成标记的任务：{html.escape(str([r["index"] for r in missing]))}。缺失不是物理失败或通过。</p>')
    if runs:
        parts.append(table(['数据','方案','seed','更新数','耗时分钟','残差RMS']+keys,[[r['task']['domain'],r['task']['arm'],r['task']['seed'],r['complete']['step'],f'{r["complete"]["seconds"]/60:.2f}',f'{r["residual_rms_last500"]:.5f}']+[f'{r["metrics"][k]:.5f}' for k in keys] for r in runs]))
    pairing=[];contrasts=[]
    for domain in ('bauer','mixture'):
        for seed in (11,22,33):
            group={r['task']['arm']:r for r in runs if r['task']['domain']==domain and r['task']['seed']==seed}
            pairing.append(dict(domain=domain,seed=seed,completed=len(group),recorded_streams_equal=len({r['complete']['sample_stream_sha256'] for r in group.values()})<=1))
            pairs=[('baseline','local_fixed'),('local_fixed','temporal_fixed'),('temporal_fixed','relational_global'),('temporal_fixed','temporal_shared'),('relational_global','relational_shared')]
            for a,b in pairs:
                if a in group and b in group:
                    contrasts.append(dict(domain=domain,seed=seed,contrast=a+' -> '+b,
                        delta={k:group[b]['metrics'][k]-group[a]['metrics'][k] for k in keys}))
            if all(k in group for k in ('temporal_fixed','relational_global','temporal_shared','relational_shared')):
                contrasts.append(dict(domain=domain,seed=seed,contrast='relational x shared interaction',delta={k:
                    group['relational_shared']['metrics'][k]-group['relational_global']['metrics'][k]-group['temporal_shared']['metrics'][k]+group['temporal_fixed']['metrics'][k] for k in keys}))
    parts.append('<h2>共同数据/随机流与配对对照</h2>'+table(['数据','seed','已完成组数','记录流一致'],[[p['domain'],p['seed'],p['completed'],p['recorded_streams_equal']] for p in pairing]))
    if contrasts:
        parts.append('<p>以下为候选减对照的RMSE，负值是该项改善；交互项为差上差，不等于显著性检验。</p>'+table(['数据','seed','对照']+keys,[[r['domain'],r['seed'],r['contrast']]+[f'{r["delta"][k]:+.5f}' for k in keys] for r in contrasts]))
    mixture_rows=[r for r in runs if r['mixture']]
    if mixture_rows:
        mk=['positive_fraction','direction_coherence','rotation_amplitude','direction_persistence']
        parts.append('<h2>合成双峰：概率与整条路径协调</h2><p>各格点不是独立重复；JSON中的区间按完整分支重采样。下表各格显示生成/参考，不能只看正向比例而忽略模式振幅和协调性。</p>'+table(['方案','seed']+mk,[[r['task']['arm'],r['task']['seed']]+[f'{r["mixture"][k]["generated"]:.4f} / {r["mixture"][k]["reference"]:.4f}' for k in mk] for r in mixture_rows]))
    for file in figures(runs):
        caption='Bauer图：每格对应一个方案，黑线为相同父状态的LLG参考，彩线为三个训练seed；纵轴是分支间增量方差，非磁矩模长。' if file.stem.startswith('bauer') else '双峰图：灰色填充为参考分布，彩色阶梯线为各seed的模型分布；横轴为连续旋转分数，可同时看模式位置、宽度和权重。固定父状态等权混合展示，不是自然初态占据率。'
        parts.append('<figure><img style="width:100%;max-width:1400px" src="data:image/png;base64,'+base64.b64encode(file.read_bytes()).decode()+'"><figcaption>'+caption+'</figcaption></figure>')
    parts.append('<p>逐父状态结果、完整生成路径、两时关联矩阵、点态bootstrap区间、阈值敏感性、checkpoint和随机流在本目录各runs中保存。没有新最终留出集或预先确认的物理容差，不自动宣布认证。能量与时空统计必须同时检查；降低FM损失、非零梯度或模式图变好都不单独证明Bauer已修复。</p>')
    body=''.join(parts);style='<style>body{font:15px/1.65 sans-serif;margin:24px}table{display:block;overflow:auto;border-collapse:collapse}th,td{border:1px solid #bbc;padding:7px}th{background:#e6edf4}figure{margin:20px 0}</style>'
    (assets/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Bauer修复验证</title>'+style+body)
    dump(assets/'summary.json',dict(runs=runs,missing=missing,pairing=pairing,contrasts=contrasts,erratum=ERRATUM,status='DEVELOPMENT_NOT_CERTIFIED'))
    main=ROOT/'tests/index.html'
    if main.exists():
        text=main.read_text();start='<!-- repair-20260924-start -->';end='<!-- repair-20260924-end -->'
        if start in text:text=text[:text.index(start)]+text[text.index(end)+len(end):]
        section=start+'<section id="bauer-repair"><details open><summary>最新修复验证与旧对照勘误</summary>'+body+'</details></section>'+end
        text=re.sub(r'(<main\b[^>]*>)',lambda m:m.group(1)+section,text,count=1)
        main.write_text(text)
        snapshot=ROOT/'tests/progress_dashboard/snapshot.json'
        if snapshot.exists():
            state=json.loads(snapshot.read_text());state['sources'][str((assets/'index.html').relative_to(ROOT))]=hashlib.sha256((assets/'index.html').read_bytes()).hexdigest();dump(snapshot,state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.parse_args();report()
