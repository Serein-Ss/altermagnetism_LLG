"""Full physical reports plus compact stratified tables, never automatic certification."""
import argparse
import json
import re
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tests.transition_block_repair.report import collect
from tests.transition_block_repair.metrics import compare_first
from tests.reference_resolution.report import brief
from tests.conditional_moment_repair.report import metrics
from tests.transport_audit.report import table
from tests.training_restart_repair.data import rows, HERE as DATA
from .run import HERE, ROOT, ARMS, tasks, source, dump, verify


def analyze(index, smoke):
    folder=HERE/('smoke' if smoke else 'train')/f'{index:02d}'
    run=json.loads((folder/'complete.json').read_text());assert run['task']==tasks()[index]
    data=collect(folder/'paths.h5',smoke);middle=[];seen=[]
    with h5py.File(folder/'intermediate.h5') as h:
        assert h.attrs['complete'] and len(h)==(1 if smoke else 72)
        for name,g in h.items():
            meta=json.loads(g.attrs['metadata']);x=g['spins'][:]
            with h5py.File(ROOT/f"tests/intermediate_restart_repair/reference/{meta['parent_index']:02d}/paths.h5") as ref:
                assert ref.attrs['complete']
                values={k:compare_first(x,ref[f'{name}/{k}/spins'][:]) for k in ('base','fine')}
            middle.append(dict(metadata=meta,metrics=values))
    for phase,path in (('historical_control',source(run['task']['seed'])/'after_restart.h5'),('after',folder/'after_restart.h5')):
        with h5py.File(path) as h:
            assert h.attrs['complete']
            for row in rows(smoke):
                name=f"state{row['index']:03d}";x=h[f'{name}/spins'][:];refs={}
                assert json.loads(h[name].attrs['metadata'])==row
                for role in ('probe','fine'):
                    with h5py.File(DATA/'reference'/f"{row['parent_index']:02d}"/f'{name}_{role}.h5') as ref:
                        assert ref.attrs['complete'] and ref.attrs['role']==role
                        refs[role]=ref['spins'][:]
                seen.append(dict(phase=phase,metadata=row,metrics={k:compare_first(x,v) for k,v in refs.items()},
                                 reference_control=compare_first(refs['probe'],refs['fine'])))
    dump(folder/'report.json',dict(run=run,data=data,intermediate=middle,restart_seen=seen,symmetry=json.loads((folder/'symmetry.json').read_text()),scientific_certification=False))


def grouped(records, keys, value):
    result=[]
    groups=sorted({tuple(keys(x)) for x in records})
    for group in groups:
        selected=[x for x in records if tuple(keys(x))==group]
        result.append(list(group)+np.mean([value(x) for x in selected],axis=0).tolist())
    return result


def aggregate(smoke):
    first=[];full=[];middle=[];seen=[];overview=[];runs=[];exposures=[];missing=[];errors=[];links=[];symmetry=[]
    for i in range(2 if smoke else 6):
        folder=HERE/('smoke' if smoke else 'train')/f'{i:02d}';path=folder/'report.json'
        if not path.exists():missing.append(i);continue
        r=json.loads(path.read_text());run=r['run'];runs.append(run);t=run['task'];label=[t['seed'],t['repair']]
        assert run['optimizer_reset'] and run['optimizer_steps_by_group']==[[16 if smoke else 12000]]
        links.append(str(path.relative_to(HERE)))
        for role,key in (('seen','probe'),('dev','fine')):
            ss=[x for x in r['data']['rows'] if x['role']==role]
            keys=lambda x:[role,x['parent']['L'],x['parent']['family'],x['sampler']]
            first += [label+x for x in grouped(ss,keys,lambda x:metrics(x['first'][key]))]
            full += [label+x for x in grouped(ss,keys,lambda x:brief(x['full'][key]))]
        middle += [label+x for x in grouped(r['intermediate'],
            lambda x:[x['metadata']['parent']['L'],x['metadata']['parent']['family'],x['metadata']['origin_time_ps']],
            lambda x:metrics(x['metrics']['fine']))]
        seen += [label+x for x in grouped(r['restart_seen'],
            lambda x:[x['phase'],x['metadata']['parent']['family'],x['metadata']['source_time_ps']],
            lambda x:metrics(x['metrics']['fine']))]
        overview.append(label+np.mean([metrics(x['metrics']['fine']) for x in r['intermediate']],axis=0).tolist())
        counts=np.array(json.loads((folder/'exposures.json').read_text()))
        total=16 if smoke else 12000
        assert counts.shape==(48,64) and counts.sum()==total*48 and (counts.sum(1)==total).all()
        exposures.append(label+[int(counts.sum()),int(counts.sum(1).min()),int(counts.sum(1).max()),int((counts==0).sum()),run['train_seconds'],run['evaluation_seconds']])
        symmetry += [label+[x['family'],x['tau'],x['angle'],x['absolute_rms'],x['relative_rms']] for x in r['symmetry']]
        del r
    for seed in ((11,) if smoke else (11,22,33)):
        group=[r for r in runs if r['task']['seed']==seed]
        for key in ('paired_input_hash','schedule_hash','group_schedule_hash'):
            if len({r[key] for r in group})!=1:errors.append([seed,key])
        dense=group
        if len({r['extra_noise_hash'] for r in dense})!=1:errors.append([seed,'dense extra noise'])
    assets=HERE/('smoke_assets' if smoke else 'assets');assets.mkdir(exist_ok=True)
    dump(assets/'summary.json',dict(first=first,full=full,intermediate=middle,restart_seen=seen,overview=overview,
        exposure_columns=['seed','arm','total','min_per_state','max_per_state','unused_state_branch_pairs','train_seconds','evaluation_seconds'],
        symmetry=symmetry,exposures=exposures,runs=runs,reports=links,missing=missing,errors=errors,scientific_certification=False))
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for k,(ax,title) in enumerate(zip(axes.flat,('Increment mean RMSE','Increment covariance RMSE','Signed variance bias','Energy mean RMSE (H/NJ)'))):
        for j,arm in enumerate(ARMS):
            values=[x[2+k] for x in overview if x[1]==arm]
            if values:
                ax.bar(j,np.mean(values),color=('#0072B2','#E69F00','#009E73')[j],edgecolor='black',width=.65)
                ax.scatter(j+np.linspace(-.1,.1,len(values)),values,color='black',s=22,zorder=3)
        ax.axhline(0,color='gray',lw=.7);ax.set_title(title);ax.set_xticks(range(2),['Warm reset','Fresh training'])
    fig.suptitle('DEV intermediate states: descriptive means; dots = training seeds'+(' (SMOKE ONLY)' if smoke else ''))
    fig.savefig(assets/'dev_comparison.png',dpi=180);plt.close(fig)
    body=f'<h1>当前数据上的完整重新训练对照</h1><p>完成{len(runs)}，缺失{missing}，配对错误{errors}。'+('软件预检。' if smoke else '开发诊断。')+'</p>'
    body+='<p>原4块hierarchical网络，当前native数据及48TRAIN中间状态64未来分支。warm组从60144步原深度control权重重新启动，fresh组随机初始化。两组均新Adam，200步热身到1e-3，随后余弦下降到3e-5，共12000步。相同数据、增强和flow随机输入。最终12000步固定用于物理评价，中途4000/8000步仅作学习过程诊断，不根据开发结果选取最好checkpoint。</p>'
    body+='<p>历史原始训练没有当前补齐的48状态/64分支监督；旧512步重置压力测试也不等于这次完整重新训练。测试训练历史和当前数据联合配方，不能以任一候选失败断言所有流模型或磁链物理不可能。fresh组的step是12000，warm组累计模型更新到72144，但两组新优化器都运行12000步。</p>'
    learning=[]
    for r in runs:
        folder=HERE/('smoke' if smoke else 'train')/f"{tasks().index(r['task']):02d}"
        rr=json.loads((folder/'restart_risk.json').read_text())
        for stage in rr:
            for k in sorted({x['bin'] for x in stage['rows']}):
                vals=np.mean([x['values'] for x in stage['rows'] if x['bin']==k],axis=0)
                learning.append([r['task']['seed'],r['task']['repair'],stage['updates'],k,*vals.tolist()])
    dump(assets/'learning_diagnostics.json',dict(rows=learning,columns=['seed','arm','updates','flow_time_bin','hybrid_teacher_risk','sampled_risk','residual_power','cross_term','field_power']))
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for j,arm in enumerate(ARMS):
        for seed in (11,22,33):
            rr=[x for x in learning if x[0]==seed and x[1]==arm]
            steps=sorted({x[2] for x in rr})
            if steps:ax.plot(steps,[np.mean([x[4] for x in rr if x[2]==step]) for step in steps],marker='o',color=('#0072B2','#E69F00')[j],alpha=.7,label=arm if seed==11 else None)
    ax.set_xlabel('New optimizer updates');ax.set_ylabel('Empirical restart teacher risk');ax.legend()
    fig.savefig(assets/'learning_curve.png',dpi=180);plt.close(fig)
    body+='<h2>是否重新学到了不同的解</h2><img src="learning_curve.png" style="max-width:100%"><p>曲线为当前TRAIN经验库的辅助运输风险，不能代替独立物理准确性。全部64分支覆盖每个tau分层，旋转增强，固定新latent；只诊断restart部分，native目标另见history.json。</p>'
    body+=table(['seed','方案','新更新数','flow时间层','教师风险','采样标签风险','标签残差','交叉项','场功率'],learning)
    body+='<img src="dev_comparison.png" style="max-width:100%"><p>柱为描述性种子均值，点为训练种子；方差偏差应接近零，其他RMSE越小越好。能量单位H/(NJ)。按构型、尺寸和物理时间逐项检查，不能用平均值掩盖退化。</p>'
    body+='<h2>历史control基准与重新训练结果：独立噪声后继</h2>'+table(['seed','方案','阶段','构型','起点ps','均值RMSE','协方差RMSE','方差偏差','能量均值RMSE'],seen)
    body+='<h2>开发中间构型</h2>'+table(['seed','方案','L','构型','起点ps','均值RMSE','协方差RMSE','方差偏差','能量均值RMSE'],middle)
    body+='<h2>原始初态</h2>'+table(['seed','方案','角色','L','构型','采样','均值RMSE','协方差RMSE','方差偏差','能量均值RMSE'],first)
    body+='<h2>完整1ps时间组织</h2>'+table(['seed','方案','角色','L','构型','采样','增量方差RMSE','两时关联RMSE','能量均值RMSE'],full)
    body+='<h2>参数与成本</h2>'+table(['seed','方案','参数量','训练秒','评估秒'],[[r['task']['seed'],r['task']['repair'],r['parameter_count'],r['train_seconds'],r['evaluation_seconds']] for r in runs])
    body+='<h2>曝光与配对</h2>'+table(['seed','方案','总曝光','每状态最少','每状态最多','未用状态分支对','训练秒','评估秒'],exposures)
    body+='<p>48TRAIN状态来自24父来源；开发数据反复用于诊断，并非最终留出。完整报告保留固定状态整分支bootstrap100和reference/flow数值对照。科学结论待结果检查，不自动认证或晋升。</p><p><a href="summary.json">汇总</a>；<a href="../protocol.json">协议</a></p><ul>'
    body+=''.join(f'<li><a href="../{p}">{p}</a></li>' for p in links)+'</ul>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1600px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    if not smoke:
        p=ROOT/'tests/index.html';text=p.read_text();start='<!-- training-restart-from-scratch:start -->';end='<!-- training-restart-from-scratch:end -->'
        section=start+'<section id="training-restart-from-scratch"><h2>最新：当前数据上的完整重新训练对照</h2><p><a href="training_restart_from_scratch/assets/index.html">查看完整图表、物理指标和构型退化检查</a></p><img style="max-width:100%" src="training_restart_from_scratch/assets/dev_comparison.png"></section>'+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:section,text,flags=re.S)
        else:text=text.replace('</body>',section+'</body>') if '</body>' in text else text+section
        tmp=p.with_suffix('.html.tmp');tmp.write_text(text);tmp.replace(p)
    if missing or errors:raise RuntimeError('Incomplete or unpaired experiment')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['analyze','aggregate']);p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);verify()
    if a.action=='analyze':analyze(a.index,a.smoke)
    else:aggregate(a.smoke)
