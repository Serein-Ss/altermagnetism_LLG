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
    for phase,path in (('before',source(run['task']['seed'])/'after_restart.h5'),('after',folder/'after_restart.h5')):
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
        total=16 if smoke else 3072
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
        ax.axhline(0,color='gray',lw=.7);ax.set_title(title);ax.set_xticks(range(2),['Depth 4','Depth 8'])
    fig.suptitle('DEV intermediate states: descriptive means; dots = training seeds'+(' (SMOKE ONLY)' if smoke else ''))
    fig.savefig(assets/'dev_comparison.png',dpi=180);plt.close(fig)
    body=f'<h1>网络深度修复对照</h1><p>完成{len(runs)}，缺失{missing}，配对错误{errors}。'+('软件预检。' if smoke else '开发诊断。')+'</p>'
    body+='<p>从上一轮control57072步checkpoint开始，三种子、3072步。对照继续原depth4；候选保留原4块并追加4个零门控块，起始函数一致。旧参数按名字映射保留Adam状态和3e-5→3e-6；新增参数新Adam状态、3e-4→3e-5。数据、物理网格、源分布、经验教师与增强相同，均恢复uniform flow权重。</p>'
    body+='<p>候选同时增加参数、计算和新增参数优化空间，不是等计算容量对照。旧hierarchical网络已有全局通路，增加深度不能解释为此前完全缺少远程信息。只检验该背景配置下的增加容量能否改善条件路径统计，不宣称唯一根因或已完成多尺度迁移。</p>'
    activity=[]
    for r in runs:
        for phase in ('before','after'):
            activity += [[r['task']['seed'],r['task']['repair'],phase,x['tau'],x['added_velocity_rms'],x['relative_rms']] for x in r['activation_'+phase]]
    body+='<h2>新增块是否实际参与生成</h2>'+table(['seed','方案','阶段','flow tau','附加场RMS','相对完整场RMS'],activity)
    body+='<p>这里比较完整8块与暂时跳过最后4块的预测，仅是训练内活动诊断。不能当作最终物理模型或概率正确性证据。</p>'
    body+='<img src="dev_comparison.png" style="max-width:100%"><p>柱为描述性种子均值，点为训练种子；方差偏差应接近零，其他RMSE越小越好。能量单位H/(NJ)。按构型、尺寸和物理时间逐项检查，不能用平均值掩盖退化。</p>'
    body+='<h2>训练中间构型：独立噪声后继的前后对比</h2>'+table(['seed','方案','阶段','构型','起点ps','均值RMSE','协方差RMSE','方差偏差','能量均值RMSE'],seen)
    body+='<h2>开发中间构型</h2>'+table(['seed','方案','L','构型','起点ps','均值RMSE','协方差RMSE','方差偏差','能量均值RMSE'],middle)
    body+='<h2>原始初态</h2>'+table(['seed','方案','角色','L','构型','采样','均值RMSE','协方差RMSE','方差偏差','能量均值RMSE'],first)
    body+='<h2>完整1ps时间组织</h2>'+table(['seed','方案','角色','L','构型','采样','增量方差RMSE','两时关联RMSE','能量均值RMSE'],full)
    body+='<h2>参数与成本</h2>'+table(['seed','方案','参数量','训练秒','评估秒'],[[r['task']['seed'],r['task']['repair'],r['parameter_count'],r['train_seconds'],r['evaluation_seconds']] for r in runs])
    body+='<h2>曝光与配对</h2>'+table(['seed','方案','总曝光','每状态最少','每状态最多','未用状态分支对','训练秒','评估秒'],exposures)
    body+='<p>48TRAIN状态来自24父来源；开发数据反复用于诊断，并非最终留出。完整报告保留固定状态整分支bootstrap100和reference/flow数值对照。科学结论待结果检查，不自动认证或晋升。</p><p><a href="summary.json">汇总</a>；<a href="../protocol.json">协议</a></p><ul>'
    body+=''.join(f'<li><a href="../{p}">{p}</a></li>' for p in links)+'</ul>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1600px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    if not smoke:
        p=ROOT/'tests/index.html';text=p.read_text();start='<!-- depth-capacity-repair:start -->';end='<!-- depth-capacity-repair:end -->'
        section=start+'<section id="depth-capacity-repair"><h2>最新：网络深度修复对照</h2><p><a href="depth_capacity_repair/assets/index.html">查看完整图表、物理指标和构型退化检查</a></p><img style="max-width:100%" src="depth_capacity_repair/assets/dev_comparison.png"></section>'+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:section,text,flags=re.S)
        else:text=text.replace('</body>',section+'</body>') if '</body>' in text else text+section
        tmp=p.with_suffix('.html.tmp');tmp.write_text(text);tmp.replace(p)
    if missing or errors:raise RuntimeError('Incomplete or unpaired experiment')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['analyze','aggregate']);p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);verify()
    if a.action=='analyze':analyze(a.index,a.smoke)
    else:aggregate(a.smoke)
