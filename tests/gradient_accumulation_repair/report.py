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
from .run import HERE, ROOT, ARMS, tasks, source, dump, verify, budget


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
    dump(folder/'report.json',dict(run=run,data=data,intermediate=middle,restart_seen=seen,scientific_certification=False))


def grouped(records, keys, value):
    result=[]
    groups=sorted({tuple(keys(x)) for x in records})
    for group in groups:
        selected=[x for x in records if tuple(keys(x))==group]
        result.append(list(group)+np.mean([value(x) for x in selected],axis=0).tolist())
    return result


def aggregate(smoke):
    first=[];full=[];middle=[];seen=[];overview=[];runs=[];exposures=[];missing=[];errors=[];links=[]
    for i in range(3 if smoke else 9):
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
        updates,accumulation,total=budget(t,smoke)
        assert counts.shape==(48,64) and counts.sum()==total*48 and (counts.sum(1)==total).all()
        assert run['optimizer_updates']==updates and run['microsteps']==total
        exposures.append(label+[int(counts.sum()),int(counts.sum(1).min()),int(counts.sum(1).max()),int((counts==0).sum()),run['train_seconds'],run['evaluation_seconds']])
        del r
    for seed in ((11,) if smoke else (11,22,33)):
        group=[r for r in runs if r['task']['seed']==seed]
        for key in ('native','extra','schedule'):
            if len({r['prefix_hashes'][key] for r in group})!=1:errors.append([seed,key])
    assets=HERE/('smoke_assets' if smoke else 'assets');assets.mkdir(exist_ok=True)
    dump(assets/'summary.json',dict(first=first,full=full,intermediate=middle,restart_seen=seen,overview=overview,
        exposure_columns=['seed','arm','total','min_per_state','max_per_state','unused_state_branch_pairs','train_seconds','evaluation_seconds'],
        exposures=exposures,runs=runs,reports=links,missing=missing,errors=errors,scientific_certification=False))
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for k,(ax,title) in enumerate(zip(axes.flat,('Increment mean RMSE','Increment covariance RMSE','Signed variance bias','Energy mean RMSE (H/NJ)'))):
        for j,arm in enumerate(ARMS):
            values=[x[2+k] for x in overview if x[1]==arm]
            if values:
                ax.bar(j,np.mean(values),color=('#0072B2','#E69F00','#009E73')[j],edgecolor='black',width=.65)
                ax.scatter(j+np.linspace(-.1,.1,len(values)),values,color='black',s=22,zorder=3)
        ax.axhline(0,color='gray',lw=.7);ax.set_title(title);ax.set_xticks(range(3),['Batch1','Acc8 same data','Acc8 same updates'])
    fig.suptitle('DEV intermediate states: descriptive means; dots = training seeds'+(' (SMOKE ONLY)' if smoke else ''))
    fig.savefig(assets/'dev_comparison.png',dpi=180);plt.close(fig)
    body=f'<h1>梯度累积修复：同数据与同更新预算对照</h1><p>完成{len(runs)}，缺失{missing}，配对错误{errors}。'+('仅软件冒烟，不能解读为物理结果。' if smoke else '开发诊断，不是最终认证。')+'</p>'
    body+='<p>从上一轮sparse_iid第24000步权重和Adam出发，每微批保留原始8条与48状态各一条重启监督。Batch1：6000更新/6000微批；Acc8 same data：750更新/6000微批；Acc8 same updates：6000更新/48000微批。累积组两项loss都除以8，累积结束后才clip和Adamstep。首6000微批的数据流/RNG配对。LR各自在更新维度从3e−5余弦降至3e−6，不自动线性放大。history中FM值为该次更新的最后微批loss，不是整个累积组均值；gradient为累积后的梯度范数。</p>'
    body+='<p>已完成8192批冻结诊断的平均梯度cosine三个种子均为正，因此不启用冲突投影。小批次波动很大，但是否造成物理误差仍需本轮检验。同数据组改变更新次数、Adam记忆与权重衰减次数；同更新组增加8倍数据曝光与训练计算。不得将收益无条件归为纯梯度方差效应。</p>'

    body+='<img src="dev_comparison.png" style="max-width:100%"><p>图示独立开发中间状态各分项误差，柱为种子均值，点为每个训练种子；不是置信区间。前三项为无量纲磁矩增量统计；方差是有符号偏差，零最好。能量采用该Bauer体系经核对的H/(NJ)，不要求有限温能量守恒。总体平均只是描述，须结合下面按构型分组的退化检查。</p>'
    body+='<h2>训练中间构型：独立噪声后继的前后对比</h2>'+table(['seed','方案','阶段','构型','起点ps','均值RMSE','协方差RMSE','方差偏差','能量均值RMSE'],seen)
    body+='<h2>独立开发中间构型：是否真正推广</h2>'+table(['seed','方案','L','构型','起点ps','均值RMSE','协方差RMSE','方差偏差','能量均值RMSE'],middle)
    body+='<h2>原始初态：是否发生遗忘或退化</h2>'+table(['seed','方案','角色','L','构型','采样','均值RMSE','协方差RMSE','方差偏差','能量均值RMSE'],first)
    body+='<h2>完整1ps：未直接优化的时间组织</h2>'+table(['seed','方案','角色','L','构型','采样','增量方差RMSE','两时关联RMSE','能量均值RMSE'],full)
    body+='<h2>真实曝光与成本</h2>'+table(['seed','方案','总新增样本曝光','每状态最少','每状态最多','未用状态分支对','训练秒','评估秒'],exposures)
    body+='<p>48训练状态来自24父状态，不是48独立准备；probe/fine只用于评价，训练从未使用。dev反复用于诊断，不是最终留出。原始逐父状态报告包含空间、磁化、数值采样128/512、每固定状态100次整分支bootstrap区间；相关状态跨组平均没有虚构总体置信区间。均衡仅保证每状态曝光，不保证总体状态覆盖。若训练状态改善而开发不改善，泛化仍未修复；若旧构型退化，不能仅报收益。各任务耗时包括完整评估，不是纯训练时间。</p>'
    body+='<p><a href="summary.json">轻量汇总</a>；<a href="../protocol.json">冻结协议</a></p><ul>'
    body+=''.join(f'<li><a href="../{p}">完整分项与区间：{p}</a></li>' for p in links)+'</ul>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1600px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    if not smoke:
        p=ROOT/'tests/index.html';text=p.read_text();start='<!-- gradient-accumulation-repair:start -->';end='<!-- gradient-accumulation-repair:end -->'
        section=start+'<section id="gradient-accumulation-repair"><h2>最新：梯度累积与训练噪声修复对照</h2><p><a href="gradient_accumulation_repair/assets/index.html">查看完整图表、物理指标和构型退化检查</a></p><img style="max-width:100%" src="gradient_accumulation_repair/assets/dev_comparison.png"></section>'+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:section,text,flags=re.S)
        else:text=text.replace('</body>',section+'</body>') if '</body>' in text else text+section
        tmp=p.with_suffix('.html.tmp');tmp.write_text(text);tmp.replace(p)
    if missing or errors:raise RuntimeError('Incomplete or unpaired experiment')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['analyze','aggregate']);p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);verify()
    if a.action=='analyze':analyze(a.index,a.smoke)
    else:aggregate(a.smoke)
