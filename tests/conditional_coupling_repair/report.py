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
        total=16 if smoke else 6000
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
        ax.axhline(0,color='gray',lw=.7);ax.set_title(title);ax.set_xticks(range(2),['Random pairs','Assigned pairs'])
    fig.suptitle('DEV intermediate states: descriptive means; dots = training seeds'+(' (SMOKE ONLY)' if smoke else ''))
    fig.savefig(assets/'dev_comparison.png',dpi=180);plt.close(fig)
    body=f'<h1>同初态分支配对：随机配对与最小距离配对</h1><p>完成{len(runs)}，缺失{missing}，配对错误{errors}。'+('仅软件冒烟。' if smoke else '开发诊断，不是最终认证。')+'</p>'
    body+='<p>上一轮C4/C8运输场投影减少部分误差，但剩余条件方差偏大，完整1ps增量方差误差未改善。本轮不再增加推理角度或flow步数；从三组axial_aug第42000步继续6000次同预算训练，骨干和采样器不变。</p>'
    body+='<p>两组都使用native8及3个TRAIN初态各16条后继，共48条额外监督；16次更新覆盖全部48个训练状态，每个状态总曝光6000次。随机组保持独立配对，候选在同一初态的16条潜变量与16条未来之间做一一最小距离指派。完整磁构型作为单位，不跨初态、不逐格点重排、不删样本、不增加物理数据；每个条件组共同绕易轴旋转，两组使用相同随机流。</p>'
    body+='<p>这是更换辅助运输耦合，不是让潜变量对应真实热噪声；每个批次的源集合、目标集合及均匀权重保持不变。有限batch指派不等于全分布最优运输，距离减少不保证学习更准确，flow loss变化也不可直接解释成物理改善。与旧48状态每状态1条的组织不同：本轮随机组是匹配对照，不能把分组本身的影响归功于配对。</p>'
    audit=json.loads((HERE/'coupling_audit.json').read_text())
    body+='<h2>训练数据几何预检</h2>'+table(['随机配对平均距离平方','指派后平均距离平方'],[[audit['mean_random_cost'],audit['mean_assignment_cost']]])
    body+='<p>以上为TRAIN数据的球面运输距离，不是物理LLG速度或物理时间。没有Hamiltonian/有效场输入，也没有新的矩损失或推理guidance。旋转增强的有限dt参考误差限制继承上一轮，物理指标继续对比原始独立参考。</p>'
    body+='<h2>固定TRAIN运输协变探针（非物理验收）</h2>'+table(['seed','方案','构型','tau','角度rad','协变绝对RMS','相对RMS'],symmetry)
    body+='<h2>参数与成本声明</h2>'+table(['seed','方案','参数量','训练秒','评估秒'],[[r['task']['seed'],r['task']['repair'],r['parameter_count'],r['train_seconds'],r['evaluation_seconds']] for r in runs])

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
        p=ROOT/'tests/index.html';text=p.read_text();start='<!-- conditional-coupling-repair:start -->';end='<!-- conditional-coupling-repair:end -->'
        section=start+'<section id="conditional-coupling-repair"><h2>最新：同初态分支配对修复对照</h2><p><a href="conditional_coupling_repair/assets/index.html">查看完整图表、物理指标和构型退化检查</a></p><img style="max-width:100%" src="conditional_coupling_repair/assets/dev_comparison.png"></section>'+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:section,text,flags=re.S)
        else:text=text.replace('</body>',section+'</body>') if '</body>' in text else text+section
        tmp=p.with_suffix('.html.tmp');tmp.write_text(text);tmp.replace(p)
    if missing or errors:raise RuntimeError('Incomplete or unpaired experiment')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['analyze','aggregate']);p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);verify()
    if a.action=='analyze':analyze(a.index,a.smoke)
    else:aggregate(a.smoke)


