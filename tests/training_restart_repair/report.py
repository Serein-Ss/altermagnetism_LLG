"""Separate fitting train-origin states from independent-parent generalization."""
import argparse
import json
import re
import h5py
import numpy as np
import torch
from tests.transition_block_repair.report import collect
from tests.transition_block_repair.metrics import compare_first
from tests.reference_resolution.report import brief
from tests.conditional_moment_repair.report import metrics
from tests.transport_audit.report import table
from .data import HERE,ROOT,rows,dump
from .run import tasks


def analyze(index,smoke):
    folder=HERE/('smoke' if smoke else 'train')/f'{index:02d}';run=json.loads((folder/'complete.json').read_text())
    assert run['task']==tasks()[index];data=collect(folder/'paths.h5',smoke);middle=[];seen=[]
    with h5py.File(folder/'intermediate.h5') as h:
        assert h.attrs['complete'] and len(h)==(1 if smoke else 72)
        for name,g in h.items():
            meta=json.loads(g.attrs['metadata']);pid=meta['parent_index'];x=g['spins'][:]
            with h5py.File(ROOT/f'tests/intermediate_restart_repair/reference/{pid:02d}/paths.h5') as ref:
                assert ref.attrs['complete']
                values={k:compare_first(x,ref[f'{name}/{k}/spins'][:]) for k in ('base','fine')}
            middle.append(dict(metadata=meta,metrics=values))
    for phase in ('before','after'):
        with h5py.File(folder/f'{phase}_restart.h5') as h:
            assert h.attrs['complete'] and len(h)==(2 if smoke else 48)
            for row in rows(smoke):
                sid=row['index'];name=f'state{sid:03d}';x=h[f'{name}/spins'][:];values={};refs={}
                for role in ('probe','fine'):
                    with h5py.File(HERE/'reference'/f"{row['parent_index']:02d}"/f'{name}_{role}.h5') as ref:
                        assert ref.attrs['complete'] and ref.attrs['role']==role
                        refs[role]=ref['spins'][:];values[role]=compare_first(x,refs[role])
                seen.append(dict(phase=phase,metadata=row,metrics=values,reference_control=compare_first(refs['probe'],refs['fine'])))
    dump(folder/'report.json',dict(run=run,data=data,intermediate=middle,restart_seen=seen,scientific_certification=False))


def aggregate(smoke):
    results=[];first=[];full=[];middle=[];seen=[];missing=[];errors=[]
    for i in range(3 if smoke else 9):
        p=HERE/('smoke' if smoke else 'train')/f'{i:02d}/report.json'
        if not p.exists():missing.append(i);continue
        r=json.loads(p.read_text());results.append(r);t=r['run']['task'];label=[t['seed'],t['repair']]
        for role,key in (('seen','probe'),('dev','fine')):
            for L in (16,25,32):
                selected=[x for x in r['data']['rows'] if x['role']==role and x['parent']['L']==L]
                for sampler in sorted({x['sampler'] for x in selected}):
                    ss=[x for x in selected if x['sampler']==sampler]
                    first.append(label+[role,L,sampler]+np.mean([metrics(x['first'][key]) for x in ss],0).tolist())
                    full.append(label+[role,L,sampler]+np.mean([brief(x['full'][key]) for x in ss],0).tolist())
        for L in (16,25,32):
            for time in (.3,.8):
                ss=[x for x in r['intermediate'] if x['metadata']['parent']['L']==L and abs(x['metadata']['origin_time_ps']-time)<1e-8]
                if ss:middle.append(label+[L,time]+np.mean([metrics(x['metrics']['fine']) for x in ss],0).tolist())
        for phase in ('before','after'):
            for time in (.3,.8):
                ss=[x for x in r['restart_seen'] if x['phase']==phase and abs(x['metadata']['source_time_ps']-time)<1e-8]
                seen.append(label+[phase,time]+np.mean([metrics(x['metrics']['fine']) for x in ss],0).tolist())
    for seed in (11,) if smoke else (11,22,33):
        pair=[r for r in results if r['run']['task']['seed']==seed]
        for key in ('paired_input_hash','extra_input_hash','schedule_hash'):
            if len({r['run'][key] for r in pair})>1:errors.append([seed,key])
    assets=HERE/('smoke_assets' if smoke else 'assets');assets.mkdir(exist_ok=True)
    dump(assets/'summary.json',dict(results=results,first=first,full=full,intermediate=middle,restart_seen=seen,missing=missing,errors=errors,scientific_certification=False))
    body=f'<h1>训练中间状态后继补全：覆盖、拟合和泛化</h1><p>完成{len(results)}，缺失{missing}，配对错误{errors}。'+('仅冒烟。' if smoke else '')+'</p>'
    body+='<p>native保留原FM；restart_fm额外加入训练中间构型的独立后继；restart_moments在相同后继数据上再加均值与协方差辅助。三个seed从相同adjacent12000模型续训6000。48状态来自24个L16训练父轨迹的固定branch0、0.3/0.8ps，全部继承来源train；每状态64训练、128独立噪声probe、128细步长参考。后两组从未进入损失，dev参考也不训练。</p>'
    body+='<h2>新训练状态的新噪声：训练前后能否学准</h2>'+table(['seed','方案','阶段','起点ps','均值RMSE','协方差RMSE','方差偏差','能量均值RMSE'],seen)
    body+='<h2>独立开发中间状态：拟合是否迁移</h2>'+table(['seed','方案','L','起点ps','均值RMSE','协方差RMSE','方差偏差','能量均值RMSE'],middle)
    body+='<h2>原始初态：是否遗忘</h2>'+table(['seed','方案','角色','L','采样','均值RMSE','协方差RMSE','方差偏差','能量均值RMSE'],first)
    body+='<h2>完整1ps：是否改善未直接约束的时间组织</h2>'+table(['seed','方案','角色','L','采样','增量方差RMSE','两时关联RMSE','能量均值RMSE'],full)
    body+='<p>新增FM每两步加权0.5，故改变覆盖和曝光权重，不是纯粹独立的数据量因果效应；统计辅助每16步进行。相同更新数不等于同计算成本。新状态只有24个来源组，不能当48独立准备；训练状态probe不是独立父状态测试。bootstrap100按固定状态的完整随机分支进行，跨状态均值只作描述，seed差异单报。参考间差别含有限样本和dt差异，不作自动容差。若只有seen改善则泛化未修复；所有方案失败也不独证架构无能力。无自动认证、阈值放宽或正式目录晋升。</p><a href="summary.json">完整指标、置信区间与计时</a>'
    (assets/'index.html').write_text('<!doctype html><meta charset="utf-8"><body style="max-width:1500px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body>')
    if not smoke:
        p=ROOT/'tests/index.html';text=p.read_text();start='<!-- training-restart-repair:start -->';end='<!-- training-restart-repair:end -->'
        block=start+'<section id="training-restart-repair">'+body.replace('href="summary.json"','href="training_restart_repair/assets/summary.json"')+'</section>'+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,text,flags=re.S)
        else:text=text.replace('</body>',block+'</body>') if '</body>' in text else text+block
        tmp=p.with_suffix('.html.tmp');tmp.write_text(text);tmp.replace(p)
    if missing or errors:raise RuntimeError('Incomplete or unpaired experiment')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['analyze','aggregate']);p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    if a.action=='analyze':analyze(a.index,a.smoke)
    else:aggregate(a.smoke)
