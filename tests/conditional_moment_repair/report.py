"""Reuse frozen physical metrics, with paired controls and no automatic pass."""
import argparse
import json
import re
import h5py
import numpy as np
import torch
from tests.transition_block_repair.report import collect
from tests.transition_block_repair.metrics import compare_first
from tests.reference_resolution.report import brief
from tests.transport_audit.report import table
from .run import HERE,ROOT,tasks,dump


def analyze(index,smoke):
    folder=HERE/('smoke' if smoke else 'train')/f'{index:02d}'
    run=json.loads((folder/'complete.json').read_text());assert run['task']==tasks()[index]
    data=collect(folder/'paths.h5',smoke);rows=[]
    with h5py.File(folder/'intermediate.h5') as h:
        assert h.attrs['complete'] and len(h)==(1 if smoke else 72)
        for name,g in h.items():
            meta=json.loads(g.attrs['metadata']);pid=meta['parent_index'];x=g['spins'][:]
            with h5py.File(ROOT/f'tests/intermediate_restart_repair/reference/{pid:02d}/paths.h5') as ref:
                assert ref.attrs['complete']
                values={k:compare_first(x,ref[f'{name}/{k}/spins'][:]) for k in ('base','fine')}
            rows.append(dict(metadata=meta,metrics=values))
    dump(folder/'report.json',dict(run=run,data=data,intermediate=rows,scientific_certification=False))


def metrics(v):
    return [v['increment_mean']['rmse'],v['increment_cov']['rmse'],v['increment_variance']['difference'],v['energy_mean']['rmse']]


def aggregate(smoke):
    results=[];first=[];full=[];middle=[];missing=[];errors=[]
    for i in range(2 if smoke else 6):
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
    for seed in (11,) if smoke else (11,22,33):
        pair=[r for r in results if r['run']['task']['seed']==seed]
        for key in ('paired_input_hash','schedule_hash'):
            if len({r['run'][key] for r in pair})>1:errors.append([seed,key])
    assets=HERE/('smoke_assets' if smoke else 'assets');assets.mkdir(exist_ok=True)
    dump(assets/'summary.json',dict(results=results,first=first,full=full,intermediate=middle,missing=missing,errors=errors,scientific_certification=False))
    body=f'<h1>条件均值与协方差修复：FM与统计辅助配对对照</h1><p>完成{len(results)}，缺失{missing}，配对错误{errors}。'+('仅短时冒烟，不是正式结果。' if smoke else '')+'</p>'
    body+='<p>三个训练seed，固定adjacent12000检查点续训6000。辅助项仅用24个L16训练父状态的72条已有分支；独立32+32生成集合交叉积约束局域向量均值与3×3协方差，避免朴素样本均值平方引入额外压低方差的偏好。参考有限样本误差仍存在；辅助随机loss允许负值。没有引入解析Hamiltonian损失、开发参考或额外物理数据。</p>'
    body+='<h2>原初态0→0.1ps：均值和涨落一起看</h2>'+table(['seed','方案','数据','L','采样','均值RMSE','协方差RMSE','方差有符号偏差','能量均值RMSE'],first)
    body+='<h2>真实LLG中间状态重新出发：是否改善未监督状态</h2>'+table(['seed','方案','L','起点ps','均值RMSE','协方差RMSE','方差有符号偏差','能量均值RMSE'],middle)
    body+='<h2>10次接续完整1ps：辅助训练未直接优化的时间统计</h2>'+table(['seed','方案','数据','L','采样','增量方差RMSE','两时相关RMSE','能量均值RMSE'],full)
    body+='<p>dev仍是开发数据，不是最终留出；seen为训练父状态的新噪声，不等于独立父状态泛化。每固定状态整分支bootstrap100次，参考与生成误差均纳入；跨状态均值只作描述，不把相关状态或训练seed重复算成独立物理父状态。未定义事后通过阈值；同时检查未优化空间、能量、完整路径和采样128/512收敛。所有详细区间、计时见JSON。相同更新数不等于相同计算量。辅助项只覆盖t0，失败仍可能涉及状态覆盖，不能直接归因为架构无能力。</p><a href="summary.json">全部分项结果</a>'
    (assets/'index.html').write_text('<!doctype html><meta charset="utf-8"><body style="max-width:1500px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body>')
    if not smoke:
        p=ROOT/'tests/index.html';text=p.read_text();start='<!-- conditional-moment-repair:start -->';end='<!-- conditional-moment-repair:end -->'
        block=start+'<section id="conditional-moment-repair">'+body.replace('href="summary.json"','href="conditional_moment_repair/assets/summary.json"')+'</section>'+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,text,flags=re.S)
        else:text=text.replace('</body>',block+'</body>') if '</body>' in text else text+block
        tmp=p.with_suffix('.html.tmp');tmp.write_text(text);tmp.replace(p)
    if missing or errors:raise RuntimeError('Incomplete or unpaired experiment')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['analyze','aggregate']);p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    if a.action=='analyze':analyze(a.index,a.smoke)
    else:aggregate(a.smoke)
