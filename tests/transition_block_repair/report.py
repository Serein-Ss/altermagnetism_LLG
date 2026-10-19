"""Separate native one-step accuracy, composed path accuracy, and joint control."""
import argparse
import html
import json
import re
import h5py
import numpy as np
import torch
from tests.reference_resolution.report import compare,brief
from tests.transport_audit.report import table
from .run import HERE,ROOT,tasks,dump
from .metrics import compare_first


def collect(path,smoke,joint=False):
    rows=[];solver=[]
    with h5py.File(path) as h:
        assert h.attrs['complete']
        for role in ('dev','seen'):
            assert len(h[role])==(1 if smoke else (18 if role=='dev' else 24))
            for name,g in h[role].items():
                i=int(name.removeprefix('parent'));parent=json.loads(g.attrs['parent'])
                refpath=ROOT/(f'tests/reference_resolution/reference/{i:02d}/paths.h5' if role=='dev' else f'tests/branch_coverage_repair/reference/{i:02d}/probe.h5')
                with h5py.File(refpath) as ref:
                    assert ref.attrs['complete'] and json.loads(ref.attrs['parent'])==parent
                    refs={k:ref[f'{k}/spins'][:] for k in ('base','fine')} if role=='dev' else dict(probe=ref['spins'][:])
                assert np.allclose(g['time_ps'][:],np.arange(11)*.1)
                arrays={k:v[:] for k,v in g.items() if k.startswith('midpoint')}
                for sampler,x in arrays.items():
                    assert x.shape[1]==11 and np.isfinite(x).all()
                    norm=float(abs(np.linalg.norm(x,axis=-1)-1).max());anchor=float(abs(x[:,0]-next(iter(refs.values()))[:1,0]).max())
                    assert norm<2e-6 and anchor<2e-6
                    rows.append(dict(role=role,parent=parent,sampler=sampler,norm_error=norm,anchor_error=anchor,
                        full={k:compare(x,v) for k,v in refs.items()},first={k:compare_first(x,v) for k,v in refs.items()}))
                if role=='dev':
                    lo,hi=(('midpoint8','midpoint16') if joint else ('midpoint4','midpoint8')) if smoke else ('midpoint128','midpoint512')
                    solver.append(dict(parent=parent,same_latent_rms=float(np.sqrt(np.mean((arrays[lo]-arrays[hi])**2)))))
    return dict(rows=rows,solver=solver)


def analyze(index,smoke):
    folder=HERE/('smoke' if smoke else 'train')/f'{index:02d}';run=json.loads((folder/'complete.json').read_text())
    assert run['task']==tasks()[index]
    new=collect(folder/'paths.h5',smoke)
    seed=run['task']['seed'];old_index=(11,22,33).index(seed)*2
    old=ROOT/'tests/temporal_source_repair'/('smoke' if smoke else 'train')/f'{old_index:02d}'
    oldrun=json.loads((old/'complete.json').read_text())
    for key in ('initial_hash','paired_input_hash','schedule_hash'):assert run[key]==oldrun[key],key
    joint=collect(old/('step00032' if smoke else 'step12000')/'paths.h5',smoke,True)
    dump(folder/'report.json',dict(task=run['task'],run=run,new=new,joint=joint,joint_run=oldrun,scientific_certification=False))


def aggregate(smoke):
    results=[];missing=[];errors=[];full=[];first=[];seen_joint=set()
    for i in range(2 if smoke else 6):
        path=HERE/('smoke' if smoke else 'train')/f'{i:02d}/report.json'
        if not path.exists():missing.append(i);continue
        r=json.loads(path.read_text());results.append(r);seed=r['task']['seed']
        entries=[(r['task']['repair'],r['new'])]
        if seed not in seen_joint:entries.append(('joint11',r['joint']));seen_joint.add(seed)
        for arm,data in entries:
            for role in ('dev','seen'):
                selected=[v for v in data['rows'] if v['role']==role];key='fine' if role=='dev' else 'probe'
                for L in sorted({v['parent']['L'] for v in selected}):
                    for sampler in sorted({v['sampler'] for v in selected}):
                        ss=[v for v in selected if v['parent']['L']==L and v['sampler']==sampler]
                        full.append([seed,arm,role,L,sampler,*np.mean([brief(v['full'][key]) for v in ss],0).tolist()])
                        values=[[v['first'][key]['increment_mean']['rmse'],v['first'][key]['increment_cov']['rmse'],
                                 v['first'][key]['increment_variance']['difference'],v['first'][key]['energy_mean']['rmse']] for v in ss]
                        first.append([seed,arm,role,L,sampler,*np.mean(values,0).tolist()])
    for seed in (11,) if smoke else (11,22,33):
        pair=[r for r in results if r['task']['seed']==seed]
        for key in ('initial_hash','paired_input_hash','schedule_hash'):
            if len({r['run'][key] for r in pair})>1:errors.append(dict(seed=seed,error='unpaired '+key))
    assets=HERE/('smoke_assets' if smoke else 'assets');assets.mkdir(exist_ok=True)
    dump(assets/'summary.json',dict(results=results,missing=missing,errors=errors,first=first,full=full,scientific_certification=False))
    body=f'<h1>单步转移与整段路径：定位短时拟合和接续误差</h1><p>新实验完成{len(results)}；缺失{missing}；配对错误{html.escape(str(errors))}。'+('仅冒烟。' if smoke else '')+'</p>'
    body+='<p>joint11复用上一轮rho0、12000步模型；prefix新训练每个原窗口的首对帧；adjacent新训练50%首对、50%其余随机相邻帧。相同原窗口/初始化/源随机流已核查。三训练seed，Bauer开放链L16/25/32。网络张量结构相同，原生块长及监督目标不同；12000次更新不等于相同FLOPs或相同目标帧数，历史对照不是同期硬件计时基准。</p>'
    body+='<h2>第一步：0→0.1 ps 条件转移</h2><p>这里只计算两帧可定义的量，不调用lag2/5。方差偏差是有符号量，其他列为RMSE。均值/协方差应一起检查，不能靠初态变化小宣布正确。</p>'+table(['seed','方案','数据','L','采样','增量均值RMSE','增量协方差RMSE','增量方差偏差','能量均值误差'],first)
    body+='<h2>完整1 ps路径：联合生成或10次单步接续</h2>'+table(['seed','方案','数据','L','采样','增量方差RMSE','两时关联RMSE','能量均值RMSE'],full)
    body+='<p>单步模型每次使用生成末态和新噪声，无真实未来重置。midpoint128/512指每次转移的积分步数；十次接续的网络调用是单块的10倍，不能按相同步数宣称等成本。保存11帧0..1ps、不重复边界，随机流可恢复。</p>'
    body+='<p>如果首步仍偏差，说明不是只有整块时间组织困难；如果首步改善而完整路径失败，需要检查累积偏差及所遇状态覆盖。adjacent改变训练条件状态分布，可能减少原始初态曝光，结果需报告取舍。dev是反复使用的开发父状态；seen是已见父状态的新噪声，不是独立父状态泛化。100次整分支bootstrap、全部分项指标及计时在JSON。无自动通过、推广或加速认证。</p><p><a href="summary.json">完整指标和区间</a></p>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1500px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    if not smoke:
        p=ROOT/'tests/index.html';text=p.read_text();start='<!-- transition-block-repair:start -->';end='<!-- transition-block-repair:end -->'
        block=start+'<section id="transition-block-repair">'+body.replace('href="summary.json"','href="transition_block_repair/assets/summary.json"')+'</section>'+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,text,flags=re.S)
        else:text=text.replace('</body>',block+'</body>') if '</body>' in text else text+block
        tmp=p.with_suffix('.html.tmp');tmp.write_text(text);tmp.replace(p)
    if missing or errors:raise RuntimeError('Incomplete or unpaired experiment')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['analyze','aggregate']);p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    if a.action=='analyze':analyze(a.index,a.smoke)
    else:aggregate(a.smoke)


if __name__=='__main__':main()
