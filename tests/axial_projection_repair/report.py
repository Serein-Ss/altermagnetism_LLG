"""Frozen physical metrics; compact aggregate, full uncertainty per run."""
import argparse
import base64
import json
import re
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tests.transition_block_repair.metrics import compare_first
from tests.reference_resolution.report import compare,brief
from tests.conditional_moment_repair.report import metrics
from tests.transport_audit.report import table
from .run import HERE,ROOT,tasks,source,dump,verify


def analyze(index,smoke):
    folder=HERE/('smoke' if smoke else 'runs')/f'{index:02d}'
    run=json.loads((folder/'complete.json').read_text());assert run['task']==tasks()[index]
    result=[];solver=[];identity=[]
    with h5py.File(folder/'paths.h5') as h:
        assert h.attrs['complete']
        for role in ('dev','middle'):
            assert len(h[role])==(1 if smoke else (6 if role=='dev' else 24))
            for name,g in h[role].items():
                meta=json.loads(g.attrs['metadata'])
                if role=='dev':
                    pid=int(name.removeprefix('parent'))
                    refpath=ROOT/f'tests/reference_resolution/reference/{pid:02d}/paths.h5'
                    keys={k:f'{k}/spins' for k in ('base','fine')}
                else:
                    pid=meta['parent_index'];refpath=ROOT/f'tests/intermediate_restart_repair/reference/{pid:02d}/paths.h5'
                    keys={k:f'{name}/{k}/spins' for k in ('base','fine')}
                with h5py.File(refpath) as ref:
                    assert ref.attrs['complete'];refs={k:ref[v][:] for k,v in keys.items()}
                levels=(4,16) if smoke else (128,512)
                arrays={k:g[f'midpoint{k}'][:] for k in levels}
                for level,x in arrays.items():
                    assert np.isfinite(x).all() and abs(np.linalg.norm(x,axis=-1)-1).max()<2e-6
                    assert abs(x[:,0]-refs['fine'][:1,0]).max()<2e-6
                    entry=dict(role=role,name=name,metadata=meta,steps=level,
                        first={k:compare_first(x,v) for k,v in refs.items()})
                    if role=='dev' and level==levels[0]:
                        entry['full']={k:compare(x,v) for k,v in refs.items()}
                    if level==levels[0]:
                        entry['reference_control']=compare_first(refs['base'],refs['fine'])
                    result.append(entry)
                solver.append(dict(role=role,name=name,first_step_only_rms=float(np.sqrt(np.mean(
                    (arrays[levels[0]][:,:2]-arrays[levels[1]])**2)))))
                if not smoke and run['task']['order']==1:
                    oldpath=source(run['task']['seed'])/('paths.h5' if role=='dev' else 'intermediate.h5')
                    oldkey=f'dev/{name}/midpoint128' if role=='dev' else f'{name}/spins'
                    with h5py.File(oldpath) as old:oldx=old[oldkey][:]
                    rms=float(np.sqrt(np.mean((arrays[128]-oldx)**2)))
                    identity.append(dict(role=role,name=name,rms=rms))
                    assert rms<1e-5,('C1 historical regression',role,name,rms)
    dump(folder/'report.json',dict(run=run,rows=result,solver=solver,identity_regression=identity,
        symmetry=json.loads((folder/'symmetry.json').read_text()),scientific_certification=False))


def aggregate(smoke):
    first=[];full=[];runs=[];symmetry=[];solver=[];reports=[]
    for i in range(3 if smoke else 9):
        p=HERE/('smoke' if smoke else 'runs')/f'{i:02d}/report.json'
        r=json.loads(p.read_text());run=r['run'];runs.append(run);label=[run['task']['seed'],run['task']['order']]
        reports.append(str(p.relative_to(HERE)))
        for row in r['rows']:
            m=row['metadata'];family=m['family'] if row['role']=='dev' else m['parent']['family']
            first.append(label+[row['role'],family,row['name'],row['steps']]+metrics(row['first']['fine']))
            if 'full' in row:full.append(label+[family,row['name']]+brief(row['full']['fine']))
        symmetry += [label+[v['family'],v['tau'],v['angle'],v['relative_rms']] for v in r['symmetry']]
        solver += [label+[v['role'],v['name'],v['first_step_only_rms']] for v in r['solver']]
        del r
    assert len({r['paired_input_hash'] for r in runs})==1,'unpaired initial/RNG streams'
    assets=HERE/('smoke_assets' if smoke else 'assets');assets.mkdir(exist_ok=True)
    summary=dict(first=first,full=full,symmetry=symmetry,solver=solver,runs=runs,reports=reports,
                 scientific_certification=False)
    dump(assets/'summary.json',summary)
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    titles=['First mean RMSE','First covariance RMSE','First variance bias',
            '1ps increment variance RMSE','1ps two-time correlation RMSE','1ps energy mean RMSE']
    overview=[]
    for seed in ((11,) if smoke else (11,22,33)):
        for order in (1,4,8):
            a=[v[6:] for v in first if v[0]==seed and v[1]==order and v[2]=='dev' and v[5]==(4 if smoke else 128)]
            b=[v[4:] for v in full if v[0]==seed and v[1]==order]
            overview.append([seed,order,*np.mean(a,0).tolist(),*np.mean(b,0).tolist()])
    for j,ax in enumerate(axes.flat):
        col=(2,3,4,6,7,8)[j]
        for k,(order,color) in enumerate(zip((1,4,8),('#4477AA','#EE7733','#228833'))):
            values=[r[col] for r in overview if r[1]==order]
            ax.bar(k,np.mean(values),color=color,edgecolor='black',width=.65)
            ax.scatter(np.full(len(values),k)+np.linspace(-.1,.1,len(values)),values,color='black',s=18,zorder=3)
        ax.set_xticks(range(3),['C1','C4','C8']);ax.set_title(titles[j]);ax.axhline(0,color='black',lw=.6)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Bauer L16: fixed augmented checkpoints'+(' — SOFTWARE SMOKE ONLY' if smoke else ' — development evidence'))
    fig.savefig(assets/'comparison.png',dpi=180);plt.close(fig)
    picture=base64.b64encode((assets/'comparison.png').read_bytes()).decode()
    body='<h1>易轴对称投影：固定模型、不重训的因果排查</h1>'
    if smoke:body+='<p>仅软件冒烟；极少轨迹、粗flow步数，不解释物理优劣。</p>'
    body+='<p>同一组3个旋转增强checkpoint，各比较C1原模型、C4四角度、C8八角度运输向量场平均。不是平均物理轨迹，不改变初态、温度、边界或0.1ps物理网格。有限群严格对称不等于连续SO(2)，更不等于正确路径分布。</p>'
    body+='<p>主要范围：L16、6个开发父状态，每个128条1ps路径；另24个真实LLG中间状态各128条0.1ps未来。固定父状态整分支bootstrap100次，完整区间在各运行报告中。24个中间状态来自6个父来源，不能当作24个独立来源；反复使用的开发集不是最终测试。</p>'
    body+=f'<img style="max-width:100%" src="data:image/png;base64,{picture}"><p>柱为三个训练seed的描述性平均，黑点为各seed；不是置信区间。方差偏差为有符号量，以接近0为好；其余误差越低越好。必须同时看均值、涨落与时间关联。</p>'
    body+=table(['seed','K','首步均值RMSE','协方差RMSE','方差偏差','首步能量RMSE','1ps增量方差RMSE','1ps两时关联RMSE','1ps能量RMSE'],overview)
    body+='<h2>采样与适用边界</h2><p>128/512步配对比较只覆盖每个起点的第一段0.1ps；本轮不称其已证明完整1ps求解器收敛。C4/C8改变运输场，若结果不同仍需组平均阶数核验。统计不改善时，不能继续把残余方位依赖认作充分病因。MT参考有限步长噪声非任意角度严格不变，保留已有旋转参考核验和base/fine控制。</p>'
    body+=table(['seed','K','数据','状态','首步128/512配对RMS'],solver)
    body+='<h2>成本与结果</h2>'+table(['seed','K','生成秒','GPU'],[[r['task']['seed'],r['task']['order'],r['seconds'],r['device']] for r in runs])
    body+='<p>K角度平均每次需要K次骨干调用，不是加速方案。不得用结构通过或更多生成条数认证准确性。不存在自动通过阈值，不修改生产模型。</p><p><a href="summary.json">精简指标</a> · <a href="../protocol.json">冻结方案</a></p>'
    body+='<ul>'+''.join(f'<li><a href="../{p}">{p}</a></li>' for p in reports)+'</ul>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1450px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    if not smoke:
        p=ROOT/'tests/index.html';text=p.read_text();start='<!-- axial-projection-repair:start -->';end='<!-- axial-projection-repair:end -->'
        dashboard_body=body.replace('href="summary.json"','href="axial_projection_repair/assets/summary.json"').replace('href="../','href="axial_projection_repair/')
        block=start+'<section id="axial-projection-repair">'+dashboard_body+'</section>'+end
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,text,flags=re.S)
        else:text=text.replace('</body>',block+'</body>') if '</body>' in text else text+block
        tmp=p.with_suffix('.html.tmp');tmp.write_text(text);tmp.replace(p)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['analyze','aggregate']);p.add_argument('--index',type=int,default=0)
    p.add_argument('--smoke',action='store_true');a=p.parse_args();torch.set_num_threads(1);verify()
    if a.action=='analyze':analyze(a.index,a.smoke)
    else:aggregate(a.smoke)
