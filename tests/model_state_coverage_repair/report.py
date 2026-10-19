"""Conditional statistics on real versus generated initial states."""
import argparse,json,re
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tests.transition_block_repair.metrics import compare_first
from tests.conditional_moment_repair.report import metrics
from tests.transport_audit.report import table
from tests.training_restart_repair.data import HERE as REAL,rows as real_rows
from tests.bauer_p2_20260923.common import dump
from .data import HERE,ROOT,rows,source,verify,seeds

def ref(row,role):
    p=HERE/'reference'/f"{row['parent_index']:02d}"/f"state{row['index']:03d}_{role}.h5"
    with h5py.File(p) as h:
        assert h.attrs['complete'] and h.attrs['role']==role
        assert json.loads(h.attrs['metadata'])==row
        assert np.array_equal(h['seeds'][:],seeds(row['index'],role))
        return h['spins'][:]

def analyze(index,smoke):
    seed=(11,22,33)[index];out=HERE/('smoke' if smoke else 'models')/str(seed)
    results=[]
    with h5py.File(out/'paths.h5') as h:
        assert h.attrs['complete']
        for row in rows(smoke):
            name=f"state{row['index']:03d}";g=h[name]
            assert json.loads(g.attrs['metadata'])==row
            refs={k:ref(row,k) for k in ('probe','fine')}
            for steps in ((8,16) if smoke else (128,512)):
                x=g[f'midpoint{steps}'][:]
                results.append(dict(origin='model',metadata=row,sampler=steps,
                    metrics={k:compare_first(x,v) for k,v in refs.items()},
                    reference_control=compare_first(refs['probe'],refs['fine'])))
    rr=real_rows(smoke)
    with h5py.File(source(seed)/'after_restart.h5') as h:
        assert h.attrs['complete']
        for row in rr:
            name=f"state{row['index']:03d}";x=h[f'{name}/spins'][:];refs={}
            assert json.loads(h[name].attrs['metadata'])==row
            for role in ('probe','fine'):
                with h5py.File(REAL/'reference'/f"{row['parent_index']:02d}"/f'{name}_{role}.h5') as rh:
                    assert rh.attrs['complete'] and rh.attrs['role']==role;refs[role]=rh['spins'][:]
            results.append(dict(origin='llg',metadata=row,sampler=128,
                metrics={k:compare_first(x,v) for k,v in refs.items()},
                reference_control=compare_first(refs['probe'],refs['fine'])))
    dump(out/'report.json',dict(seed=seed,rows=results,scientific_certification=False))

def aggregate(smoke):
    seeds_=(11,) if smoke else (11,22,33);records=[];runs=[]
    for seed in seeds_:
        folder=HERE/('smoke' if smoke else 'models')/str(seed)
        run=json.loads((folder/'complete.json').read_text());runs.append(run)
        data=json.loads((folder/'report.json').read_text())
        for row in data['rows']:
            m=row['metadata']
            records.append([seed,row['origin'],row['sampler'],m['parent']['family'],m['source_time_ps'],m['index'],
                *metrics(row['metrics']['fine']),*metrics(row['reference_control'])])
    out=HERE/('smoke_assets' if smoke else 'assets');out.mkdir(exist_ok=True)
    dump(out/'summary.json',dict(rows=records,runs=runs,
        columns=['seed','origin','sampler','family','origin_time_ps','state','mean_RMSE','cov_RMSE','signed_variance_bias','energy_RMSE',
                 'ref_ref_mean_RMSE','ref_ref_cov_RMSE','ref_ref_variance_bias','ref_ref_energy_RMSE'],scientific_certification=False))
    baseline=8 if smoke else 128
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for k,(ax,title) in enumerate(zip(axes.flat,('Increment mean RMSE','Increment covariance RMSE','Signed variance bias','Energy mean RMSE (H/NJ)'))):
        for j,origin in enumerate(('llg','model')):
            vals=[np.mean([r[6+k] for r in records if r[0]==seed and r[1]==origin and r[2]==(128 if origin=='llg' else baseline)]) for seed in seeds_]
            ax.bar(j,np.mean(vals),color=('#0072B2','#E69F00')[j],edgecolor='black',width=.65)
            ax.scatter(j+np.linspace(-.1,.1,len(vals)),vals,color='black',zorder=3)
        ax.axhline(0,color='gray',lw=.7);ax.set_xticks([0,1],['LLG states','Model states']);ax.set_title(title)
    fig.suptitle('Frozen warm checkpoints: independent LLG futures'+(' (SOFTWARE SMOKE)' if smoke else ''))
    fig.savefig(out/'conditional_comparison.png',dpi=180);plt.close(fig)
    coverage=json.loads((HERE/'state_comparison.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,key in zip(axes,('energy_over_NJ','mz','neighbor_alignment')):
        real={r['state']:r[key] for r in coverage if r['origin']=='llg'}
        model={r['state']:r[key] for r in coverage if r['origin']=='model'}
        xx=[real[i] for i in sorted(real)];yy=[model[i] for i in sorted(model)]
        ax.scatter(xx,yy,s=20,color='#0072B2');lo,hi=min(xx+yy),max(xx+yy)
        ax.plot([lo,hi],[lo,hi],color='gray',lw=.8);ax.set_xlabel('LLG state');ax.set_ylabel('Model state');ax.set_title(key)
    fig.savefig(out/'state_comparison.png',dpi=180);plt.close(fig)
    body='<h1>模型访问状态的独立LLG后继核验与训练数据补齐</h1><p>48个状态来自24个TRAIN父来源；warm seed11固定生成源，branch0，0.3/0.8ps，按索引选取，不按结果筛选。其他两个模型种子在相同状态上评价，不称为它们各自的滚动分布。每状态64train+128probe+128fine，后继0.1ps；TRAIN样本暂未进入新模型训练。</p>'
    body+='<p>原模型完整重训后短时误差改善而1ps滚动变差，本轮核查条件偏差是否在模型实际访问状态加重。数据来自相同Bauer Hamiltonian、T、开放边界与MT5.9积分器，base0.25fs/fine0.125fs。模型输出的未来不是物理训练目标，仅取其当前磁构型作为新的给定初态。</p>'
    body+='<img src="state_comparison.png" style="max-width:100%"><p>每点为同来源同物理时间的两种初态。偏离对角线表示两种状态分布已经不同，不单独证明缺少训练覆盖；此前native训练还有其他时间窗口。</p>'
    body+='<img src="conditional_comparison.png" style="max-width:100%"><p>柱为训练种子描述性均值，点为各个模型种子；不是总体置信区间。llg行复用同一当前warmcheckpoint已经保存的独立未来；model行从新模型状态生成，再与该状态的独立LLG未来比较。状态分布不同，误差差异提供诊断线索而非唯一因果证据。</p>'
    body+=table(['seed','初态来源','flow步','构型','起点ps','state','增量均值RMSE','协方差RMSE','方差偏差','能量RMSE','参考-参考均值','参考-参考协方差','参考-参考方差','参考-参考能量'],records)
    body+='<p>每固定状态按整条独立未来bootstrap100，原始report.json包含点态区间。状态和时间来自同父来源，不能当48独立父状态。reference-control包含抽样及数值差异，不是自动通过阈值。若参考不充分或差异无明确证据，保留不确定性，不强行判定。128/512仅flow采样收敛，不能增加物理时长。</p><p><a href="summary.json">汇总</a>；<a href="../protocol.json">协议</a></p>'
    (out/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1600px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    if not smoke:
        p=ROOT/'tests/index.html';txt=p.read_text();start='<!-- model-state-coverage-repair:start -->';end='<!-- model-state-coverage-repair:end -->'
        section=start+'<section id="model-state-coverage-repair"><h2>模型访问状态与LLG独立后继</h2><a href="model_state_coverage_repair/assets/index.html">查看状态分布、条件误差和新数据</a><img style="max-width:100%" src="model_state_coverage_repair/assets/conditional_comparison.png"></section>'+end
        if start in txt:txt=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:section,txt,flags=re.S)
        else:txt=txt.replace('</body>',section+'</body>') if '</body>' in txt else txt+section
        tmp=p.with_suffix('.html.tmp');tmp.write_text(txt);tmp.replace(p)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['analyze','aggregate']);p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    verify()
    analyze(a.index,a.smoke) if a.action=='analyze' else aggregate(a.smoke)
