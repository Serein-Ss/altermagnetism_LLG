"""Fixed final checkpoints; paired replay intervention and physical regressions."""
import argparse,json,re
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tests.transition_block_repair.report import collect
from tests.transition_block_repair.metrics import compare_first
from tests.reference_resolution.report import brief
from tests.conditional_moment_repair.report import metrics
from tests.transport_audit.report import table

from tests.training_restart_repair.data import HERE as REAL,rows as real_rows
from .run import HERE,ROOT,ARMS,tasks,source,dump,verify,visited

def ref(row,role):
    with h5py.File(visited.HERE/'reference'/f"{row['parent_index']:02d}"/f"state{row['index']:03d}_{role}.h5") as h:
        assert h.attrs['complete'] and h.attrs['role']==role and json.loads(h.attrs['metadata'])==row
        assert np.array_equal(h['seeds'][:],visited.seeds(row['index'],role))
        return h['spins'][:]

def analyze(index,smoke):
    out=HERE/('smoke' if smoke else 'train')/f'{index:02d}'
    run=json.loads((out/'complete.json').read_text());assert run['task']==tasks()[index]
    middle=[];conditional=[]
    with h5py.File(out/'intermediate.h5') as h:
        assert h.attrs['complete']
        for name,g in h.items():
            meta=json.loads(g.attrs['metadata'])
            with h5py.File(ROOT/f"tests/intermediate_restart_repair/reference/{meta['parent_index']:02d}/paths.h5") as rh:
                assert rh.attrs['complete']
                vals={k:compare_first(g['spins'][:],rh[f'{name}/{k}/spins'][:]) for k in ('base','fine')}
            middle.append(dict(metadata=meta,metrics=vals))
    for origin,filename,rows_ in (('llg','after_restart.h5',real_rows(smoke)),('model','model_states.h5',visited.rows(smoke)),('model_before','before_model_states.h5',visited.rows(smoke))):
        with h5py.File(out/filename) as h:
            assert h.attrs['complete']
            for row in rows_:
                name=f"state{row['index']:03d}";g=h[name];assert json.loads(g.attrs['metadata'])==row
                refs={}
                for role in ('probe','fine'):
                    if origin!='llg':refs[role]=ref(row,role)
                    else:
                        with h5py.File(REAL/'reference'/f"{row['parent_index']:02d}"/f'{name}_{role}.h5') as rh:
                            assert rh.attrs['complete'] and rh.attrs['role']==role
                            assert json.loads(rh.attrs['metadata'])==row
                            refs[role]=rh['spins'][:]
                conditional.append(dict(origin=origin,metadata=row,metrics={k:compare_first(g['spins'][:],v) for k,v in refs.items()}))
    dump(out/'report.json',dict(run=run,data=collect(out/'paths.h5',smoke),intermediate=middle,conditional=conditional,
         symmetry=json.loads((out/'symmetry.json').read_text()),scientific_certification=False))

def aggregate(smoke):
    runs=[];dev=[];full=[];cond=[];errors=[];auxiliary=[]
    for i in range(3 if smoke else 9):
        out=HERE/('smoke' if smoke else 'train')/f'{i:02d}'
        r=json.loads((out/'report.json').read_text());run=r['run'];t=run['task'];seed=t['seed'];arm=t['repair'];runs.append(run)
        dev.append([seed,arm,*np.mean([metrics(x['metrics']['fine']) for x in r['intermediate']],0).tolist()])
        for L in sorted({x['parent']['L'] for x in r['data']['rows'] if x['role']=='dev'}):
            for sampler in ('midpoint4','midpoint8') if smoke else ('midpoint128','midpoint512'):
                rr=[x for x in r['data']['rows'] if x['role']=='dev' and x['parent']['L']==L and x['sampler']==sampler]
                full.append([seed,arm,L,sampler,*np.mean([brief(x['full']['fine']) for x in rr],0).tolist()])
        for x in r['conditional']:
            m=x['metadata'];cond.append([seed,arm,x['origin'],m['parent']['family'],m['source_time_ps'],m['index'],*metrics(x['metrics']['fine'])])
        c=np.array(json.loads((out/'exposures.json').read_text()));updates=16 if smoke else 6000
        assert c.shape==(3,48,64) and c.sum()==updates*48
        assert c[0].sum()==updates*24
        assert c[1].sum()==0 and c[2].sum()==updates*24
        aux=json.loads((out/'auxiliary.json').read_text());assert len(aux)==(4 if smoke else 187)
        assert all(x['active']==(arm!='control') and x['norm_error']<2e-6 for x in aux)
        if arm!='control':assert any(x['gradient_norm']>0 for x in aux)
        auxiliary.append([seed,arm,len(aux),max(x['gradient_norm'] for x in aux),run['auxiliary_seconds'],run['train_seconds'],run['evaluation_seconds']])
    for seed in ((11,) if smoke else (11,22,33)):
        pair=[r for r in runs if r['task']['seed']==seed];assert len(pair)==3
        for key in ('initial_hash','paired_input_hash','extra_noise_hash','schedule_hash','group_schedule_hash','restart_data_hashes','auxiliary_input_hash'):
            if any(pair[0][key]!=x[key] for x in pair[1:]):errors.append([seed,key])
    assert not errors,errors
    baseline=[]
    for seed in ((11,) if smoke else (11,22,33)):
        r=json.loads((source(seed)/'report.json').read_text())
        baseline.append([seed,'before',*np.mean([metrics(x['metrics']['fine']) for x in r['intermediate']],0).tolist()])
        for L in sorted({x['parent']['L'] for x in r['data']['rows'] if x['role']=='dev'}):
            ss=[x for x in r['data']['rows'] if x['role']=='dev' and x['parent']['L']==L and x['sampler']=='midpoint128']
            full.append([seed,'before',L,'midpoint128',*np.mean([brief(x['full']['fine']) for x in ss],0).tolist()])
    assets=HERE/('smoke_assets' if smoke else 'assets');assets.mkdir(exist_ok=True)
    dump(assets/'summary.json',dict(dev=dev,baseline=baseline,full=full,conditional=cond,runs=runs,errors=errors,auxiliary=auxiliary,
         dev_columns=['seed','arm','mean_RMSE','cov_RMSE','signed_variance_bias','energy_RMSE'],
         full_columns=['seed','arm','L','sampler','increment_variance_RMSE','two_time_RMSE','energy_RMSE'],
         conditional_columns=['seed','arm','origin','family','source_time_ps','state','mean_RMSE','cov_RMSE','signed_variance_bias','energy_RMSE'],
         scientific_certification=False))
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for k,ax in enumerate(axes.flat):
        for j,arm in enumerate(('before',)+ARMS):
            if k<4:values=[x[2+k] for x in baseline+dev if x[1]==arm]
            else:
                sampler='midpoint128' if arm=='before' or not smoke else 'midpoint4'
                values=[np.mean([x[4+k-4] for x in full if x[0]==s and x[1]==arm and x[3]==sampler]) for s in ((11,) if smoke else (11,22,33))]
            ax.bar(j,np.mean(values),color=('#999999','#0072B2','#E69F00','#009E73')[j],edgecolor='black',width=.65)
            ax.scatter(j+np.linspace(-.08,.08,len(values)),values,color='black',zorder=3)
        ax.set_xticks(range(4),['Before','FM','Mom 0.1','Mom 0.5']);ax.axhline(0,color='gray',lw=.7)
        ax.set_title(('DEV conditional mean','DEV conditional covariance','DEV signed variance bias','DEV energy (H/NJ)','Full 1ps increment variance','Full 1ps two-time correlation')[k])
    fig.suptitle('Same architecture and replay budget; dots = training seeds'+(' (SOFTWARE SMOKE; unequal evaluation counts)' if smoke else ''))
    fig.savefig(assets/'dev_comparison.png',dpi=180);plt.close(fig)
    body='<h1>部署采样统计辅助：当前配方下的受控修复</h1><p>从84144步refreshed-replay权重开始，3seed×3组：FM对照、矩损失峰值0.1、峰值0.5。均重新Adam、6000更新，200步热身到3e-4，余弦到9e-6；native监督及50%真实/50%刷新模型状态监督相同，使用相同经验teacher。控制数据/网络/随机输入，实际反传成本不同并单列。</p>'
    body+='<p>每32次更新选择一个TRAIN状态，用真实部署的128步归一化midpoint采样64分支，然后比较条件增量均值和局域3×3协方差。64生成分成独立32+32，两组差值做内积，减少直接平方随机均值引入的额外方差压缩。条件于有限64TRAIN目标，并不消除参考目标误差；损失允许负数，不把0当物理验收。权重1000步热身。FM对照计算相同诊断但不反传辅助项。</p>'
    body+='<p>旧conditional_moment_repair/training_restart_repair曾用这一辅助形式，但当时没有当前经验teacher与模型访问状态库及这轮优化起点。本轮检验在当前已改善配方上的增量效应，不声称全新方法或过去没做过。只用TRAIN物理后继，不用probe/fine/DEV；不训练能量、不要求有限温能量守恒。</p>'
    body+='<p>模型结构、晶格几何和物理条件不变；0.1ps转移、10次新latent接续成1ps，flow128/512与LLG .25/.125fs积分步分开。分别检查独立DEV、训练状态的新噪声、完整时间/空间统计；局域矩改善不等于整个联合路径正确。固定最终90144步，旧结果保留，不自动选优或晋升。</p>'
    if smoke:body+='<p>仅软件预检：16更新，4条模型路径，低flow步；历史基线为正式样本，不能从此图判断物理修复。</p>'
    body+='<img src="dev_comparison.png" style="max-width:100%"><p>柱为描述性均值，点为训练seed；不是总体置信区间。固定最终6000步，不自动选最好checkpoint。原始report保留固定状态整分支bootstrap100；来源/中间时间相关，聚合不能视为独立帧。重复使用DEV仍是开发，不是最终留出认证。</p>'
    body+='<h2>辅助项实际作用和成本</h2>'+table(['seed','arm','calls','max auxiliary gradient','auxiliary seconds','training seconds','evaluation seconds'],auxiliary)
    body+='<h2>独立开发真实状态</h2>'+table(['seed','arm','mean RMSE','cov RMSE','signed variance bias','energy RMSE'],baseline+dev)
    body+='<h2>完整1ps：按尺寸与采样器检查副作用</h2>'+table(['seed','arm','L','sampler','incvar RMSE','two-time RMSE','energy RMSE'],full)
    body+='<h2>训练来源状态的新噪声：真实/模型访问初态</h2>'+table(['seed','arm','origin','family','source ps','state','mean RMSE','cov RMSE','signed variance bias','energy RMSE'],cond)
    body+='<p>若只改善模型访问状态而独立DEV或完整1ps不改善，不能称为闭环修复。若FM对照同样改善，不能独归因于覆盖补齐。无预先确认的物理容差时不自动宣布通过。已保存完整生成轨迹、模型配置、RNG及实际计算成本；不做有效加速或尺寸迁移认证。</p><a href="summary.json">完整汇总</a> · <a href="../protocol.json">协议</a>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1600px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    if not smoke:
        path=ROOT/'tests/index.html';txt=path.read_text();start='<!-- deployment-moment-repair:start -->';end='<!-- deployment-moment-repair:end -->'
        block=start+'<section id="deployment-moment-repair"><h2>部署采样统计辅助修复对照</h2><a href="deployment_moment_repair/assets/index.html">查看本轮完整结果</a><img style="max-width:100%" src="deployment_moment_repair/assets/dev_comparison.png"></section>'+end
        txt=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,txt,flags=re.S) if start in txt else txt.replace('</body>',block+'</body>')
        tmp=path.with_suffix('.html.tmp');tmp.write_text(txt);tmp.replace(path)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['analyze','aggregate']);p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    verify();analyze(a.index,a.smoke) if a.action=='analyze' else aggregate(a.smoke)
