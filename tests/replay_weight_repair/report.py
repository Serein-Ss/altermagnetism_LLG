import argparse,json,re
import h5py
import numpy as np
from tests.transition_block_repair.report import collect
from tests.transition_block_repair.metrics import compare_first
from tests.reference_resolution.report import brief
from tests.conditional_moment_repair.report import metrics
from tests.transport_audit.report import table
from tests.training_restart_repair.data import HERE as REAL,rows as real_rows
from tests.model_state_refresh_repair import data as visited
from tests.deployment_moment_repair.report import ref
from .run import HERE,ROOT,tasks,source,dump,verify
from .recipe import WEIGHTS


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
    for origin,filename,rows in (('llg','after_restart.h5',real_rows(smoke)),('model','model_states.h5',visited.rows(smoke))):
        with h5py.File(out/filename) as h:
            assert h.attrs['complete']
            for row in rows:
                name=f"state{row['index']:03d}";g=h[name];assert json.loads(g.attrs['metadata'])==row
                refs={}
                for role in ('probe','fine'):
                    if origin=='model':refs[role]=ref(row,role)
                    else:
                        with h5py.File(REAL/'reference'/f"{row['parent_index']:02d}"/f'{name}_{role}.h5') as rh:
                            assert rh.attrs['complete'] and rh.attrs['role']==role and json.loads(rh.attrs['metadata'])==row
                            refs[role]=rh['spins'][:]
                conditional.append(dict(origin=origin,metadata=row,metrics={k:compare_first(g['spins'][:],v) for k,v in refs.items()}))
    dump(out/'report.json',dict(run=run,data=collect(out/'paths.h5',smoke),intermediate=middle,conditional=conditional,
         symmetry=json.loads((out/'symmetry.json').read_text()),gradients=json.loads((out/'gradients.json').read_text()),scientific_certification=False))


def aggregate(smoke):
    reports=[json.loads((HERE/('smoke' if smoke else 'train')/f'{i:02d}/report.json').read_text()) for i in range(3 if smoke else 9)]
    runs=[r['run'] for r in reports];seeds=(11,) if smoke else (11,22,33)
    for seed in seeds:
        pair=[r for r in reports if r['run']['task']['seed']==seed];assert len(pair)==3
        for key in ('initial_hash','paired_input_hash','extra_noise_hash','schedule_hash','group_schedule_hash','restart_data_hashes'):
            assert all(pair[0]['run'][key]==r['run'][key] for r in pair),key
        for key in ('native_norm','replay_norm','cosine'):
            assert all(np.isclose(pair[0]['gradients'][0][key],r['gradients'][0][key],rtol=1e-5,atol=1e-6) for r in pair),key
    dev=[];full=[];conditional=[];grad=[];baseline=[]
    for r in reports:
        run=r['run'];seed=run['task']['seed'];arm=run['task']['repair']
        dev.append([seed,arm,*np.mean([metrics(x['metrics']['fine']) for x in r['intermediate']],0).tolist()])
        for L in sorted({x['parent']['L'] for x in r['data']['rows'] if x['role']=='dev'}):
            for sampler in (('midpoint4','midpoint8') if smoke else ('midpoint128','midpoint512')):
                rr=[x for x in r['data']['rows'] if x['role']=='dev' and x['parent']['L']==L and x['sampler']==sampler]
                full.append([seed,arm,L,sampler,*np.mean([brief(x['full']['fine']) for x in rr],0).tolist()])
        for x in r['conditional']:
            m=x['metadata'];conditional.append([seed,arm,x['origin'],m['parent']['family'],m['index'],*metrics(x['metrics']['fine'])])
        values=r['gradients'];assert len(values)==(16 if smoke else 47)
        assert all(np.isfinite(x['native_norm']) and np.isfinite(x['replay_norm']) for x in values)
        grad.append([seed,arm,np.mean([x['native_norm'] for x in values]),np.mean([x['replay_norm'] for x in values]),
                     np.mean([x['cosine'] for x in values if x['cosine'] is not None]),np.mean([x['weighted_ratio'] for x in values if x['weighted_ratio'] is not None]),
                     run['train_seconds'],run['evaluation_seconds']])
    for seed in seeds:
        r=json.loads((source(seed)/'report.json').read_text());baseline.append([seed,'before',*np.mean([metrics(x['metrics']['fine']) for x in r['intermediate']],0).tolist()])
    assets=HERE/('smoke_assets' if smoke else 'assets');assets.mkdir(exist_ok=True)
    dump(assets/'summary.json',dict(dev=dev,baseline=baseline,full=full,conditional=conditional,gradients=grad,runs=runs,pairing_errors=[],scientific_certification=False))
    body='<h1>原始窗口—额外快照监督权重修复对照</h1><p>三seed×权重1/0.25/0；相同90144步FM控制组模型为起点，重置Adam，6000次更新到96144。模型、原始窗口和分支库、teacher和物理设置不变。额外项未加权梯度在所有组都计算，所以零权重组是受控计算，不代表节省后的最优原始FM训练成本。</p>'
    body+='<p>完整覆盖半径2平均距离约0.074，旧增强快照约0.233；多数最近邻来自native库。构型邻近不证明动力学等同，也不是覆盖认证。当前检验梯度分配而非扩大数据。旧balanced_replay比较更早的采样密度/平衡且没有当前模型状态和经验teacher；本轮不是声称此前从未检验权重。</p>'
    body+='<p>每次原始batch8与额外48样本各自平均；样本数比例不等于梯度比例。记录未裁剪原始/额外梯度范数和夹角，不能用负夹角独自证明有害，更不能用梯度改善替代物理结果。三组相同seed配对输入及首步梯度核查。不同加权优化路径之后梯度当然可不同。</p>'
    if smoke:body+='<p>软件预检：16更新、少量生成、低flow步；以下不能判断物理改善。历史before是正式数据，不与预检样本数匹配。</p>'
    body+='<h2>独立开发状态：均值、协方差、方差、能量</h2>'+table(['seed','arm','meanRMSE','covRMSE','signed variance bias','energyRMSE'],baseline+dev)
    body+='<h2>完整1ps路径与flow收敛</h2>'+table(['seed','arm','L','sampler','increment variance RMSE','two-time RMSE','energyRMSE'],full)
    body+='<h2>梯度与实际成本</h2>'+table(['seed','arm','native norm','replay norm','cosine','weighted ratio','train seconds','eval seconds'],grad)
    body+='<h2>TRAIN来源的新噪声诊断</h2>'+table(['seed','arm','origin','family','state','meanRMSE','covRMSE','signed variance bias','energyRMSE'],conditional)
    body+='<p>保留48真实/48模型访问TRAIN状态的独立probe/fine与72真实DEV快照，以及18DEV完整路径、24旧TRAIN首态；固定最终模型，不选最好seed。固定状态整分支bootstrap100在各report.json，不把来源关联窗口或格点当独立实验。重复DEV不是最终留出。0.1ps转移、1ps接续、LLG .25/.125fs和flow128/512分开；无阈值放宽、无事件硬门槛、无尺寸迁移或速度认证。只有独立路径统计改善且副作用可接受才支持修复；失败也如实保留。</p><a href="summary.json">汇总JSON</a>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1500px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    if not smoke:
        path=ROOT/'tests/index.html';txt=path.read_text();start='<!-- replay-weight-repair:start -->';end='<!-- replay-weight-repair:end -->'
        block=start+'<section id="replay-weight-repair"><h2>原始窗口—快照梯度分配修复对照</h2><a href="replay_weight_repair/assets/index.html">三seed×三监督权重、完整路径和梯度诊断</a></section>'+end
        txt=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,txt,flags=re.S) if start in txt else txt.replace('</body>',block+'</body>')
        tmp=path.with_suffix('.html.tmp');tmp.write_text(txt);tmp.replace(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['analyze','aggregate']);p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    verify();analyze(a.index,a.smoke) if a.action=='analyze' else aggregate(a.smoke)
