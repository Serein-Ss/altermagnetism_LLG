import argparse,json,re
import h5py
import numpy as np
import torch
from tests.transition_block_repair.report import collect
from tests.transition_block_repair.metrics import compare_first
from tests.reference_resolution.report import brief
from tests.conditional_moment_repair.report import metrics
from tests.transport_audit.report import table
from tests.training_restart_repair.data import HERE as REAL,rows as real_rows
from tests.model_state_refresh_repair import data as visited
from tests.deployment_moment_repair.report import ref
from .run import HERE,ROOT,tasks,source,dump,verify,ARMS
from .objective import distance_with_ci


def analyze(index,smoke):
    torch.set_num_threads(1);out=HERE/('smoke' if smoke else 'train')/f'{index:02d}'
    run=json.loads((out/'complete.json').read_text());assert run['task']==tasks()[index]
    middle=[];conditional=[];joint_full=[]
    with h5py.File(out/'intermediate.h5') as h:
        assert h.attrs['complete']
        for name,g in h.items():
            meta=json.loads(g.attrs['metadata']);a=g['spins'][:]
            with h5py.File(ROOT/f"tests/intermediate_restart_repair/reference/{meta['parent_index']:02d}/paths.h5") as rh:
                assert rh.attrs['complete'];r=rh[f'{name}/fine/spins'][:]
                vals={k:compare_first(a,rh[f'{name}/{k}/spins'][:]) for k in ('base','fine')}
            middle.append(dict(metadata=meta,metrics=vals,joint_configuration_distance=distance_with_ci(a,r)))
    with h5py.File(out/'paths.h5') as h:
        assert h.attrs['complete']
        for name,g in h['dev'].items():
            i=int(name.removeprefix('parent'))
            with h5py.File(ROOT/f'tests/reference_resolution/reference/{i:02d}/paths.h5') as rh:
                assert rh.attrs['complete'];r=rh['fine/spins'][:]
            for sampler in (('midpoint4','midpoint8') if smoke else ('midpoint128','midpoint512')):
                joint_full.append(dict(parent=json.loads(g.attrs['parent']),sampler=sampler,joint_path_distance=distance_with_ci(g[sampler][:],r)))
    for origin,filename,rows in (('llg','after_restart.h5',real_rows(smoke)),('model','model_states.h5',visited.rows(smoke))):
        with h5py.File(out/filename) as h:
            assert h.attrs['complete']
            for row in rows:
                name=f"state{row['index']:03d}";g=h[name];assert json.loads(g.attrs['metadata'])==row;refs={}
                for role in ('probe','fine'):
                    if origin=='model':refs[role]=ref(row,role)
                    else:
                        with h5py.File(REAL/'reference'/f"{row['parent_index']:02d}"/f'{name}_{role}.h5') as rh:
                            assert rh.attrs['complete'] and rh.attrs['role']==role and json.loads(rh.attrs['metadata'])==row;refs[role]=rh['spins'][:]
                conditional.append(dict(origin=origin,metadata=row,metrics={k:compare_first(g['spins'][:],v) for k,v in refs.items()}))
    dump(out/'report.json',dict(run=run,data=collect(out/'paths.h5',smoke),intermediate=middle,conditional=conditional,joint_full=joint_full,
         auxiliary=json.loads((out/'auxiliary.json').read_text()),symmetry=json.loads((out/'symmetry.json').read_text()),scientific_certification=False))


def aggregate(smoke):
    reports=[json.loads((HERE/('smoke' if smoke else 'train')/f'{i:02d}/report.json').read_text()) for i in range(2 if smoke else 6)]
    runs=[r['run'] for r in reports];seeds=(11,) if smoke else (11,22,33)
    for seed in seeds:
        pair=[r for r in reports if r['run']['task']['seed']==seed];assert len(pair)==2
        for key in ('initial_hash','paired_input_hash','extra_noise_hash','schedule_hash','group_schedule_hash','restart_data_hashes','auxiliary_input_hash'):
            assert pair[0]['run'][key]==pair[1]['run'][key],key
    dev=[];full=[];aux=[];joint=[];baseline=[];conditional=[]
    for r in reports:
        run=r['run'];seed=run['task']['seed'];arm=run['task']['repair']
        dev.append([seed,arm,*np.mean([metrics(x['metrics']['fine']) for x in r['intermediate']],0).tolist()])
        for L in sorted({x['parent']['L'] for x in r['data']['rows'] if x['role']=='dev'}):
            for sampler in (('midpoint4','midpoint8') if smoke else ('midpoint128','midpoint512')):
                rr=[x for x in r['data']['rows'] if x['role']=='dev' and x['parent']['L']==L and x['sampler']==sampler]
                full.append([seed,arm,L,sampler,*np.mean([brief(x['full']['fine']) for x in rr],0).tolist()])
        for x in r['conditional']:
            m=x['metadata'];conditional.append([seed,arm,x['origin'],m['parent']['family'],m['index'],*metrics(x['metrics']['fine'])])
        for x in r['intermediate']:joint.append([seed,arm,'configuration',x['metadata']['parent']['L'],x['metadata']['index'],x['joint_configuration_distance']])
        for x in r['joint_full']:joint.append([seed,arm,x['sampler'],x['parent']['L'],x['parent']['parent_id'],x['joint_path_distance']])
        a=r['auxiliary'];assert len(a)==(4 if smoke else 187) and all(x['norm_error']<2e-6 and x['active']==(arm=='joint_score') for x in a)
        if arm=='joint_score':assert any(x['gradient_norm']>0 for x in a)
        aux.append([seed,arm,len(a),max(x['gradient_norm'] for x in a),run['auxiliary_seconds'],run['train_seconds'],run['evaluation_seconds']])
    for seed in seeds:
        r=json.loads((source(seed)/'report.json').read_text());baseline.append([seed,'before',*np.mean([metrics(x['metrics']['fine']) for x in r['intermediate']],0).tolist()])
    assets=HERE/('smoke_assets' if smoke else 'assets');assets.mkdir(exist_ok=True)
    dump(assets/'summary.json',dict(dev=dev,baseline=baseline,full=full,joint=joint,conditional=conditional,auxiliary=aux,runs=runs,pairing_errors=[],scientific_certification=False))
    body='<h1>联合磁构型分布评分修复</h1><p>从三seed96144步weight0.25模型出发，配对FM对照与FM+联合评分；源组经重复DEV选择，非最终留出认证。所有组native FM+0.25经验teacher相同，6000更新到102144，Adam重启/热身配方相同。模型结构、数据、0.1ps物理lag和开放边界不变。</p><p>每32更新选择一个TRAIN真实/模型访问状态，128步真实部署midpoint生成64未来，分为独立32+32。比较整个链端点向量的欧氏距离（按sqrt(N)归一化）：生成—参考吸引、生成—生成排斥、参考常数。目标仅64TRAIN经验分布，不把它当真实LLG律。不平均轨迹、不逐帧随机配对、不把格点当独立样本。峰值权重0.1，1000步热身。原始统计估计可负，不裁零。</p>'
    body+='<p>这叫统计energy score/energy distance，不是Hamiltonian能量；没有引入解析场、能量守恒或Hamiltonian训练知识。总体评分性质并不保证有限样本高维灵敏度、可优化性或物理准确。<a href="https://doi.org/10.1198/016214506000001437">Gneiting &amp; Raftery (2007)</a>。对照计算同一评分但不反传，成本分别记录。</p>'
    if smoke:body+='<p>仅软件预检，16更新、低步数少生成；before为正式结果，不据此判断物理修复。</p>'
    body+='<h2>开发条件局部统计</h2>'+table(['seed','arm','meanRMSE','covRMSE','signed variance bias','Hamiltonian energy RMSE'],baseline+dev)
    body+='<h2>完整1ps未优化物理统计</h2>'+table(['seed','arm','L','sampler','incvarRMSE','two-timeRMSE','Hamiltonian energy RMSE'],full)
    body+='<h2>联合距离与整分支bootstrap100</h2>'+table(['seed','arm','grid','L','state/parent','statistical distance and CI'],joint)
    body+='<h2>实际辅助梯度和计算成本</h2>'+table(['seed','arm','calls','max gradient','aux seconds','train seconds','eval seconds'],aux)
    body+='<h2>训练来源独立新噪声</h2>'+table(['seed','arm','origin','family','state','meanRMSE','covRMSE','variance bias','Hamiltonian energyRMSE'],conditional)
    body+='<p>开发未来从不用于训练；完整路径指标包括空间/时间关联，联合距离也按保存的1ps全部帧计算。置信区间为固定状态整独立分支bootstrap100，原指标同法；来源组/快照相关，跨状态均值是描述，不是总体CI。含零不证明等价，未冻结容差不自动通过。symmetry.json是spin-only轴向诊断，与联合SO3结构协变分开；保留真实S0、球面模长/切向和绑定开放边界。无生产晋升、事件认证、尺寸/长时推广或有效加速结论。</p><a href="summary.json">全部数值</a>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1600px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    if not smoke:
        path=ROOT/'tests/index.html';txt=path.read_text();start='<!-- joint-score-repair:start -->';end='<!-- joint-score-repair:end -->';block=start+'<section id="joint-score-repair"><h2>联合磁构型分布评分修复</h2><a href="joint_score_repair/assets/index.html">三seed配对物理统计、联合距离与成本</a></section>'+end
        txt=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,txt,flags=re.S) if start in txt else txt.replace('</body>',block+'</body>')
        tmp=path.with_suffix('.html.tmp');tmp.write_text(txt);tmp.replace(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['analyze','aggregate']);p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    verify();analyze(a.index,a.smoke) if a.action=='analyze' else aggregate(a.smoke)
