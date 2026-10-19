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
from .run import HERE,ROOT,tasks,source,dump,verify
from tests.joint_uncertainty_repair.metric import distance_with_uncertainty as distance_with_ci


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
         reversal=json.loads((out/'reversal.json').read_text()),gauge=json.loads((out/'gauge.json').read_text()),symmetry=json.loads((out/'symmetry.json').read_text()),scientific_certification=False))


def aggregate(smoke):
    reports=[json.loads((HERE/('smoke' if smoke else 'train')/f'{i:02d}/report.json').read_text()) for i in range(2 if smoke else 6)]
    runs=[r['run'] for r in reports]
    for seed in ((11,) if smoke else (11,22,33)):
        rr=[r for r in runs if r['task']['seed']==seed];assert len(rr)==2
        for key in ('initial_hash','paired_input_hash','extra_noise_hash','schedule_hash','group_schedule_hash','restart_data_hashes'):
            assert rr[0][key]==rr[1][key],key
    dev=[];full=[];conditional=[];joint=[];sym=[];gauge=[];cost=[]
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
        for x in r['symmetry']:sym.append([seed,arm,x['family'],x['tau'],x['angle'],x['absolute_rms'],x['relative_rms']])
        if arm=='reversal':assert max(x['relative_rms'] for x in r['symmetry'])<1e-4
        if arm=='reversal':assert max(x['relative_rms'] for x in r['reversal'])<1e-4
        gauge.append([seed,arm,dict(gauge=r['gauge'],reversal=r['reversal'])]);cost.append([seed,arm,run['parameter_count'],run['train_seconds'],run['evaluation_seconds']])
    assets=HERE/('smoke_assets' if smoke else 'assets');assets.mkdir(exist_ok=True)
    dump(assets/'summary.json',dict(dev=dev,full=full,conditional=conditional,joint=joint,symmetry=sym,gauge=gauge,cost=cost,runs=runs,pairing_errors=[],scientific_certification=False))
    body='<h1>Bauer链编号反转对称化修复</h1><p>三seed11/22/33均从continuous_axial_repair的gauge2最终114144步开始，control保留原输出，reversal平均原顺序及反转顺序的共享骨干运输速度。6000更新至120144，数据、native FM+0.25 teacher、轴向增广、Adam重启和配对随机流保持一致。参数相同，候选的骨干分支数2→4，计算成本不同。</p>'
    body+='<p>只针对已审计的无场、交换+单轴各向异性Bauer同质开放链和等价左右端点。反转格点编号，不反射或翻转磁矩分量；不是磁轴向量反射等变，也不套用DMI或非等价边界。已有三个checkpoint操作探针最大相对RMS约0.65/0.55/0.44，说明存在编号偏差，不证明这是物理误差唯一根因。共享骨干速度对称化不等于平均物理轨迹，不输入Hamiltonian系数、解析场或干净未来。</p>'
    body+='<p>轴向gauge坐标必须同时支持左右两端各两个锚点；任一端横向总权重≤1e-12拒绝。audit.json检查TRAIN库和复用DEV初态，两端分别统计；数量含相关重复，不是独立样本量。新接续状态逐调用检查。保持任意轴向SO2、joint spin/crystal SO3、切向速度、实际初帧、球面采样、固定开放边界。软件对称性不认证动力学。</p>'
    if smoke:body+='<p>16更新、少分支、低flow步只检查软件，不证明物理改善。</p>'
    body+='<h2>独立开发0.1ps条件统计</h2>'+table(['seed','arm','meanRMSE','covRMSE','variance bias','Hamiltonian energyRMSE'],dev)
    body+='<h2>完整1ps未优化时间／能量统计</h2>'+table(['seed','arm','L','sampler','incvarRMSE','two-timeRMSE','energyRMSE'],full)
    body+='<h2>联合距离修订版（非验收区间）</h2>'+table(['seed','arm','grid','L','state','U estimate/approximate interval/equality permutation'],joint)
    body+='<p>联合距离保留无偏点估计，采用v2中心化近似区间和999次整分支置换。上一轮已知真值强备择校验中该近似区间覆盖仅91.25%，不能当可靠95%置信区间或硬性通过证据；零／近零亦未认证。p未拒绝不证明等价，原始逐状态p未作多重校正。其他物理统计区间方法与样本单位见report.json，三seed不混为大量格点重复。</p>'
    body+='<h2>任意角度协变操作测试</h2>'+table(['seed','arm','family','tau','angle','absolute rms','relative rms'],sym)
    body+='<h2>坐标系与链编号反转操作探针</h2>'+table(['seed','arm','probe'],gauge)
    body+='<h2>实际成本</h2>'+table(['seed','arm','parameters','train seconds','eval seconds'],cost)
    body+='<h2>TRAIN来源独立新噪声</h2>'+table(['seed','arm','origin','family','state','meanRMSE','covRMSE','variance bias','energyRMSE'],conditional)
    body+='<p>所有生成轨迹保存。L16/25/32已参与训练，不是未见尺寸；单温度63.824849668K开放链，仅0.1ps转移、1ps接续。LLG.25/.125fs、保存.1ps、flow128/512不混用。DEV反复使用非最终留出，快照相关，跨状态均值是描述。对称性操作通过只是结构证据，不是物理通过；只有独立条件/全路径统计改善才支持本候选。不晋升生产、不修改事件阈值、不声称有效加速。</p><a href="summary.json">全部数据</a>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1600px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    if not smoke:
        path=ROOT/'tests/index.html';txt=path.read_text();start='<!-- chain-reversal-repair:start -->';end='<!-- chain-reversal-repair:end -->'
        block=start+'<section id="chain-reversal-repair"><h2>Bauer链编号反转修复候选</h2><a href="chain_reversal_repair/assets/index.html">三seed编号对称化对照、独立路径统计和成本</a></section>'+end
        txt=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,txt,flags=re.S) if start in txt else txt.replace('</body>',block+'</body>')
        tmp=path.with_suffix('.html.tmp');tmp.write_text(txt);tmp.replace(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['analyze','aggregate']);p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    verify();analyze(a.index,a.smoke) if a.action=='analyze' else aggregate(a.smoke)
