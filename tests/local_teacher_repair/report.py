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
from tests.spatial_covariance_repair.metrics import compare as spatial_compare
from .recipe import expected_calls
from tests.joint_uncertainty_repair.metric import distance_with_uncertainty as distance_with_ci


def analyze(index,smoke):
    torch.set_num_threads(1);out=HERE/('smoke' if smoke else 'train')/f'{index:02d}'
    run=json.loads((out/'complete.json').read_text());assert run['task']==tasks()[index]
    middle=[];conditional=[];joint_full=[];space=[];space_time=[]
    with h5py.File(out/'intermediate.h5') as h:
        assert h.attrs['complete']
        for name,g in h.items():
            meta=json.loads(g.attrs['metadata']);a=g['spins'][:]
            with h5py.File(ROOT/f"tests/intermediate_restart_repair/reference/{meta['parent_index']:02d}/paths.h5") as rh:
                assert rh.attrs['complete'];r=rh[f'{name}/fine/spins'][:]
                vals={k:compare_first(a,rh[f'{name}/{k}/spins'][:]) for k in ('base','fine')}
            middle.append(dict(metadata=meta,metrics=vals,joint_configuration_distance=distance_with_ci(a,r)))
            space.append(dict(metadata=meta,metrics=spatial_compare(a,r)))
    with h5py.File(out/'paths.h5') as h:
        assert h.attrs['complete']
        for name,g in h['dev'].items():
            i=int(name.removeprefix('parent'))
            with h5py.File(ROOT/f'tests/reference_resolution/reference/{i:02d}/paths.h5') as rh:
                assert rh.attrs['complete'];r=rh['fine/spins'][:]
            a=g['midpoint4' if smoke else 'midpoint128'][:]
            for k in (1,3,5,10):space_time.append(dict(parent=json.loads(g.attrs['parent']),time_ps=k*.1,metrics=spatial_compare(a[:,[0,k]],r[:,[0,k]])))
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
         space=space,space_time=space_time,auxiliary=json.loads((out/'auxiliary.json').read_text()),teacher=json.loads((out/'teacher.json').read_text()),reversal=json.loads((out/'reversal.json').read_text()),gauge=json.loads((out/'gauge.json').read_text()),symmetry=json.loads((out/'symmetry.json').read_text()),scientific_certification=False))


def aggregate(smoke):
    reports=[json.loads((HERE/('smoke' if smoke else 'train')/f'{i:02d}/report.json').read_text()) for i in range(2 if smoke else 6)]
    runs=[r['run'] for r in reports]
    for seed in ((11,) if smoke else (11,22,33)):
        rr=[r for r in runs if r['task']['seed']==seed];assert len(rr)==2
        for key in ('initial_hash','paired_input_hash','extra_noise_hash','schedule_hash','group_schedule_hash','restart_data_hashes','auxiliary_hash','shared64_auxiliary_hash'):
            assert all(rr[0][key]==r[key] for r in rr[1:]),key
        paired=[r['teacher'] for r in reports if r['run']['task']['seed']==seed]
        assert paired[0]==paired[1],'teacher drafts or posterior statistics unpaired'
        assert len(paired[0])==3*(16 if smoke else 2048)
    dev=[];full=[];conditional=[];joint=[];sym=[];gauge=[];cost=[];space=[];space_time=[];auxiliary=[];teacher=[]
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
        assert max(x['relative_rms'] for x in r['symmetry'])<1e-4
        assert max(x['relative_rms'] for x in r['reversal'])<1e-4
        for lag in (0,1,2,4,8):
            rows=[q for x in r['space'] for q in x['metrics']['lags'] if q['lag']==lag]
            space.append([seed,arm,lag,float(np.mean([q['covariance_rmse'] for q in rows])),float(np.mean([q['covariance_squared_bias_crossfold'] for q in rows])),float(np.mean([q['generated_trace']-q['reference_trace'] for q in rows]))])
        for x in r['space_time']:
            for q in x['metrics']['lags']:space_time.append([seed,arm,x['parent']['L'],x['parent']['parent_id'],x['time_ps'],q['lag'],q['covariance_rmse'],q['covariance_squared_bias_crossfold']])
        tr=r['teacher'];teacher.append([seed,arm,len(tr),*np.mean([[x['whole_effective_modes'],x['patch_effective_modes'],x['target_difference']] for x in tr],0).tolist()])
        records=r['auxiliary'];assert len(records)==expected_calls(16 if smoke else 2048,arm,smoke)
        active=True
        assert all(x['active']==active and (x['gradient_norm']>0 if active else x['gradient_norm']==0) for x in records)
        auxiliary.append([seed,arm,len(records),sum(x['seconds'] for x in records),float(np.mean([x['mean_loss'] for x in records])),float(np.mean([x['covariance_loss'] for x in records])),float(np.mean([x['spatial_loss'] for x in records])),float(np.mean([x['gradient_norm'] for x in records]))])
        gauge.append([seed,arm,dict(gauge=r['gauge'],reversal=r['reversal'])]);cost.append([seed,arm,run['parameter_count'],run['train_seconds'],run['evaluation_seconds']])
    assets=HERE/('smoke_assets' if smoke else 'assets');assets.mkdir(exist_ok=True)
    dump(assets/'summary.json',dict(teacher=teacher,space=space,space_time=space_time,auxiliary=auxiliary,dev=dev,full=full,conditional=conditional,joint=joint,symmetry=sym,gauge=gauge,cost=cost,runs=runs,pairing_errors=[],scientific_certification=False))
    body='<h1>Bauer局部经验教师修复</h1><p>三seed11/22/33各从auxiliary_frequency_repair的sparse64 step128192开始，2048更新至130240，同期whole/patch2对照。结构、参数、native数据、源潜变量、Adam重启、0.25 teacher和每64步0.1矩辅助固定。只有TRAIN经验教师的条件范围变化：整链对比开放半径2邻域，每个格点截取实际邻域，不循环填充。</p>'
    body+='<p>有限64未来库在8个预选TRAIN状态的tau0.5平均ESS整链约2、radius2约12；tau0.7约1/2.15。这是运输后验的经验模式数，不是物理事件数或热噪声。局部教师是较少信息下的经验条件期望，并非完整链的精确联合流速度。可能压低跨区关联；保留native全链FM，检查未优化完整路径和远程空间指标。不得因ESS提高宣称机制正确。</p>'
    body+='<p>两臂均计算两类标签，草稿/随机流/数据访问配对，只切换实际监督标签；tau大于0.9保留普通CFM，未截断运输域。独立父状态开发评估与TRAIN来源新噪声评估分开；历史复用DEV不是最终留出。不得使用低loss/合法模长代替物理验证，不自动晋升。</p>'
    body+='<h2>教师诊断（非物理事件权重）</h2>'+table(['seed','arm','groups','whole ESS mean (zero if no eligible tau)','patch ESS mean','target change'],teacher)
    if smoke:body+='<p>16更新、少分支、低flow步只检查软件，不证明物理改善。</p>'
    body+='<h2>独立开发距离分辨协方差</h2>'+table(['seed','arm','lag','tensor RMSE','crossfold squared bias','signed trace difference'],space)
    body+='<p>crossfold squared bias用独立半数生成/参考的协方差差值交叉积，理想独立分支下估计总体协方差平方差，可为负；单次拆分非置信区间，不把格点对计为独立样本，不用于硬验收。tensor RMSE仍有有限样本噪声。原冻结诊断另有参考base/fine对照。</p>'
    body+='<h2>辅助作用和实测成本</h2>'+table(['seed','arm','calls','aux seconds','mean term','local covariance term','spatial term','gradient norm'],auxiliary)
    body+='<h2>独立开发0.1ps条件统计</h2>'+table(['seed','arm','meanRMSE','covRMSE','variance bias','Hamiltonian energyRMSE'],dev)
    body+='<h2>完整1ps未优化时间／能量统计</h2>'+table(['seed','arm','L','sampler','incvarRMSE','two-timeRMSE','energyRMSE'],full)
    body+='<h2>联合距离修订版（非验收区间）</h2>'+table(['seed','arm','grid','L','state','U estimate/approximate interval/equality permutation'],joint)
    body+='<p>联合距离保留无偏点估计，采用v2中心化近似区间和999次整分支置换。上一轮已知真值强备择校验中该近似区间覆盖仅91.25%，不能当可靠95%置信区间或硬性通过证据；零／近零亦未认证。p未拒绝不证明等价，原始逐状态p未作多重校正。其他物理统计区间方法与样本单位见report.json，三seed不混为大量格点重复。</p>'
    body+='<h2>任意角度协变操作测试</h2>'+table(['seed','arm','family','tau','angle','absolute rms','relative rms'],sym)
    body+='<h2>坐标系与链编号反转操作探针</h2>'+table(['seed','arm','probe'],gauge)
    body+='<h2>1ps内分时刻空间涨落</h2>'+table(['seed','arm','L','parent','ps','lag','tensor RMSE','crossfold squared bias'],space_time)
    body+='<h2>实际成本</h2>'+table(['seed','arm','parameters','train seconds','eval seconds'],cost)
    body+='<h2>TRAIN来源独立新噪声</h2>'+table(['seed','arm','origin','family','state','meanRMSE','covRMSE','variance bias','energyRMSE'],conditional)
    body+='<p>所有生成轨迹保存。L16/25/32已参与训练，不是未见尺寸；单温度63.824849668K开放链，仅0.1ps转移、1ps接续。LLG.25/.125fs、保存.1ps、flow128/512不混用。DEV反复使用非最终留出，快照相关，跨状态均值是描述。对称性操作通过只是结构证据，不是物理通过；只有独立条件/全路径统计改善才支持本候选。不晋升生产、不修改事件阈值、不声称有效加速。</p><a href="summary.json">全部数据</a>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1600px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    if not smoke:
        path=ROOT/'tests/index.html';txt=path.read_text();start='<!-- local-teacher-repair:start -->';end='<!-- local-teacher-repair:end -->'
        block=start+'<section id="local-teacher-repair"><h2>Bauer局部经验教师修复</h2><a href="local_teacher_repair/assets/index.html">整链/局部教师三seed对照、路径和空间统计</a></section>'+end
        txt=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,txt,flags=re.S) if start in txt else txt.replace('</body>',block+'</body>')
        tmp=path.with_suffix('.html.tmp');tmp.write_text(txt);tmp.replace(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['analyze','aggregate']);p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    verify();analyze(a.index,a.smoke) if a.action=='analyze' else aggregate(a.smoke)
