import argparse,hashlib,json,re,time
from pathlib import Path
import h5py
import numpy as np
import torch
from tests.bauer_p2_20260923.common import dump
from tests.spatial_gate_repair.run import verify as verify_previous
from tests.transport_audit.report import table
from .metric import distance_with_uncertainty,summarize

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def tasks():
    return [dict(campaign=c,index=i) for c,n in (('joint_score_repair',6),('spatial_gate_repair',9)) for i in range(n)]


def inputs(task):
    out=ROOT/'tests'/task['campaign']/'train'/f"{task['index']:02d}"
    files=[out/name for name in ('report.json','paths.h5','intermediate.h5')]
    with h5py.File(out/'intermediate.h5') as h:
        files += [ROOT/f"tests/intermediate_restart_repair/reference/{json.loads(g.attrs['metadata'])['parent_index']:02d}/paths.h5" for g in h.values()]
    with h5py.File(out/'paths.h5') as h:
        files += [ROOT/f"tests/reference_resolution/reference/{int(name.removeprefix('parent')):02d}/paths.h5" for name in h['dev']]
    return files


def prepare():
    assert not (HERE/'source_lock.json').exists();verify_previous()
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json']
    for task in tasks():files+=inputs(task)
    dump(HERE/'source_lock.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in set(files)})


def verify():
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name


def analyze(index,smoke):
    task=tasks()[index];source=ROOT/'tests'/task['campaign']/'train'/f"{task['index']:02d}"
    old=json.loads((source/'report.json').read_text());rows=[];tick=time.perf_counter()
    kwargs=dict(repeats=99 if smoke else 999,permutations=99 if smoke else 999)
    with h5py.File(source/'intermediate.h5') as h:
        assert h.attrs['complete']
        for k,(name,g) in enumerate(h.items()):
            if smoke and k:break
            meta=json.loads(g.attrs['metadata'])
            with h5py.File(ROOT/f"tests/intermediate_restart_repair/reference/{meta['parent_index']:02d}/paths.h5") as rh:
                a=g['spins'][:];r=rh[f'{name}/fine/spins'][:]
            result=distance_with_uncertainty(a,r,**kwargs)
            previous=next(x['joint_configuration_distance'] for x in old['intermediate'] if x['metadata']==meta)
            assert np.isclose(result['estimate'],previous['estimate'],rtol=1e-10,atol=1e-12)
            rows.append(dict(grid='configuration',metadata=meta,legacy=previous,corrected=result))
    with h5py.File(source/'paths.h5') as h:
        assert h.attrs['complete']
        for k,(name,g) in enumerate(h['dev'].items()):
            if smoke and k:break
            i=int(name.removeprefix('parent'));parent=json.loads(g.attrs['parent'])
            with h5py.File(ROOT/f'tests/reference_resolution/reference/{i:02d}/paths.h5') as rh:r=rh['fine/spins'][:]
            for sampler in ('midpoint128','midpoint512'):
                result=distance_with_uncertainty(g[sampler][:],r,**kwargs)
                previous=next(x['joint_path_distance'] for x in old['joint_full'] if x['parent']==parent and x['sampler']==sampler)
                assert np.isclose(result['estimate'],previous['estimate'],rtol=1e-10,atol=1e-12)
                rows.append(dict(grid=sampler,metadata=parent,legacy=previous,corrected=result))
    out=HERE/('smoke' if smoke else 'reports');out.mkdir(exist_ok=True)
    dump(out/f'{index:02d}.json',dict(task=task,model_task=old['run']['task'],rows=rows,seconds=time.perf_counter()-tick,point_estimates_unchanged=True,scientific_certification=False))


def calibration(smoke):
    # Binary unit-spin endpoint law: population energy distance=4(p-q)^2.
    rng=np.random.default_rng(979700);rows=[];tick=time.perf_counter()
    trials=20 if smoke else 400;n=64;reps=99 if smoke else 499
    for p,q in ((.5,.5),(.5,.1),(.5,.48)):
        truth=4*(p-q)**2;old=[];new=[];reject=[];bias=[]
        for _ in range(trials):
            g=rng.random(n)<p;r=rng.random(n)<q
            gg=2*(g[:,None]!=g[None]);gr=2*(g[:,None]!=r[None]);rr=2*(r[:,None]!=r[None])
            s=summarize(gg.astype(float),gr.astype(float),rr.astype(float),repeats=reps,permutations=199,seed=int(rng.integers(2**31)))
            old.append(s['legacy_raw_percentile95'][0]<=truth<=s['legacy_raw_percentile95'][1])
            new.append(s['centered_basic_approx95'][0]<=truth<=s['centered_basic_approx95'][1]);reject.append(s['permutation_equality_p']<=.05)
            bias.append(s['observed_bootstrap_center_shift']-s['exact_bootstrap_center_shift'])
        rows.append(dict(p=p,q=q,population_distance=truth,trials=trials,branches_each=n,legacy_coverage=float(np.mean(old)),centered_coverage=float(np.mean(new)),
            equality_rejection_fraction=float(np.mean(reject)),bootstrap_center_residual_mean=float(np.mean(bias)),coverage_MC_standard_error_bound=float(.5/np.sqrt(trials))))
    dump(HERE/('smoke_calibration.json' if smoke else 'calibration.json'),dict(rows=rows,seconds=time.perf_counter()-tick,
        scope='Exploratory finite-support calibration, not proof for high-dimensional spin ensembles. Null/near-null basic interval remains uncertified; permutation nonrejection is not equivalence.'))


def aggregate(smoke):
    reports=[json.loads((HERE/('smoke' if smoke else 'reports')/f'{i:02d}.json').read_text()) for i in ((0,6,8) if smoke else range(15))]
    calibration=json.loads((HERE/('smoke_calibration.json' if smoke else 'calibration.json')).read_text())
    rows=[];overview=[]
    for report in reports:
        mt=report['model_task'];task=report['task']
        for grid in ('configuration','midpoint128','midpoint512'):
            rr=[r for r in report['rows'] if r['grid']==grid]
            overview.append([task['campaign'],mt['seed'],mt['repair'],grid,len(rr),np.mean([r['corrected']['estimate'] for r in rr]),np.mean([r['corrected']['exact_bootstrap_center_shift'] for r in rr])])
        for r in report['rows']:
            s=r['corrected'];meta=r['metadata'];ident=meta['index'] if 'index' in meta else meta['parent_id']
            rows.append([task['campaign'],mt['seed'],mt['repair'],r['grid'],ident,s['estimate'],r['legacy']['pointwise_ci95'],s['centered_basic_approx95'],s['permutation_equality_p'],s['exact_bootstrap_center_shift']])
    assets=HERE/('smoke_assets' if smoke else 'assets');assets.mkdir(exist_ok=True)
    dump(assets/'summary.json',dict(reports=reports,calibration=calibration,point_estimates_unchanged=True,scientific_certification=False))
    body='<h1>联合距离不确定性修复（评估器v2）</h1><p>原点估计为无偏U统计量，双方独立整分支有放回bootstrap的条件均值却是经验V统计量。旧raw-percentile区间未处理两者差，存在可精确计算的上移：mean(Dgg)/(n-1)+mean(Drr)/(m-1)。不是格点独立性问题，也不是模型或真实LLG数据错误。</p>'
    body+='<p>保留原U点估计。新近似区间使用bootstrapU减经验V的中心化误差，再对误差分位数反射；仅适用于非退化备择的近似，零／近零距离不保证覆盖。另按完整轨迹交换标签，报告999次加一校正的单侧相等分布置换检验。非拒绝不等于等价，无物理通过阈值；逐状态原始p未做多重比较，不能用于整批验收。区间不裁零。冻结旧代码、JSON和网页均保留，仅新增修订版。</p>'
    if smoke:body+='<p>仅三个来源模型、各一个条件状态／父状态的软件预检，99重采样；不能代替正式结果。</p>'
    body+='<h2>上一轮物理结论未被改变</h2><p>空间门控三seed原通路meanRMSE约0.066799，关闭约0.146667；完整1ps两时RMSE原通路0.018325，关闭0.067288。减弱也没有全面改善。当前证据不支持移除全局通路；关闭成熟模型且短预算适应的负结果也不证明所有从头训练局域模型无效。这里只修复联合距离区间，不能消除已有模型均值／协方差／时间偏差。</p>'
    body+='<h2>来源与点估计（保持不变）</h2>'+table(['campaign','seed','arm','grid','fixed states','mean distance','mean bootstrap center shift'],overview)
    body+='<h2>已知真值有限支持校验</h2>'+table(list(calibration['rows'][0]),[list(r.values()) for r in calibration['rows']])
    body+='<h2>逐固定状态结果</h2>'+table(['campaign','seed','arm','grid','state','U estimate','legacy95','centered basic approx95','equality p','center shift'],rows)
    body+='<p>0.1ps完整构型与1ps全保存网格分别评估。双方整独立分支重采样，所有时空结构保留；相关快照不作独立总体样本，训练seed独立列出。旧两轮6+9模型均用已保存轨迹，不重新训练、不新增LLG数据，不修改loss或物理网格，不称最终留出认证／加速。<a href="summary.json">全部结果与方法标签</a></p>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1700px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    if not smoke:
        path=ROOT/'tests/index.html';txt=path.read_text();start='<!-- joint-uncertainty-repair:start -->';end='<!-- joint-uncertainty-repair:end -->'
        block=start+'<section id="joint-uncertainty-repair"><h2>联合距离区间修订版：旧区间存在上移偏差</h2><p>模型点估计和物理偏差不变；旧联合距离区间需参照本修订，不作为认证依据。</p><a href="joint_uncertainty_repair/assets/index.html">6+9模型重算、置换检验与校验</a></section>'+end
        txt=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,txt,flags=re.S) if start in txt else txt.replace('</body>',block+'</body>')
        tmp=path.with_suffix('.html.tmp');tmp.write_text(txt);tmp.replace(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','analyze','calibration','aggregate']);p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(1)
    if a.action=='prepare':prepare()
    else:
        verify()
        if a.action=='analyze':analyze(a.index,a.smoke)
        elif a.action=='calibration':calibration(a.smoke)
        else:aggregate(a.smoke)
