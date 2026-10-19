"""Finite-sample calibration and failure-aware campaign report; no automatic pass."""
import argparse
import hashlib
import html
import json
import time
import numpy as np
from .run import HERE, P2, ROOT, dump, references, comparison
from tests.bauer_p2_diagnostics_20260923.run import table


def finite_sample(generated, reference, seed):
    rng=np.random.default_rng(seed);ref=[];model=[]
    for _ in range(32):
        # Two DISJOINT sets, both conditioned on exactly the same parent.
        order=rng.permutation(len(reference))
        a,b=reference[order[:8]],reference[order[8:16]]
        ref.append({k:v['rmse'] for k,v in comparison(a,b).items()})
        if generated is not None:
            g=generated[rng.choice(len(generated),8,replace=False)]
            model.append({k:v['rmse'] for k,v in comparison(g,a).items()})
    def summarize(rows):
        return {k:dict(mean=float(np.mean([r[k] for r in rows])),
                       resplit_quantiles=np.quantile([r[k] for r in rows],[.05,.5,.95]).tolist()) for k in rows[0]}
    return dict(reference_reference=summarize(ref), model_reference=summarize(model) if model else None,
                note='overlapping re-splits are descriptive variability, NOT 32 independent experiments or confidence intervals')


def audit():
    started=time.perf_counter();result=[];ids={};initial_hashes={};coverage=[]
    for split in ('train','dev','diagnostic'):
        file=P2/f'manifest_{split}.json'
        if not file.exists():continue
        rows=json.loads(file.read_text());ids[split]=[r['parent_id'] for r in rows]
        assert len(ids[split])==len(set(ids[split])),f'duplicate parent in {split}'
    for a in ids:
        for b in ids:
            if a<b:assert not set(ids[a]) & set(ids[b]),f'parent leakage {a}/{b}'
    for split in ('train','dev'):
        for row,truth in references('bauer',split):
            assert np.isfinite(truth).all()
            assert np.max(np.abs(np.linalg.norm(truth,axis=-1)-1))<2e-6
            assert np.max(np.abs(truth[:,0]-truth[:1,0]))<2e-6
            digest=hashlib.sha256(truth[0,0].tobytes()).hexdigest()
            if digest in initial_hashes:
                assert initial_hashes[digest]==split,'exact initial configuration duplicated across splits'
            initial_hashes[digest]=split
            from tests.bauer_p2_20260923.common import energy
            coverage.append(dict(parent_id=row['parent_id'],split=split,L=row['L'],family=row['family'],
                initial_mz=float(truth[0,0,:,2].mean()),initial_energy_per_spin_over_J=float(energy(truth[0,0])),
                initial_neighbor_dot=float((truth[0,0,:-1]*truth[0,0,1:]).sum(-1).mean())))
            if split=='dev':
                result.append(dict(parent_id=row['parent_id'],L=row['L'],family=row['family'],
                    finite_sample=finite_sample(None,truth,923700+len(result))))
    dump(HERE/'reference_audit.json',dict(parents=result,coverage=coverage,
        split_counts={k:len(v) for k,v in ids.items()},exact_initial_check='18 selected train calibration parents versus all 36 dev parents; not a complete configuration-space coverage proof',
        interpretation='descriptive reference sampling scale, not acceptance tolerance; old P0 full audit and immutable data retained',
        seconds=time.perf_counter()-started))


def report():
    protocol=json.loads((HERE/'protocol.json').read_text())
    parts=['<h1>Bauer 全面诊断：数据、学习能力与网络设计</h1>',
        '<p>本轮为开发诊断。真实体系仍为L16/25/32开放链、固定温度、1 ps与0.1 ps保存网格。合成路径仅用于已知规律学习测试，不是LLG参考，也没有真实ps单位。</p>',
        '<p>固定最终训练步作主要比较；三个训练seed全部展示；不凭loss、相对最好或参考分样范围宣布物理通过。尚无独立最终认证或大尺寸迁移结论。</p>',
        table(['对照','要检验的因素'],[[a,html.escape(d)] for a,d in protocol['controls'].items()])]
    parts.append('<h2>计算状态与主要指标</h2>')
    runs=[];missing=[];display=[];keys=None
    for index,task in enumerate(protocol['tasks']):
        out=HERE/'runs'/f'{index:02d}_{task["domain"]}_{task["arm"]}_{task["seed"]}'
        file=out/'complete.json'
        if not file.exists():
            missing.append(dict(index=index,**task));continue
        complete=json.loads(file.read_text());final=json.loads((out/'dev_final128.json').read_text())
        keys=list(final['mean_parent_RMSE'])
        display.append([index,task['domain'],task['arm'],task['seed'],complete['parameters'],
                        f'{complete["seconds"]/60:.2f}']+[f'{final["mean_parent_RMSE"][k]:.5f}' for k in keys])
        calibration=[];convergence=[]
        for j,p in enumerate(sorted((out/'generated_dev_128').glob('*.npz'))):
            with np.load(p) as z:
                g=z['generated'];r=z['reference']
            calibration.append(dict(parent_id=p.stem,**finite_sample(g,r,923710+j)))
            other=out/'generated_dev_256'/p.name
            with np.load(other) as z:hi=z['generated']
            # Same latent source and model: direct path change is integration error,
            # NOT a requirement to pair independent physical LLG paths.
            convergence.append(dict(parent_id=p.stem,common_source_path_RMSE=float(np.sqrt(np.mean((g[:16]-hi)**2))),
                statistics_change=comparison(g[:16],hi)))
        dump(out/'matched_sample_calibration.json',calibration)
        dump(out/'sampler_common_source.json',convergence)
        runs.append(dict(index=index,task=task,complete=complete,metrics=final['mean_parent_RMSE']))
    if keys:parts.append(table(['ID','数据','候选','seed','参数','总分钟']+keys,display))
    parts.append(f'<p>已有完成标记：{len(runs)}/{len(protocol["tasks"])}；缺少完成标记：{html.escape(str(missing))}。缺失不会被默认为通过或跳过。</p>')
    # Stream equality audit: source and capacity interventions share underlying
    # noise draws/targets; anchors deliberately change target sampling.
    pairing=[]
    for domain in ('bauer','deterministic','gaussian','mixture'):
        for seed in (11,22,33):
            group=[r for r in runs if r['task']['domain']==domain and r['task']['seed']==seed and r['task']['arm']!='anchors']
            hashes={r['complete']['common_stream_sha256'] for r in group}
            pairing.append(dict(domain=domain,seed=seed,runs=len(group),recorded_streams_equal=len(hashes)<=1))
    parts.append('<h2>共同抽样流核验</h2>'+table(['数据','seed','完成候选数','前3000步记录流相同'],
                 [[p['domain'],p['seed'],p['runs'],p['recorded_streams_equal']] for p in pairing]))
    contrasts=[]
    if keys:
        pairs=[('baseline','wide'),('baseline','local_extra'),('local_extra','temporal'),
               ('temporal','global_temporal'),('baseline','source'),('temporal','temporal_source'),('baseline','anchors'),('baseline','baseline_long'),('temporal_source','temporal_source_long'),('global_temporal','global_temporal_long')]
        for domain in ('bauer','deterministic','gaussian','mixture'):
            for a,b in pairs:
                for seed in (11,22,33):
                    candidates={r['task']['arm']:r for r in runs if r['task']['domain']==domain and r['task']['seed']==seed}
                    if a in candidates and b in candidates:
                        delta={k:candidates[b]['metrics'][k]-candidates[a]['metrics'][k] for k in keys}
                        contrasts.append(dict(domain=domain,seed=seed,control=a,treatment=b,rmse_change=delta))
        parts.append('<h2>预先指定对照：RMSE变化</h2><p>负数代表此项误差下降，正数代表上升；不是显著性或等价检验。不得只选择有利指标。</p>')
        parts.append(table(['数据','seed','对照→候选']+keys,[[c['domain'],c['seed'],c['control']+' → '+c['treatment']]+[f'{c["rmse_change"][k]:+.5f}' for k in keys] for c in contrasts]))
    audit_file=HERE/'reference_audit.json'
    if audit_file.exists():
        audit_data=json.loads(audit_file.read_text());parents=audit_data['parents']
        sample=parents[0]['finite_sample']['reference_reference']
        parts.append('<h2>真实参考之间的有限样本差异</h2><p>每父状态把16条独立参考分支分成不重叠的8+8；32次重新分组。下面是父状态等权平均误差，不是验收容差。</p>')
        parts.append(table(list(sample),[[f'{np.mean([p["finite_sample"]["reference_reference"][k]["mean"] for p in parents]):.5f}' for k in sample]]))
    parts.append('<h2>判读规则</h2><ul>'+''.join('<li>'+html.escape(d)+'</li>' for d in protocol['decisions'])+'</ul>')
    parts.append('<p>各run保存完整生成轨迹NPZ、原始参考、checkpoint、配置和训练记录；dev_final128.json含逐父状态磁化/能量/时空统计及分支bootstrap，增量方差、相邻增量协方差、完整两时关联矩阵和逐点区间；matched_sample_calibration.json提供相同8条分支预算的比较；sampler_common_source.json单独检查128/256步。历史test不参与本轮调参。</p>')
    body=''.join(parts);assets=HERE/'assets';assets.mkdir(exist_ok=True)
    page='<!doctype html><meta charset="utf-8"><title>Bauer全面诊断</title><style>body{font:15px/1.6 sans-serif;margin:24px}table{display:block;overflow:auto;border-collapse:collapse}td,th{border:1px solid #abb;padding:7px}th{background:#e6edf4}</style>'+body
    (assets/'index.html').write_text(page)
    dump(assets/'summary.json',dict(runs=runs,missing=missing,pairing=pairing,contrasts=contrasts,status='DEVELOPMENT_NOT_CERTIFIED'))
    # Embed all content without local-file links; downloaded main page stays usable.
    main=ROOT/'tests/index.html'
    if main.exists():
        text=main.read_text();start='<!-- comprehensive-diagnostic-start -->';end='<!-- comprehensive-diagnostic-end -->'
        if start in text:text=text[:text.index(start)]+text[text.index(end)+len(end):]
        section=start+'<section id="bauer-comprehensive"><details open><summary>最新：全面诊断结果</summary>'+body+'</details></section>'+end
        main.write_text(text.replace('</main>',section+'</main>',1))
        snapshot=ROOT/'tests/progress_dashboard/snapshot.json'
        if snapshot.exists():
            state=json.loads(snapshot.read_text());state['sources'][str((assets/'index.html').relative_to(ROOT))]=hashlib.sha256((assets/'index.html').read_bytes()).hexdigest();dump(snapshot,state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['audit','report']);args=parser.parse_args()
    globals()[args.mode]()
