"""Independent-future statistics; descriptive, never automatic certification."""
import argparse,json,html
import h5py
import numpy as np
from tests.transition_block_repair.metrics import compare_first
from tests.conditional_moment_repair.report import metrics
from tests.training_restart_repair.data import rows,HERE as DATA
from .run import HERE,ROOT,tasks,source,dump,verify

def analyze(index,smoke):
    folder=HERE/('smoke' if smoke else 'train')/f'{index:02d}'
    run=json.loads((folder/'complete.json').read_text());assert run['task']==tasks()[index]
    middle=[];seen=[]
    with h5py.File(folder/'intermediate.h5') as h:
        assert h.attrs['complete'] and len(h)==(1 if smoke else 72)
        for name,g in h.items():
            meta=json.loads(g.attrs['metadata']);x=g['spins'][:]
            with h5py.File(ROOT/f"tests/intermediate_restart_repair/reference/{meta['parent_index']:02d}/paths.h5") as ref:
                assert ref.attrs['complete']
                result={k:compare_first(x,ref[f'{name}/{k}/spins'][:]) for k in ('base','fine')}
            middle.append(dict(metadata=meta,metrics=result))
    for phase,path in (('before',source(run['task']['seed'])/'after_restart.h5'),('after',folder/'after_restart.h5')):
        with h5py.File(path) as h:
            assert h.attrs['complete']
            for row in rows(smoke):
                name=f"state{row['index']:03d}";x=h[f'{name}/spins'][:];refs={}
                assert json.loads(h[name].attrs['metadata'])==row
                for role in ('probe','fine'):
                    with h5py.File(DATA/'reference'/f"{row['parent_index']:02d}"/f'{name}_{role}.h5') as ref:
                        assert ref.attrs['complete'] and ref.attrs['role']==role
                        refs[role]=ref['spins'][:]
                seen.append(dict(phase=phase,metadata=row,metrics={k:compare_first(x,v) for k,v in refs.items()}))
    dump(folder/'report.json',dict(intermediate=middle,restart_seen=seen,scientific_certification=False))

def aggregate(smoke):
    results=[];runs=[]
    for i in range(4 if smoke else 12):
        folder=HERE/('smoke' if smoke else 'train')/f'{i:02d}'
        run=json.loads((folder/'complete.json').read_text());runs.append(run)
        response=json.loads((folder/'response.json').read_text())
        report=json.loads((folder/'report.json').read_text())
        total=16 if smoke else 512
        counts=np.array(json.loads((folder/'exposures.json').read_text()))
        assert counts.shape==(48,64) and (counts.sum(1)==total).all()
        reset=run['task']['repair'].startswith('reset')
        assert response['optimizer_steps_before']==([] if reset else [54000])
        assert response['optimizer_steps_after']==[total if reset else 54000+total]
        risk=lambda phase:float(np.mean([x['conditional_risk'] for x in response[phase]]))
        values=np.mean([metrics(x['metrics']['fine']) for x in report['intermediate']],axis=0).tolist()
        results.append(dict(index=i,task=run['task'],risk_before=risk('before'),risk_after=risk('after'),
            relative_parameter_change=response['relative_parameter_change'],dev_metrics=values,
            seen_before=np.mean([metrics(x['metrics']['fine']) for x in report['restart_seen'] if x['phase']=='before'],axis=0).tolist(),
            seen_after=np.mean([metrics(x['metrics']['fine']) for x in report['restart_seen'] if x['phase']=='after'],axis=0).tolist()))
    for seed in ((11,) if smoke else (11,22,33)):
        group=[r for r in runs if r['task']['seed']==seed]
        assert len(group)==4
        for key in ('paired_input_hash','extra_noise_hash','group_schedule_hash','schedule_hash'):
            assert len({r[key] for r in group})==1,(seed,key)
    out=HERE/('smoke_assets' if smoke else 'assets');out.mkdir(exist_ok=True)
    dump(out/'summary.json',dict(results=results,runs=runs,paired_contract_passed=True,
        metric_columns=['increment_mean_RMSE','increment_covariance_RMSE','signed_variance_bias','energy_mean_RMSE_H_per_NJ'],
        scientific_certification=False))
    body='<h1>优化器响应：512步受控诊断</h1><p>四组 × 三个种子。训练风险不是物理准确性；以下为重复使用的开发集，不是最终测试。仅验证0.1 ps独立未来，本轮不认证1 ps滚动。</p>'
    body+='<table border="1"><tr><th>seed / arm</th><th>TRAIN risk 前 → 后</th><th>参数相对变化</th><th>DEV 增量均值RMSE</th><th>协方差RMSE</th><th>有符号方差偏差</th><th>能量均值RMSE (H/NJ)</th></tr>'
    for r in results:
        body+='<tr><td>'+html.escape(str(r['task']['seed'])+' / '+r['task']['repair'])+'</td><td>'+f"{r['risk_before']:.6g} → {r['risk_after']:.6g}"+'</td><td>'+f"{r['relative_parameter_change']:.6g}"+'</td>'+''.join(f'<td>{v:.6g}</td>' for v in r['dev_metrics'])+'</tr>'
    body+='</table><p>均值、协方差、能量RMSE越小越好；有符号方差偏差应接近零。均为条件状态的描述性平均，不能当作独立样本误差条；逐状态bootstrap及base/fine结果见每次运行report.json。</p><p><a href="summary.json">全部汇总与运行记录</a></p>'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Optimizer response</title>'+body)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['analyze','aggregate']);p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    verify()
    analyze(a.index,a.smoke) if a.action=='analyze' else aggregate(a.smoke)
