"""Explicit artifact registration; never submit jobs, modify archives or push Git."""
import argparse,csv,hashlib,html,json,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'assets/research/bauer_path/progress_20261019'


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def copy_file(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if hashlib.sha256(source.read_bytes()).digest()!=hashlib.sha256(target.read_bytes()).digest():
            raise FileExistsError(f'destination differs; refusing overwrite: {target}')
    else:shutil.copy2(source,target)


def materialize():
    target=ROOT/'data/research/datasets/bauer_open_chain';native=target/'native'
    source=ROOT/'tests/bauer_active_data_20260918/data'
    if not native.exists():shutil.copytree(source,native)
    real=json.loads((ROOT/'tests/training_restart_repair/manifest.json').read_text())
    model=json.loads((ROOT/'tests/model_state_refresh_repair/manifest.json').read_text())
    for name,rows in (('real',real),('model',model)):
        manifest=[]
        package='training_restart_repair' if name=='real' else 'model_state_refresh_repair'
        for row in rows:
            sid,pid=row['index'],row['parent_index'];src=ROOT/f'tests/{package}/reference/{pid:02d}/state{sid:03d}_train.h5'
            dest=target/f'conditional/{name}/state{sid:03d}_train.h5';copy_file(src,dest)
            manifest.append(dict(file=str(dest.relative_to(target/'conditional')),parent_id=row['parent']['parent_id'],state=sid))
        dump(target/f'conditional/{name}.json',manifest)
    refill=[]
    for row in real[::2]:
        pid=row['parent_index'];dest=target/f'refill/{pid:02d}.h5'
        copy_file(ROOT/f'tests/branch_coverage_repair/reference/{pid:02d}/train.h5',dest)
        refill.append(dict(parent_id=row['parent']['parent_id'],old_file=str((native/'L16'/row['parent']['file']).relative_to(ROOT)),new_file=str(dest.relative_to(ROOT))))
    dump(target/'refill.json',refill)
    checkpoints=[]
    for i,seed in enumerate((11,22,33)):
        src=ROOT/f'tests/local_teacher_repair/train/{i*2:02d}/checkpoint.pt';dest=ROOT/f'output/development/bauer_path/step130240/seed{seed}/checkpoint.pt'
        copy_file(src,dest)
        checkpoints.append(dict(seed=seed,step=130240,file=str(dest.relative_to(ROOT)),source=str(src.relative_to(ROOT)),sha256=hashlib.sha256(src.read_bytes()).hexdigest(),scientific_certification=False))
    dump(ROOT/'output/development/bauer_path/checkpoint_index.json',checkpoints)
    dump(ROOT/'conf/research/bauer_path/data_manifest.json',dict(native=str(native.relative_to(ROOT)),conditional_states_per_origin=48,refill_sources=24,checkpoints=checkpoints,
         status='datasets stay local; frozen originals retained as test evidence; active entrypoints use data/ and output/'))
    print('registered native dataset,96 conditional TRAIN banks,24 refill banks and3 development checkpoints')


def build_report():
    campaigns=[]
    catalog=[]
    for source in sorted((ROOT/'tests').glob('*/assets/summary.json')):
        catalog.append(dict(experiment=source.parents[1].name,local_summary=str(source.relative_to(ROOT)),bytes=source.stat().st_size,
            evidence='historical diagnostic; not a certification; raw reports remain local'))
    dump(ASSETS/'experiment_catalog.json',catalog)
    for name in ('continuous_axial_repair','chain_reversal_repair','spatial_covariance_repair','auxiliary_frequency_repair','local_teacher_repair'):
        src=ROOT/f'tests/{name}/assets/summary.json';raw=json.loads(src.read_text())
        result=dict(name=name,source=str(src.relative_to(ROOT)),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest())
        for key in ('dev','full','space','cost','auxiliary'):
            if key in raw:result[key]=raw[key]
        campaigns.append(result)
    cancellation=json.loads((ROOT/'tests/on_policy_refresh_repair/cancellation.json').read_text())
    execution=json.loads((ROOT/'tests/on_policy_refresh_repair/execution.json').read_text())
    progress=dict(date='2026-10-19',status='all project computation stopped by user; no automatic resume',campaigns=campaigns,
        current_physics=dict(system='Bauer open ferromagnetic chain',temperature_K=63.824849668,theta=.11,trained_sizes=[16,25,32],learned_lag_ps=.1,composed_duration_ps=1.,reference_dt_fs=[.25,.125],flow_steps=[128,512]),
        current_coverage_diagnostic=execution['confirmed_provenance_gap'],pilot=execution['completed_checks'],cancellation=cancellation,
        scientific_certification=False,limits=['No accepted quantitative path-distribution accuracy yet','No unseen-size transfer or accuracy-matched speedup certification','Repeated DEV use; not untouched final test','Known joint-v2 interval undercoverage; no equivalence claims'])
    dump(ASSETS/'progress.json',progress)
    body='<h1>有限温磁性随机路径：当前研究进展</h1><p>2026-10-19。模型可训练和生成，但条件路径统计尚未验证通过。所有项目计算和自动续提已按用户要求停止。</p>'
    body+='<p>当前：Bauer开放链，L16/25/32均参与训练，63.824849668K，学习0.1ps转移并接续1ps。LLG步长0.25/0.125fs、输出0.1ps与flow128/512分别登记。单位球面/首帧/适用对称性的软件通过不代表物理准确性。</p>'
    columns={'dev':['seed','arm','meanRMSE','covRMSE','signed variance bias','H/(NJ) meanRMSE'],
             'full':['seed','arm','L','sampler','increment varianceRMSE','two-timeRMSE','H/(NJ) meanRMSE'],
             'cost':['seed','arm','parameters','train seconds','evaluation seconds']}
    for campaign in campaigns:
        body+='<h2>'+html.escape(campaign['name'])+'</h2>'
        for key,headers in columns.items():
            rows=campaign.get(key,[])
            if not rows:continue
            dest=ASSETS/f'{campaign["name"]}_{key}.csv';dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('w',newline='') as stream:
                writer=csv.writer(stream,lineterminator='\n');writer.writerow(headers);writer.writerows(rows)
            body+='<h3>'+key+'</h3><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'
            for row in rows:body+='<tr>'+''.join('<td>'+html.escape(f'{x:.7g}' if isinstance(x,float) else str(x))+'</td>' for x in row)+'</tr>'
            body+='</tbody></table><a href="'+dest.name+'">CSV</a>'
    body+='<h2>最新未完成诊断</h2><p>旧模型状态库来自seed11 step78144，当前模型已到step130240。已提取144个TRAIN派生状态，完成一个来源的1920条0.1ps真实LLG先导。43项软件回归和先导数据核验通过；GPU预检/正式刷新对照尚未完成，任务已经取消，不能写成修复成功。</p>'
    body+='<p>误差越小通常越接近参考，但有限样本误差仍在，不用单一综合分数或事后阈值宣布通过。跨状态均值为描述，格点/帧不是独立重复。区间含零不证明等价。</p><p><a href="progress.json">紧凑原始统计与来源校验</a> · <a href="experiment_catalog.json">历史诊断索引（大报告仅本地）</a> · <a href="../../../../GUIDE/CURRENT_PROGRESS_20261019.md">研究进展文档</a></p>'
    style='body{max-width:1400px;margin:28px auto;padding:16px;font:16px/1.7 sans-serif;color:#182438}table{border-collapse:collapse;font-size:14px;margin:12px 0}th,td{border:1px solid #ccd6e0;padding:5px 9px;text-align:right}th{background:#e9f0f7}h1,h2{color:#173f66}a{color:#155ca0}'
    (ASSETS/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Research progress</title><style>'+style+'</style><body>'+body+'</body></html>')
    print(ASSETS/'index.html')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--materialize',action='store_true');a=p.parse_args()
    if a.materialize:materialize()
    build_report()
