"""Reuse all nine frozen final checkpoints' saved paths; no new training."""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .metrics import patches, nearest, bias

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = ROOT/'tests/deployment_moment_repair'


def run(smoke=False):
    tick = time.perf_counter(); hashes = {}
    for path in list(HERE.glob('*.py')) + list(HERE.glob('*.sh')):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

    def read(path):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return h5py.File(path)

    banks = {radius: [[], []] for radius in (1, 2, 4)}
    sources = set()
    # Both actual extra-supervision banks; only TRAIN initial states, no futures.
    for directory in ('training_restart_repair', 'model_state_refresh_repair'):
        base = ROOT/'tests'/directory
        rows = json.loads((base/'manifest.json').read_text())
        hashes[str((base/'manifest.json').relative_to(ROOT))] = hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest()
        with read(base/'states.h5') as h:
            assert h.attrs['complete']
            for row in rows:
                assert row['split'] == row['parent']['split'] == 'train'
                sources.add(row['parent']['parent_id'])
                state = h[f"state{row['index']:03d}"][:]
                assert hashlib.sha256(state.tobytes()).hexdigest() == row['state_sha256']
                for radius in banks:
                    p, m = patches(state, radius)
                    banks[radius][0].append(p); banks[radius][1].append(m)
    banks = {r: tuple(np.concatenate(x) for x in pair) for r, pair in banks.items()}
    records = []; sites = []
    for index in range(1 if smoke else 9):
        out = PREVIOUS/'train'/f'{index:02d}'
        completion = json.loads((out/'complete.json').read_text()); task = completion['task']
        with read(out/'intermediate.h5') as h:
            assert h.attrs['complete']
            for j, (name, group) in enumerate(h.items()):
                if smoke and j >= 2:
                    break
                meta = json.loads(group.attrs['metadata']); parent = meta['parent']
                assert meta['origin'] == 'llg' and parent['split'] == 'dev'
                assert parent['parent_id'] not in sources
                g = group['spins'][:]
                with read(ROOT/f"tests/intermediate_restart_repair/reference/{meta['parent_index']:02d}/paths.h5") as ref:
                    assert ref.attrs['complete']
                    r = ref[f'{name}/fine/spins'][:]; base = ref[f'{name}/base/spins'][:]
                b = bias(g, r); numeric = bias(base, r)
                row = dict(seed=task['seed'], arm=task['repair'], state=meta['index'],
                           parent=parent['parent_id'], family=parent['family'], L=parent['L'],
                           time_ps=meta['origin_time_ps'], reference_count=len(r), generated_count=len(g))
                for key in ('mean_rmse', 'corrected_mse', 'longitudinal_rmse', 'transverse_rmse', 'estimator_variance'):
                    row[key] = float(b[key])
                    row['reference_'+key] = float(numeric[key])
                support = {}
                for radius, (p, m) in banks.items():
                    q, qm = patches(g[0, 0], radius)
                    support[radius] = nearest(q, qm, p, m)
                    row[f'patch_r{radius}_mean_distance'] = float(np.nanmean(support[radius]))
                    row[f'patch_r{radius}_unmatched'] = int(np.isnan(support[radius]).sum())
                records.append(row)
                for site in range(parent['L']):
                    sites.append(dict(**{k: row[k] for k in ('seed', 'arm', 'state', 'parent', 'family', 'L')},
                                      site=site, edge_distance=min(site, parent['L']-1-site),
                                      mean_error=b['error'][site].tolist(), site_rmse=float(b['site_rmse'][site]),
                                      **{f'patch_r{radius}_distance': float(support[radius][site]) for radius in support}))
    assets = HERE/('smoke_assets' if smoke else 'assets'); assets.mkdir(exist_ok=True)
    result = dict(records=records, sites=sites, input_sha256=hashes, seconds=time.perf_counter()-tick,
                  scientific_certification=False, training_parent_sources=len(sources), training_snapshots=96,
                  limitations='Snapshot banks describe extra supervision, not all native-window training coverage. '
                  'Axial-aligned nearest patches measure configuration similarity, not transition-law equivalence. '
                  'Sites and correlated snapshots are descriptive, not independent replicates. '
                  'Squared bias correction includes reference and generated branch sampling noise; negative is allowed. '
                  'Base/fine difference includes sampling and numerical effects, not pure solver error. '
                  'All reused DEV remains development. No coverage/physical pass threshold or causal claim.')
    (assets/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    colors = {'tilted': '#0072B2', 'interior_domain': '#D55E00', 'single_wall': '#009E73'}
    for family, color in colors.items():
        points = [x for x in sites if x['arm'] == 'control' and x['seed'] == 11 and x['family'] == family]
        axes[0].scatter([x['patch_r2_distance'] for x in points], [x['site_rmse'] for x in points],
                        label=family, color=color, s=10, alpha=.4)
    axes[0].set(xlabel='Nearest TRAIN patch RMS chordal distance (r=2)', ylabel='Local mean increment RMSE', title='Seed11 FM control: descriptive sites')
    axes[0].legend(fontsize=8)
    arms = ('control', 'moments_0p1', 'moments_0p5')
    for i, arm in enumerate(arms):
        values = [np.mean([x['corrected_mse'] for x in records if x['arm'] == arm and x['seed'] == seed]) for seed in sorted({x['seed'] for x in records if x['arm'] == arm})]
        if values:
            axes[1].bar(i, np.mean(values), color=('#0072B2', '#E69F00', '#009E73')[i], edgecolor='black')
            axes[1].scatter(i+np.linspace(-.08,.08,len(values)), values, color='black')
    axes[1].set(xticks=range(3), xticklabels=['FM','Mom .1','Mom .5'], ylabel='Sampling-corrected squared mean bias', title='Fixed states; dots = training seeds')
    axes[1].axhline(0, color='gray', lw=.8)
    for j, family in enumerate(colors):
        rows = [x for x in records if x['arm']=='control' and x['family']==family]
        if rows:
            for k, key in enumerate(('longitudinal_rmse','transverse_rmse')):
                axes[2].bar(j+(k-.5)*.35,np.mean([x[key] for x in rows]),width=.35,
                            color=('#0072B2','#E69F00')[k],edgecolor='black',label=('z','xy')[k] if j==0 else None)
    axes[2].set(xticks=range(3),xticklabels=['Tilted','Interior','Wall'],ylabel='Component RMSE',title='FM control: parent-family strata')
    axes[2].legend(); fig.savefig(assets/'local_diagnosis.png',dpi=180);plt.close(fig)
    body='<h1>局域覆盖与条件均值偏差诊断</h1><p>固定上一轮全部9组最终90144步模型的已保存路径；不重训、不生成新参考。72个独立开发LLG来源状态，每模型128分支；参考128分支。两个TRAIN增强库共96快照、24来源；不是完整native训练覆盖。</p>'
    body+='<img style="max-width:100%" src="local_diagnosis.png"><p>左：每点为格点，横轴是与训练快照中局域构型的最近距离（允许整块绕登记z轴对齐，保持开放边缘掩码）；纵轴是局域条件均值误差。格点不是统计独立重复，趋势不构成因果证明。中：扣除独立分支抽样噪声的均值偏差平方，点为训练seed；负值允许，不裁零、不设通过阈值。右：均值偏差的z方向与xy分量，区分轴向松弛与横向演化。</p>'
    body+='<p>半径1/2/4和逐状态、逐格点误差均在JSON。距离小但误差大：不能简单归因于此尺度快照覆盖；距离大且误差大：提示覆盖线索，不证明补数据一定解决。局域近邻不能保证整个构型或非局域路径相同。参考base/fine差包含独立抽样和数值变化。重复使用DEV不是最终认证。</p><a href="summary.json">全部原始诊断数据</a>'
    body+='<h2>实际结果：是否只是有限样本噪声？</h2><table border="1" cellpadding="8"><tr><th>组别</th><th>校正后均值偏差平方</th><th>均值估计抽样方差</th></tr>'
    for arm in arms:
        subset=[x for x in records if x['arm']==arm]
        if subset:
            body+=f"<tr><td>{arm}</td><td>{np.mean([x['corrected_mse'] for x in subset]):.8f}</td><td>{np.mean([x['estimator_variance'] for x in subset]):.8f}</td></tr>"
    body+='</table><p>以上是固定开发状态描述性平均，不是总体置信区间，也没有预先确认的通过容差。校正后偏差仍明显，不支持把现有误差简单归结为分支数量太少；强辅助项未改善该指标。仍不能据此确定唯一病因。</p>'
    body+='<h2>覆盖线索：按状态汇总，不将格点当独立重复</h2><ul>'
    subset=[x for x in records if x['arm']=='control']
    for radius in banks:
        if len(subset)>2:
            correlation=np.corrcoef([x[f'patch_r{radius}_mean_distance'] for x in subset],[x['mean_rmse'] for x in subset])[0,1]
            body+=f'<li>半径{radius}：状态平均最近距离与均值RMSE的描述性相关系数 {correlation:.3f}。</li>'
    body+='</ul><p>同一个状态在不同seed重复、相邻快照同源，因此不对这些相关系数附伪独立p值。该覆盖库仅含两个额外监督库，未包括全部native训练窗口；后续应先审计完整训练状态覆盖，再决定补父状态还是修改条件表示。不追加同一种矩约束权重、不自动扩大训练预算。</p>'
    (assets/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><body style="max-width:1500px;margin:30px auto;font:16px/1.6 sans-serif">'+body+'</body></html>')
    if not smoke:
        path=ROOT/'tests/index.html';txt=path.read_text();start='<!-- local-support-diagnosis:start -->';end='<!-- local-support-diagnosis:end -->'
        block=start+'<section id="local-support-diagnosis"><h2>最新：局域覆盖与条件偏差</h2><a href="local_support_diagnosis/assets/index.html">逐状态/逐格点诊断</a><img style="max-width:100%" src="local_support_diagnosis/assets/local_diagnosis.png"></section>'+end
        txt=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:block,txt,flags=re.S) if start in txt else txt.replace('</body>',block+'</body>')
        tmp=path.with_suffix('.html.tmp');tmp.write_text(txt);tmp.replace(path)
    print(json.dumps(dict(states=len(records),sites=len(sites),seconds=time.perf_counter()-tick)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true')
    run(parser.parse_args().smoke)
