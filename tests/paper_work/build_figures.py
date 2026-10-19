"""Read-only evidence assembly for the Bauer paper; no training or simulation.

Run from project root in zrs-mag. Outputs stay in tests/paper_work.
All ensemble summaries are conditional on a named initial state. No frames or
sites are counted as independent paths; cross-state averages are descriptive.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import base64
import html
import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'tests/paper_work'
ARMS = ['reference0', 'reference1', 'baseline', 'flow11', 'flow22', 'flow33']
LABELS = ['MT reference', 'Heun control', 'Gaussian baseline', 'Flow seed 11', 'Flow seed 22', 'Flow seed 33']
COLORS = ['#000000', '#777777', '#009E73', '#0072B2', '#D55E00', '#882255']
STYLES = ['-', '--', '-.', '-', '--', ':']
DISPLAY_ARMS = ['reference0', 'flow22']
SOURCES = {}
FIGURES = []


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def register(path, expected=None):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    key = str(path.relative_to(ROOT))
    if key not in SOURCES:
        SOURCES[key] = dict(sha256=sha(path), bytes=path.stat().st_size)
    if expected is not None and SOURCES[key]['sha256'] != expected:
        raise ValueError(f'Changed source: {key}')
    return path


def read_json(path):
    return json.loads(register(path).read_text())


def table(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / 'tables' / (name + '.csv')).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def savefig(fig, name, caption):
    fig.savefig(OUT / 'figures' / (name + '.png'), dpi=180, facecolor='white')
    plt.close(fig)
    FIGURES.append(dict(name=name, caption=caption))


def panel(ax, title, xlabel, ylabel):
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.spines[['top', 'right']].set_visible(False)


def note(ax, title, message):
    ax.axis('off')
    ax.set_title(title, loc='left')
    ax.text(.02, .85, message, transform=ax.transAxes, va='top', fontsize=11,
            bbox=dict(facecolor='#eeeeee', edgecolor='#aaaaaa', pad=12))


def log_increment(p):
    x, y = p[:, 0], p[:, 1]
    dot = np.clip((x*y).sum(-1, keepdims=True), -1, 1)
    v = y-dot*x
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    return v * np.divide(np.arctan2(length, dot), length,
                         out=np.ones_like(length), where=length > 1e-12)


def observables(p):
    mz = p[..., 2].mean(-1)
    bond = (p[:, :, :-1] * p[:, :, 1:]).sum(-1)
    return dict(mz=mz, bond=bond, spatial_std=p[..., 2].std(-1),
                mean_spin=p.mean(0), variance=p.var(0).sum(-1).mean(-1),
                sign_bonds=(p[:, :, :-1, 2]*p[:, :, 1:, 2] < 0).sum(-1),
                endpoint=p[:, -1], increment=log_increment(p),
                examples=p[:1, ::8, :, 2],
                energy=-bond.sum(-1)/p.shape[2]-.1*(p[..., 2]**2).mean(-1))


def local_features(x):
    """Finite local descriptors, not a sufficient statistic for the path law."""
    left = np.zeros_like(x); right = np.zeros_like(x)
    left[..., 1:, :] = x[..., :-1, :]
    right[..., :-1, :] = x[..., 1:, :]
    return np.stack([x[..., 2], (x*left).sum(-1), (x*right).sum(-1),
                     left[..., 2], right[..., 2],
                     np.cross(x, left+right)[..., 2]], -1)


def probability_interval(k, n):
    z = 1.959963984540054
    p = np.asarray(k)/n; den = 1+z*z/n
    mid = (p+z*z/(2*n))/den
    r = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return np.maximum(0, mid-r), np.minimum(1, mid+r)


def grouped_variance(p):
    """Initial-state x branch x time x site x component, never pool initials."""
    if p.ndim != 5 or p.shape[-1] != 3:
        raise ValueError('Expected initial, branch, time, site, component')
    np.testing.assert_allclose(p[:, :, 0], np.repeat(p[:, :1, 0], p.shape[1], axis=1), atol=1e-6)
    return p.astype(float).var(1).sum(-1).mean(-1)


def arrival(mz, t, a, b, dwell):
    """Competing core confirmation, including time-zero membership."""
    from tests.experiments.committor_pilot_analysis import classify
    return classify(mz, t, a, b, dwell)


def training_data():
    from tests.experiments.coverage_transition_dataset import CoverageTransitionDataset, replacement_index
    from tests.experiments.fine_transition_dataset import FineTransitionDataset
    train = CoverageTransitionDataset(); dev = FineTransitionDataset('development')
    inputs = {}; indices = []; noise = {}
    for split, dataset in [('train', train), ('development', dev)]:
        x = []; seeds = []
        for i in range(len(dataset)):
            r = dataset[i]; x.append(r['initial']); seeds.append(r['noise_seed'])
            if split == 'train' and replacement_index(i) is not None:
                anchor, path, window = replacement_index(i)
                source = train.rows[anchor]['path']; parent = 'coverage_initial_'+str(train.rows[anchor]['state']['initial_id'])
            else:
                ds = train.original if split == 'train' else dev
                group, offset = divmod(i, 256*8)
                path, window = divmod(offset, 8)
                source = ds.rows[group]['path']; parent = 'pilot_parent_'+str(r['parent'])
            indices.append(dict(split=split, sample=i, source=source, path_index=path,
                                source_family=parent, noise_seed=r['noise_seed'], start_frame=r['start'],
                                end_frame=r['start']+4, theta=r['theta'], lag=r['lag']))
        inputs[split] = np.asarray(x); noise[split] = np.asarray(seeds)
    for source in sorted({r['source'] for r in indices}):
        register(source)
    assert set(noise['train']).isdisjoint(noise['development'])
    assert set(train.original.parents).isdisjoint(dev.parents)
    table('T01_transition_index', indices)
    rows = []
    for split in inputs:
        chosen = [r for r in indices if r['split'] == split]
        rows.append(dict(split=split, pairs=len(chosen), independent_noise_paths=len(set(noise[split])),
                         source_files=len({r['source'] for r in chosen}),
                         families=len({r['source_family'] for r in chosen}), sites=25, lag=.01, source_duration=.08,
                         role='optimizer input' if split == 'train' else 'development; reused for design, not final test'))
    table('T01_dataset_inventory', rows)
    (OUT/'source_data/split_audit.json').write_text(json.dumps(dict(
        train_pilot_parents=train.original.parents, development_pilot_parents=dev.parents,
        overlapping_noise_seeds=0, inventory=rows,
        caveat='Family identities are source-qualified. Coverage initial families and pilot parents have different provenance; no claim of population-wide independence.'), indent=2))
    return inputs, noise, rows


def current_branches():
    manifest = read_json('tests/results/committor_pilot/manifest.json')
    for model in manifest['frozen_models']:
        register(model['checkpoint'], model['sha256'])
    register(manifest['baseline'],manifest['baseline_sha256'])
    base = ROOT/'tests/results/committor_pilot'
    mh = sha(base/'manifest.json')
    records = {}; timing = []; audit = []
    for state in range(12):
        for arm in ARMS:
            r = read_json(base/f'{arm}_state{state}.json')
            assert r['status'] == 'COMPLETE_NUMERICAL_ONLY' and r['manifest_sha256'] == mh
            with np.load(register(r['path'], r['sha256'])) as z:
                p = z['spins'].astype(float); t = z['reduced_time']
            assert p.shape == (128, 801, 25, 3) and np.isfinite(p).all()
            assert np.max(abs(np.linalg.norm(p, axis=-1)-1)) < 1e-5
            np.testing.assert_allclose(t, np.arange(801)*.01, atol=1e-12)
            if arm == 'reference0':
                initial = p[:, 0].copy()
            else:
                np.testing.assert_allclose(p[:, 0], initial, atol=2e-7, rtol=0)
            ob = observables(p)
            ob['coverage_states'] = p[:, [1, 8, 80, 800]]
            ob['spatial_correlation'] = np.array([(p[:, -1, :25-r]*p[:, -1, r:]).sum(-1).mean() for r in range(13)])
            # Fixed-time connected covariance; no stationarity assumption.
            centered = p-p.mean(0, keepdims=True)
            ob['temporal_connected'] = (centered[:, 80:]*centered[:, 80:81]).sum(-1).mean((0, 2))
            records[state, arm] = ob
            timing.append(dict(state=state, arm=arm, paths=len(p), sites=25, duration=8,
                               seconds_including_save=r['seconds_including_save'],
                               seconds_per_path=r['seconds_including_save']/len(p), job_id=r['job_id'],
                               scope='historical per-arm elapsed incl save; different hardware; not speedup'))
            audit.append(dict(state=state, arm=arm, source=r['path'], paths=len(p), frames=len(t),
                              max_norm_error=float(np.max(abs(np.linalg.norm(p, axis=-1)-1))), initial_match=True))
        print(f'Validated branch state {state+1}/12', flush=True)
    table('T02_branch_audit', audit); table('T06_runtime', timing)
    return manifest, records, np.arange(801)*.01, timing


def coverage(inputs, seeds, records, manifest):
    rng = np.random.default_rng(20260914)
    # Split all windows of one noise path together for descriptor calibration.
    unique = np.unique(seeds['train']); rng.shuffle(unique)
    base_mask = np.isin(seeds['train'], unique[:len(unique)//2])
    f = local_features(inputs['train'][base_mask]).reshape(-1, 6)
    mean = f.mean(0); sd = np.maximum(f.std(0), 1e-8)
    pool_ids = rng.choice(len(f), min(30000, len(f)), replace=False)
    tree = cKDTree((f[pool_ids]-mean)/sd)
    def distance(x):
        return tree.query((local_features(x).reshape(-1, 6)-mean)/sd, workers=1)[0]
    calibration = distance(inputs['train'][~base_mask])
    threshold = float(np.quantile(calibration, .99))
    rows = []; series = [('train held-out paths', calibration)]
    series.append(('development', distance(inputs['development'])))
    for j, label in enumerate(['reference t=.01', 'reference t=.08', 'reference t=.8', 'reference t=8']):
        x = np.concatenate([records[s, 'reference0']['coverage_states'][:, j] for s in range(12)])
        series.append((label, distance(x)))
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for label, x in inputs.items():
        mz = x[..., 2].mean(-1); b = (x[:, :-1]*x[:, 1:]).sum(-1).mean(-1)
        ax[0, 0].scatter(mz[::8], b[::8], s=6, alpha=.35, label=label, marker='o' if label=='train' else 'x')
        ax[0, 1].hist(x[..., 2].ravel(), bins=np.linspace(-1, 1, 41), density=True, histtype='step', label=label)
        table('S01_'+label+'_state_descriptors', [dict(sample=i, mz=mz[i], bond=b[i]) for i in range(len(x))])
    panel(ax[0,0], 'A  Sampled state coverage (one window/path shown)', 'Mz', 'Mean nearest-neighbor dot product')
    panel(ax[0,1], 'B  Local orientation coverage (correlated sites/windows)', 'Local sz', 'Descriptive density')
    for label, values in series:
        row=dict(group=label, descriptor_points=len(values), median_distance=float(np.median(values)),
                 p99_train_distance=threshold, fraction_above_train_p99=float(np.mean(values>threshold)))
        rows.append(row)
        order=np.sort(values); keep=np.linspace(0, len(order)-1, 300).astype(int)
        ax[1,0].plot(order[keep], (keep+1)/len(order), label=label)
    ax[1,0].axvline(threshold,color='black',ls='--')
    panel(ax[1,0], 'C  Six-descriptor nearest-neighbor distance', 'Train-standardized distance', 'Empirical CDF')
    ax[1,1].barh([r['group'] for r in rows], [r['fraction_above_train_p99'] for r in rows], color='#0072B2')
    panel(ax[1,1], 'D  Outside train-calibrated descriptor neighborhood', 'Fraction (heuristic, not certification)', '')
    for a in ax.flat:
        if a is not ax[1,1]: a.legend(fontsize=7)
    savefig(fig, 'S01_state_coverage', '状态覆盖：A/B 为描述性投影；C/D 为六维局部特征近邻距离。按噪声路径划分校准，训练池最多30000格点，99%阈值仅作外推提示，不是充分覆盖认证。')
    table('T03_coverage', rows)
    np.savez_compressed(OUT/'source_data/coverage_normalization.npz',mean=mean,std=sd,pool_indices=pool_ids,threshold=threshold)
    return rows


def metric_rows(records, manifest, times):
    rows = []; local = []
    for state in range(12):
        ref=records[state,'reference0']; d=ref['increment']; rc=np.cov(d.reshape(128,-1),rowvar=False)
        for arm in ARMS:
            ob=records[state,arm]; dc=ob['increment']; cov=np.cov(dc.reshape(128,-1),rowvar=False)
            local.append(dict(state=state,arm=arm,theta=manifest['rows'][state]['theta'],kind=manifest['rows'][state]['kind'],
                drift_rmse=float(np.sqrt(np.mean((dc.mean(0)-d.mean(0))**2))/.01),
                tangent_variance_ratio=float(np.trace(cov)/np.trace(rc)),
                covariance_relative_error=float(np.linalg.norm(cov-rc)/np.linalg.norm(rc)),
                mean_site_increment_W1=float(np.mean([wasserstein_distance(dc[:,i,j],d[:,i,j]) for i in range(25) for j in range(3)]))))
            for horizon in [.01,.08,.8,2.,8.]:
                i=round(horizon/.01)
                rows.append(dict(state=state,arm=arm,theta=manifest['rows'][state]['theta'],kind=manifest['rows'][state]['kind'],horizon=horizon,
                    mean_rmse=float(np.sqrt(np.mean((ob['mean_spin'][i]-ref['mean_spin'][i])**2))),
                    bond_rmse=float(np.sqrt(np.mean((ob['bond'][:,i].mean(0)-ref['bond'][:,i].mean(0))**2))),
                    variance_ratio=float(ob['variance'][i]/ref['variance'][i]),
                    mz_W1=float(wasserstein_distance(ob['mz'][:,i],ref['mz'][:,i]))))
    table('T04_local_transition',local);table('T04_rollout_metrics',rows)
    return rows,local


def make_main(records, manifest, times, inventory, metrics, local, timing):
    state=3  # First wall state at theta .11, not selected by agreement.
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,r in enumerate(inventory):
        ax[0,0].bar(i,r['pairs'],color=['#0072B2','#D55E00'][i])
        ax[0,0].text(i,r['pairs'],f"{r['pairs']} pairs\n{r['independent_noise_paths']} noise paths",ha='center',va='bottom',fontsize=9)
    ax[0,0].set_xticks([0,1],['Training','Development']);ax[0,0].set_ylim(0,16000)
    panel(ax[0,0],'A  Current transition data','Split','Transition pairs')
    for i,s in enumerate(manifest['rows']):
        ax[0,1].scatter(s['theta'],i,marker={'aligned':'o','tilted':'s','wall':'^'}[s['kind']],color='#0072B2')
    panel(ax[0,1],'B  Twelve fixed branch initial states','Reduced temperature theta','State ID (0-11)')
    ax[0,1].set_xticks([.11,.13])
    horizons=[.01,.08,.8,8,200,750000]
    for x,label in zip(horizons,['Learned lag','Raw training','Old rollout','Current rollout','Reference calibration','Literature target']):
        ax[1,0].scatter(x,25 if x<750000 else 100,marker='o' if x<=8 else 's')
        ax[1,0].annotate(label,(x,25 if x<750000 else 100),rotation=25,xytext=(0,8),textcoords='offset points',fontsize=8)
    ax[1,0].set_xscale('log');ax[1,0].set_ylim(0,135)
    panel(ax[1,0],'C  Evidence range; markers are not pass certificates','Reduced time (log scale)','Sites L')
    for arm,i in [('reference0',0),('flow22',4)]:
        for k in range(4):ax[1,1].plot(times,records[state,arm]['mz'][k],color=COLORS[i],ls=STYLES[i],alpha=.5,label=LABELS[i] if k==0 else None)
    panel(ax[1,1],'D  Same initial state; four independent futures','Reduced time','Mz');ax[1,1].legend()
    savefig(fig,'Fig01_data_and_scope','数据与范围：当前L25、两温度；训练原始窗口0.08、学习间隔0.01、最新滚动到8。L100长时数据是历史真实参考，不是当前生成模型推广结果。D固定展示state3前四条轨迹，不按结果挑选。')

    fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for i,arm in enumerate(ARMS):
        if arm not in DISPLAY_ARMS:continue
        ob=records[state,arm];q=np.quantile(ob['mz'],[.1,.5,.9],axis=0)
        ax[0,0].plot(times,q[1],color=COLORS[i],ls=STYLES[i],label=LABELS[i])
        ax[0,0].fill_between(times,q[0],q[2],color=COLORS[i],alpha=.12)
        ax[0,1].hist(ob['mz'][:,-1],bins=np.linspace(-1,1,41),histtype='step',density=True,color=COLORS[i],ls=STYLES[i])
        ax[0,2].plot(times,ob['bond'].mean((0,2)),color=COLORS[i],ls=STYLES[i])
        ax[1,0].plot(np.arange(13),ob['spatial_correlation'],color=COLORS[i],ls=STYLES[i])
        ax[1,1].plot(times[80:]-.8,ob['temporal_connected'],color=COLORS[i],ls=STYLES[i])
        ax[1,2].plot(times,ob['spatial_std'].mean(0),color=COLORS[i],ls=STYLES[i])
        table('Fig02_'+arm,[dict(time=t,median_mz=q[1,k],q10_mz=q[0,k],q90_mz=q[2,k],mean_bond=ob['bond'][:,k].mean(),mean_spatial_std=ob['spatial_std'][:,k].mean()) for k,t in enumerate(times)])
    for a,title,x,y in zip(ax.flat,['A  Mz median; shading = path 10-90%','B  Endpoint Mz distribution','C  Nearest-neighbor alignment','D  Equal-time correlation at t=8','E  Connected covariance from t0=.8','F  Within-chain spatial variation'],['Reduced time','Mz at t=8','Reduced time','Site separation r','Time lag from .8','Reduced time'],['Mz','Density','Mean bond dot product','Mean spin dot product','Connected spin dot product','Mean spatial std(sz)']):panel(a,title,x,y)
    ax[0,0].legend(fontsize=8)
    fig.suptitle('Fig 2 | Selected flow seed 22 vs MT: state 3, theta=.11, 128 paths/arm')
    savefig(fig,'Fig02_conditional_paths','主图仅展示选中的flow22（橙色）与真实MT（黑色）。区间是轨迹分布10–90%而非置信区间；state3为固定壁起点示例，全12状态和其他模型仍保留在补充图表。局部/短时较接近，但存在滚动误差，不能认证完整路径。')

    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    hs=[.01,.08,.8,2.,8.]
    for i,arm in enumerate(ARMS[1:],1):
        if arm not in ['reference1','flow22']:continue
        for a,key in [(ax[0,0],'mean_rmse'),(ax[0,1],'bond_rmse')]:
            means=[np.mean([r[key] for r in metrics if r['arm']==arm and r['horizon']==h]) for h in hs]
            a.plot(hs,means,color=COLORS[i],ls=STYLES[i],marker='o',label=LABELS[i])
    for a,title,y in [(ax[0,0],'A  Mean-field error accumulation','Mean-spin RMSE'),(ax[0,1],'B  Spatial error accumulation','Bond-profile RMSE')]:
        a.set_xscale('log');panel(a,title,'Reduced time (log scale)',y);a.legend(fontsize=7)
    mat=np.array([[next(r['mean_rmse'] for r in metrics if r['arm']=='flow22' and r['state']==s and r['horizon']==h) for s in range(12)] for h in hs])
    im=ax[1,0].imshow(mat,aspect='auto',cmap='viridis');fig.colorbar(im,ax=ax[1,0],label='Flow22 mean-spin RMSE vs MT')
    ax[1,0].set_yticks(range(5),[str(h) for h in hs]);panel(ax[1,0],'C  Selected seed 22: all initial states','State ID','Reduced time')
    note(ax[1,1],'D  Size-generalization evidence gap','Current frozen model: L=25 only.\nTime rollout: .08 -> 8 (100x window).\nLonger generation is not proof of accuracy.\n\nNo matched frozen-model L100 dataset.\nNo size-error curve can be claimed.\nHistorical models are not pooled here.')
    savefig(fig,'Fig03_time_transfer_and_size_gap','主图展示flow22，Heun仅为独立数值参考对照。A/B为12个固定起点等权描述性平均，不是总体置信区间；C保留flow22全部起点和时间。到8其均值场RMSE约0.0420，Heun对照约0.0267；尺寸推广仍缺。')

    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,arm in enumerate(ARMS[1:],1):
        selected=[r for r in local if r['arm']==arm]
        ax[0,0].plot(range(12),[r['drift_rmse'] for r in selected],marker='o',color=COLORS[i],ls=STYLES[i],label=LABELS[i])
        ax[0,1].plot(range(12),[r['tangent_variance_ratio'] for r in selected],marker='o',color=COLORS[i],ls=STYLES[i])
        ax[1,0].plot(range(12),[r['covariance_relative_error'] for r in selected],marker='o',color=COLORS[i],ls=STYLES[i])
        ax[1,1].plot(range(12),[r['mean_site_increment_W1'] for r in selected],marker='o',color=COLORS[i],ls=STYLES[i])
    ax[0,1].axhline(1,color='black',ls=':')
    for a,title,y in zip(ax.flat,['A  Finite-lag mean increment / lag','B  Tangent fluctuation magnitude','C  Full spin-increment covariance','D  Marginal increment distributions'],['Drift RMSE vs MT','Variance trace ratio vs MT','Relative Frobenius error','Mean marginal W1']):panel(a,title,'State ID',y)
    ax[0,0].legend(fontsize=8)
    savefig(fig,'S02_local_stochastic_evolution','局部随机演化：球面log增量在固定0.01间隔的均值、协方差和边缘分布。不是Δt→0的瞬时生成元；128分支下协方差误差有明显抽样底噪，Heun是数值与独立噪声对照。')


def events_and_literature(records, manifest, t):
    base=ROOT/'tests/results/committor_pilot';cores={};tails={};calrows=[]
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ti,theta in enumerate([.11,.13]):
        for case in range(2):
            r=read_json(base/f'calibration_{ti*2+case}.json')
            with np.load(register(r['path'],r['sha256'])) as z:p=z['spins'];ct=z['reduced_time']
            mz=p[...,2].mean(-1);tail=mz[:,ct>=150].ravel();tails[theta,case]=tail
            if case==1:
                pos=tail[tail>0];neg=tail[tail<0]
                assert min(len(pos),len(neg))>1
                cores[theta]={q:(float(np.quantile(neg,1-q)),float(np.quantile(pos,q))) for q in [.05,.1,.2]}
            ax[ti,0].hist(tail,bins=np.linspace(-1,1,51),density=True,histtype='step',label=f'MT dt={.0025 if case==0 else .00125}')
            ax[ti,1].plot(ct,abs(mz).mean(0),ls='-' if case==0 else '--',label=f'case {case}')
            delta=abs(mz[:,(ct>=150)&(ct<200)]).mean(1)-abs(mz[:,(ct>=100)&(ct<150)]).mean(1)
            calrows.extend(dict(theta=theta,case=case,path=i,late_abs_m_change=v) for i,v in enumerate(delta))
        panel(ax[ti,0],f'theta={theta}: tail 150-200','Mz','Density (correlated frames)')
        panel(ax[ti,1],f'theta={theta}: reference relaxation','Reduced time','Mean |Mz|')
        ax[ti,0].legend();ax[ti,1].legend()
    savefig(fig,'S03_reference_basin_calibration','经验磁盆校准：展示全部32条/组的晚时分布和平均|Mz|。分位数可用于探索性首达统计，但不能据此宣称稳态、双稳态或承诺时间已认证。')
    table('T05_calibration_path_changes',calrows)
    rows=[];curves={};sensitivity=[]
    for state,s in enumerate(manifest['rows']):
        for arm in ARMS:
            mz=records[state,arm]['mz']
            for q,(a,b) in cores[s['theta']].items():
                for dwell in [0.,.1,.5]:
                    labels,tau=arrival(mz,t,a,b,dwell)
                    for horizon in [.8,2.,8.]:
                        nb=int(((labels==-1)&(tau<=horizon)).sum());na=int(((labels==1)&(tau<=horizon)).sum());n=len(mz)
                        lo,hi=probability_interval(nb,n);alo,ahi=probability_interval(na,n)
                        r=dict(state=state,arm=arm,theta=s['theta'],kind=s['kind'],quantile=q,dwell=dwell,horizon=horizon,
                               A_min=b,B_max=a,n=n,B=nb,A=na,unresolved=n-na-nb,p_B=nb/n,p_A=na/n,p_unresolved=1-(nb+na)/n,
                               p_B_lo=float(lo),p_B_hi=float(hi),p_A_lo=float(alo),p_A_hi=float(ahi),starts_inside_core=bool(mz[0,0]<=a or mz[0,0]>=b))
                        sensitivity.append(r)
                        if q==.1 and dwell==0:rows.append(r)
                    if q==.1 and dwell==0:
                        curves[state,arm]=dict(pB=((labels[:,None]==-1)&(tau[:,None]<=t)).mean(0),
                            pA=((labels[:,None]==1)&(tau[:,None]<=t)).mean(0),unresolved=(tau[:,None]>t).mean(0))
    table('T05_arrival_probabilities',rows);table('T05_event_sensitivity',sensitivity)
    table('T05_core_thresholds',[dict(theta=th,quantile=q,A_min=b,B_max=a) for th,qs in cores.items() for q,(a,b) in qs.items()])
    state=3
    fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for i,arm in enumerate(DISPLAY_ARMS):
        im=ax[0,i].imshow(records[state,arm]['examples'][0].T,origin='lower',aspect='auto',extent=[0,8,0,25],cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest')
        panel(ax[0,i],['A  MT: first path','B  Flow22: first path'][i],'Reduced time','Site')
    fig.colorbar(im,ax=list(ax[0,:2]),label='sz (same scale)')
    for arm in DISPLAY_ARMS:
        i=ARMS.index(arm)
        ax[0,2].plot(np.arange(25),records[state,arm]['mean_spin'][-1,:,2],color=COLORS[i],ls=STYLES[i],label=LABELS[i])
    panel(ax[0,2],'C  Ensemble mean profile at t=8','Site','Mean sz');ax[0,2].legend(fontsize=8)
    for i,arm in enumerate(ARMS):
        if arm not in DISPLAY_ARMS:continue
        ax[1,0].plot(t,curves[1,arm]['pA'],color=COLORS[i],ls=STYLES[i],label=LABELS[i])
        ax[1,1].plot(t,curves[1,arm]['unresolved'],color=COLORS[i],ls=STYLES[i])
        if arm=='reference0':continue
        selected=[r for r in rows if r['arm']==arm and r['horizon']==8 and not r['starts_inside_core']]
        for j,r in enumerate(selected):
            ref=next(x for x in rows if x['state']==r['state'] and x['arm']=='reference0' and x['horizon']==8)
            ax[1,2].errorbar(ref['p_A'],r['p_A'],xerr=[[ref['p_A']-ref['p_A_lo']],[ref['p_A_hi']-ref['p_A']]],yerr=[[r['p_A']-r['p_A_lo']],[r['p_A_hi']-r['p_A']]],fmt=['o','s','^','v','D'][i-1],color=COLORS[i],alpha=.7,label=LABELS[i] if j==0 else None)
    ax[1,2].plot([0,1],[0,1],'k--');ax[1,2].set_xlim(-.04,1.04);ax[1,2].set_ylim(-.04,1.04)
    panel(ax[1,0],'D  State 1: A-first incidence (all B counts=0)','Reduced time','Probability')
    panel(ax[1,1],'E  State 1: no A/B core reached yet','Reduced time','Unresolved fraction')
    panel(ax[1,2],'F  Non-core states; pointwise 95% Wilson','MT P(A first by 8)','Candidate P(A first by 8)')
    ax[1,0].legend(fontsize=7);ax[1,2].legend(fontsize=7)
    savefig(fig,'Fig04_mechanisms_and_arrival','主图仅比较MT和flow22。A/B为state3各自首条轨迹，C为其终态平均空间剖面；D/E为state1的A首达与未决；F保留全部非核心起点。各方法B首达均为0，不是翻转概率认证。经验核心q=.1、保持时间0；其他方法/阈值仍保留表中。')
    table('Fig04_event_curves',[dict(state=s,arm=arm,time=float(ti),p_A=float(curves[s,arm]['pA'][k]),p_B=float(curves[s,arm]['pB'][k]),unresolved=float(curves[s,arm]['unresolved'][k])) for s in [1,3] for arm in ARMS for k,ti in enumerate(t)])
    # Real historical L100 reference, explicitly separate from learned model.
    old='assets/literature_reproduction/bauer_2011/experiments/reference_dynamics/20260913_classical_independent_original_dt002/analysis/'
    lit=read_json(old+'analysis.json');status=read_json(old+'result.json')
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    duration=lit['case']['duration'];ft=np.asarray(lit['first_pass_times']);observed=np.asarray(lit['first_pass_observed']).astype(bool)
    grid=np.r_[0,np.sort(np.unique(ft[observed])),duration]
    survival=np.array([1-np.mean(observed&(ft<=v)) for v in grid])
    assert np.all(ft[~observed]==duration), 'This display assumes common administrative censoring'
    ax[0,0].step(grid,survival,where='post',color='black');ax[0,0].plot(duration,survival[-1],marker='+',color='black',ms=12)
    panel(ax[0,0],'A  Historical L100 LLG reference ONLY','Reduced time','First-passage survival')
    ax[0,0].text(.03,.08,f"n={len(ft)}, events={observed.sum()}, censored={(~observed).sum()}\nMT dt=.02; not certified",transform=ax[0,0].transAxes)
    me=lit['mechanism'];ax[0,1].bar(['Edge proxy','Other resolved'],[me['edge_events'],me['resolved_events']-me['edge_events']],color=['#0072B2','#777777'])
    panel(ax[0,1],'B  Historical L100 event classifier','Mechanism proxy','Event count (clustered by path)')
    note(ax[1,0],'C  Current-model evidence gap','No L100 generated lifetime dataset.\nNo matched model/reference survival curve.\nNo completed length/temperature scaling.\nNo generated Arrhenius barrier estimate.\n\nDo not fit a lifetime from short branches.')
    note(ax[1,1],'D  Bauer literature correspondence','Paper Fig. 1: barrier versus chain length.\nPaper Fig. 2: Mz(t) and site-time map.\nPaper Fig. 3: lifetime versus length.\nPaper Fig. 4: temperature / triaxiality;\nnot the same anisotropy as current data.\n\nHistorical LLG agreement is NOT\ngenerative-model agreement.')
    savefig(fig,'Fig05_literature_reference_and_gap','文献层证据：A/B是项目已完成的历史L100真实LLG，而非生成模型。64条、约化时长750000、MT步长0.02，记录仍为complete_not_certified。首达与重复翻转间隔不是同一寿命估计；C/D明确尚缺模型大体系寿命和标度数据。')
    table('Fig05_historical_first_passage',[dict(path=i,time=ft[i],observed=int(observed[i]),source_status=status['status']) for i in range(len(ft))])
    table('Fig05_historical_survival',[dict(time=x,survival=y) for x,y in zip(grid,survival)])
    return rows,cores


def selection_and_cost(timing, metrics):
    budget=read_json('tests/results/director_budget_analysis/summary.json')
    rows=[]
    for r in budget['rows']:
        for b in ['6000','24000']:
            rows.append(dict(seed=r['seed'],anchor=r['anchor'],split=r['split'],budget=int(b),
                mean_rmse=r['metrics'][b]['mean_rmse']['estimate'][-1],
                bond_rmse=r['metrics'][b]['bond_rmse']['estimate'][-1],
                variance_difference=r['metrics'][b]['variance_difference']['estimate'][-1]))
    table('T07_historical_budget_selection',rows)
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    history=[];cost=[]
    for seed,color in zip([11,22,33],COLORS[3:]):
        r=read_json(f'tests/results/reflection_augmentation_training/{seed}/summary.json')
        ax[0,0].plot([h['step'] for h in r['history']],[h['loss'] for h in r['history']],label=f'seed {seed}',color=color,lw=.8)
        history.extend(dict(seed=seed,step=h['step'],training_loss=h['loss']) for h in r['history'])
        cost.append(dict(seed=seed,training_seconds=r['training_seconds'],updates=r['config']['updates'],checkpoint=r['checkpoint'],selection='fixed final EMA; not validation-selected'))
    panel(ax[0,0],'A  Current training loss (not validation loss)','Optimizer updates','Batch flow-matching loss');ax[0,0].legend()
    for a,key,title in [(ax[0,1],'mean_rmse','B  Historical controlled-budget mean error'),(ax[1,0],'bond_rmse','C  Historical controlled-budget bond error')]:
        for seed,color in zip([11,22,33],COLORS[3:]):
            for split,ls in [('train','-'),('development','--')]:
                values=[np.mean([r[key] for r in rows if r['seed']==seed and r['split']==split and r['budget']==b]) for b in [6000,24000]]
                a.plot([6000,24000],values,color=color,ls=ls,marker='o',label=f'{seed} {split}')
        panel(a,title,'Updates (schedule also changes)',key);a.legend(fontsize=7)
    note(ax[1,1],'D  What has NOT been done','Current final EMA was budget-fixed.\nNo per-checkpoint validation curve exists.\nOld development families informed design.\nThey are not an untouched final test.\n\nChoose candidates on several physical\nmetrics; freeze selection before new test.\nDo not choose a seed only for a good plot.')
    savefig(fig,'S04_model_selection','选型证据：A为当前模型训练损失，没有伪造验证loss。B/C来自旧的同数据预算对照，训练/开发按原始parent分开，且学习率日程随预算改变；不能视为当前增强模型的逐checkpoint选择曲线。')
    table('T07_training_history',history);table('T06_training_cost',cost)
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for x,arm in enumerate(DISPLAY_ARMS):
        i=ARMS.index(arm)
        vals=[r['seconds_per_path'] for r in timing if r['arm']==arm]
        ax[0,0].scatter([x]*len(vals),vals,color=COLORS[i],marker='o');ax[0,0].plot([x-.2,x+.2],[np.median(vals)]*2,color=COLORS[i])
        errors=[r['mean_rmse'] for r in metrics if r['arm']==arm and r['horizon']==8]
        if arm!='reference0':ax[1,0].scatter(np.median(vals),np.mean(errors),color=COLORS[i],label=LABELS[i],marker='s')
    ax[0,0].set_xticks(range(2),['MT (CPU)','Flow22 (GPU)']);ax[0,0].set_yscale('log')
    panel(ax[0,0],'A  Recorded cost, INCLUDING save','Method (hardware not matched)','Seconds/path (log scale)')
    selected_cost=next(r for r in cost if r['seed']==22)
    ax[0,1].bar(['seed 22'],[selected_cost['training_seconds']/60],color=COLORS[4])
    panel(ax[0,1],'B  Selected model training cost','Training seed','Minutes')
    ax[1,0].set_xscale('log');panel(ax[1,0],'C  Descriptive cost/error, NOT speedup','Median seconds/path (log scale)','Mean-field RMSE at t=8');ax[1,0].legend(fontsize=8)
    note(ax[1,1],'D  Efficiency claims still require','Matched CPU/GPU reference benchmark.\nSame size, duration, precision and I/O.\nEqual statistical error, not equal n only.\nData-generation + training amortization.\n\nNo measured dataset-size learning curve.\nNo certified end-to-end acceleration.')
    savefig(fig,'Fig06_cost_and_missing_controls','展示选中flow22的记录成本：约1.57秒/条（批量摊销，含保存），本次训练约4.92分钟。真实MT约0.0867秒/条，但CPU/GPU硬件不同，不能作公平加速比。C保留精度背景，D列出缺失控制；其他种子/基线成本仍在表中。')
    return cost


def historical_baselines():
    base='data/research/runs/bauer_first_loop/20260913_v1/'
    fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained');rows=[]
    for j,theta in enumerate([.11,.13]):
        for kind,color in [('deterministic','#777777'),('rfm','#0072B2')]:
            for seed in [11,22,33]:
                p=np.load(register(base+f'{kind}_{seed}_T{theta:.2f}.npy'),mmap_mode='r')
                # Four initial states, each with sixteen branches. Average within-state variances.
                assert p.shape == (4,16,201,25,3)
                variance=grouped_variance(np.asarray(p)).mean(0)
                ax[j].plot(range(len(variance)),variance,color=color,ls=['-','--',':'][[11,22,33].index(seed)],label=f'{kind} {seed}')
                rows.extend(dict(theta=theta,kind=kind,seed=seed,frame=i,conditional_spin_variance=v,initials=4,paths_per_initial=16) for i,v in enumerate(variance))
        panel(ax[j],f'Historical first-loop ONLY: theta={theta}','Saved frame (old protocol)','Conditional spin variance');ax[j].legend(fontsize=8)
    savefig(fig,'S05_historical_deterministic_baseline','历史确定性对照：旧first-loop同骨干确定性模型与旧RFM，同起点条件方差。它不是当前局部转移流的同预算对照，不能用来证明当前模型胜出。确定性条件分布退化，重复相同输入不会产生热噪声分支。')
    table('T08_historical_baseline_variance',rows)


def all_state_views(records, manifest, t):
    fig,axs=plt.subplots(3,4,figsize=(16,10),layout='constrained')
    for state,ax in enumerate(axs.flat):
        for i,arm in enumerate(ARMS):
            q=np.quantile(records[state,arm]['mz'],[.1,.5,.9],axis=0)
            ax.plot(t,q[1],color=COLORS[i],ls=STYLES[i],label=LABELS[i])
            if arm=='reference0':ax.fill_between(t,q[0],q[2],color='black',alpha=.1)
        s=manifest['rows'][state]
        panel(ax,f"State {state}: {s['kind']}, theta={s['theta']}",'Reduced time','Mz median')
        ax.set_ylim(-1.03,1.03)
    axs[0,0].legend(fontsize=6)
    savefig(fig,'S06_all_initial_states','全部12个起点，无删除失败状态。线为条件中位数，黑色阴影为真实参考10–90%轨迹区间。共同纵轴为[-1,1]；不是置信区间。')
    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained');rows=[]
    edges_m=np.linspace(-1,1,21);edges_e=np.linspace(-1.06,0,31)
    for i,(arm,ax) in enumerate(zip(ARMS,axs.flat)):
        ob=records[3,arm];h,_,_=np.histogram2d(ob['mz'][:,-1],ob['energy'][:,-1],bins=[edges_m,edges_e])
        im=ax.imshow(h.T,origin='lower',extent=[-1,1,-1.06,0],aspect='auto',vmin=0,vmax=128,cmap='viridis')
        panel(ax,LABELS[i],'Mz at t=8','Energy per site / J')
        rows.extend(dict(arm=arm,path=k,mz=ob['mz'][k,-1],energy_per_site=ob['energy'][k,-1]) for k in range(128))
    fig.colorbar(im,ax=list(axs.flat),label='Paths per fixed bin (shared scale)')
    savefig(fig,'S07_joint_energy_magnetization','state3终态能量—磁化联合分布：E/J=-sum邻键点积-0.1sum(sz²)，再除L。所有面板同分箱、同色标，128路径/方法；数据用于检查均值相似是否掩盖结构差异。')
    table('S07_joint_endpoint',rows)
    old=read_json('tests/results/sample_tangent_independent_comparison/summary.json')
    rows=[]
    for r in old['rows']:
        rows.append(dict(arm=r['label'],anchor=r['anchor'],theta=r['state']['theta'],kind=r['state']['kind'],
                         **r['terminal'],joint_energy=r['joint_energy'],source=r['path']))
    table('T09_previous_24_state_comparison',rows)
    fig,axs=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for j,key in enumerate(['mean_rmse','bond_rmse','variance_ratio']):
        for i,arm in enumerate(ARMS[2:],2):
            rs=sorted([r for r in rows if r['arm']==arm],key=lambda r:r['anchor'])
            axs[j].plot([r['anchor'] for r in rs],[r[key] for r in rs],color=COLORS[i],ls=STYLES[i],marker='o',ms=3,label=LABELS[i])
        panel(axs[j],f'Old independent bank: {key}','Initial-state ID (24 conditions)',key)
    axs[2].axhline(1,color='black',ls=':');axs[0].legend(fontsize=7)
    savefig(fig,'S08_previous_24_state_validation','旧独立初态库：24条件×4候选，512分支/条件，终点0.8。此库已用于开发讨论，不再是未揭盲最终测试。保留全部96组描述性结果，不能与新12起点当作同一随机总体拼接。')


def evidence_tables():
    rows=[]
    for p in sorted((ROOT/'tests/results').rglob('summary.json')):
        try:d=json.loads(p.read_text())
        except (ValueError,OSError):continue
        rows.append(dict(report=str(p.relative_to(ROOT)),status=d.get('status','unspecified'),
                         rows=len(d.get('rows',[])),scope=d.get('scope',''),
                         treatment='indexed, not pooled: source protocol/candidate may differ'))
    table('T00_existing_experiment_index',rows)
    table('T10_literature_mapping',[
        dict(source='Bauer2011 Eq.2 / Appendix A',observable='drift, thermal fluctuation, conditional increment',ours='S02',status='new ML diagnostic; not a paper numerical curve',url='https://arxiv.org/html/1010.4730v2'),
        dict(source='Bauer2011 Fig.2',observable='Mz(t), site-time sz, edge nucleation and wall return/propagation',ours='Fig02 / Fig04',status='L25 short-time analogue; target L100 model result missing',url='https://arxiv.org/html/1010.4730v2'),
        dict(source='Bauer2011 Fig.3',observable='lifetime versus chain length',ours='Fig05',status='current frozen-model length scan missing; paper K=.1 curve is visually scaled x10',url='https://arxiv.org/html/1010.4730v2'),
        dict(source='Bauer2011 Fig.1 / Eq.4',observable='Arrhenius barrier and chain-length crossover',ours='Fig05',status='current model temperature/length scan missing',url='https://arxiv.org/html/1010.4730v2'),
        dict(source='Bauer2011 Fig.4',observable='temperature dependence with triaxial anisotropy',ours='not direct comparison',status='different parameter family; current K=.1 uniaxial data cannot reproduce it',url='https://arxiv.org/html/1010.4730v2'),
        dict(source='Bauer2011 Appendix A',observable='mean interval between successive reversals',ours='historical L100 audit',status='not identical to first-passage survival or finite-horizon B-first probability',url='https://arxiv.org/html/1010.4730v2')])
    table('T11_model_definitions',[
        dict(model='current flow',training='12288 coverage-mixed pairs, lag=.01, 24000 updates, seeds11/22/33',distribution='non-Gaussian conditional spherical transition',prior='local equivariance, director sign, reflection augmentation',selection='fixed final EMA, not validation-selected',comparison='current12-state and previous24-state banks'),
        dict(model='Gaussian tangent baseline',training='same coverage-mixed pairs; least-squares fit separately at theta=.11/.13',distribution='four-feature tangent mean + isotropic projected Gaussian; exponential map',prior='neighbor and easy-axis local basis; no supplied force coefficients',selection='training-only closed-form fit',comparison='matched current initial states and lag; stronger structural assumptions'),
        dict(model='historical deterministic',training='old first-loop data/backbone/budget, not current flow training',distribution='one output per fixed complete input; conditional point mass',prior='old same-backbone experiment; unit-sphere output',selection='historical fixed budget',comparison='S05 historical only; no current matched deterministic campaign')])


def main():
    import os
    os.chdir(ROOT)
    for sub in ['figures','tables','source_data','logs']:(OUT/sub).mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':8})
    inputs,seeds,inventory=training_data();print('Training split audited',flush=True)
    manifest,records,t,timing=current_branches()
    cover=coverage(inputs,seeds,records,manifest)
    metrics,local=metric_rows(records,manifest,t)
    ranking=[]
    for arm in ['flow11','flow22','flow33']:
        rs=[r for r in metrics if r['arm']==arm and r['horizon']==8]
        ranking.append(dict(arm=arm,mean_rmse=float(np.mean([r['mean_rmse'] for r in rs])),
                            bond_rmse=float(np.mean([r['bond_rmse'] for r in rs])),mz_W1=float(np.mean([r['mz_W1'] for r in rs]))))
    ranking.sort(key=lambda r:r['mean_rmse'])
    assert ranking[0]['arm']=='flow22','Display selection no longer matches the documented rule'
    table('T12_display_model_ranking',ranking)
    (OUT/'source_data/display_selection.json').write_text(json.dumps(dict(selected='flow22',checkpoint='tests/output/reflection_augmentation/22/final.pt',
        criterion='Post-hoc display choice: minimum mean field RMSE over 12 development states at t=8; other metrics reported, not combined.',
        scope='Best among these three flow seeds on this development bank, not global best or independent final selection; all seed data retained.',ranking=ranking),indent=2))
    make_main(records,manifest,t,inventory,metrics,local,timing)
    events,cores=events_and_literature(records,manifest,t)
    costs=selection_and_cost(timing,metrics)
    historical_baselines()
    all_state_views(records,manifest,t)
    evidence_tables()
    # Compact derived numbers for final narrative; no raw dataset copy.
    aggregates=[]
    for arm in ARMS[1:]:
        for horizon in [.01,.08,.8,2.,8.]:
            rs=[r for r in metrics if r['arm']==arm and r['horizon']==horizon]
            aggregates.append(dict(arm=arm,horizon=horizon,mean_rmse=float(np.mean([r['mean_rmse'] for r in rs])),
                bond_rmse=float(np.mean([r['bond_rmse'] for r in rs])),mz_W1=float(np.mean([r['mz_W1'] for r in rs]))))
    table('T04_descriptive_aggregate',aggregates)
    summary=dict(status='EXPLORATORY_PAPER_PROGRESS_NOT_CERTIFIED',built_at=datetime.datetime.now().astimezone().isoformat(),
                 figures=FIGURES,inventory=inventory,coverage=cover,aggregate=aggregates,training_cost=costs,
                 source_files=SOURCES,code_sha256=sha(__file__),python=sys.version,numpy=np.__version__,matplotlib=matplotlib.__version__)
    (OUT/'source_data/build_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    # Single HTML contains all images: downloading only this file works offline.
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Bauer paper progress</title>',
           '<style>body{max-width:1250px;margin:30px auto;padding:20px;font:17px sans-serif;line-height:1.6}img{width:100%;height:auto}section{border-top:1px solid #bbb;margin:30px 0}a{color:#005a8b}</style>',
           '<h1>Bauer 路径生成：当前六图证据包</h1><p>主图只展示当前三个流模型中选中的seed22，与真实LLG比较。选择依据是12个开发起点t=8的平均均值场RMSE最低；这是展示选择，不是独立最终认证。其他模型保留在补充图表。图像已内嵌，单独下载本HTML即可离线查看。详细口径见README.md与tables/。</p>']
    for idx,f in enumerate(sorted(FIGURES,key=lambda f:f['name'])):
        if idx==6:parts.append('<details><summary>展开补充图：数据覆盖、局部统计和所有候选对照（完整保留）</summary>')
        data=base64.b64encode((OUT/'figures'/(f['name']+'.png')).read_bytes()).decode()
        parts.append(f'<section><h2>{html.escape(f["name"])}</h2><p>{html.escape(f["caption"])}</p><img alt="{html.escape(f["caption"],quote=True)}" src="data:image/png;base64,{data}"></section>')
    parts.append('</details></html>');(OUT/'index.html').write_text('\n'.join(parts))
    print(json.dumps(dict(figures=len(FIGURES),sources=len(SOURCES),aggregate_at8=[r for r in aggregates if r['horizon']==8]),indent=2),flush=True)


if __name__=='__main__':
    main()
