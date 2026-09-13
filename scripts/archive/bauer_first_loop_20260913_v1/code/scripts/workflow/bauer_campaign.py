"""Frozen classical-chain campaign. Scientific failures are records, never successes.

CPU Numba dynamics, CUDA learned models. Each stage writes its own result; the
final report inventories missing/failed jobs without an interactive monitor.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import h5py
import numpy as np
from scripts.core.bauer_fast import ensemble
from scripts.literature.bauer_2011.analyze import reversals


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def seed(*parts):
    return int.from_bytes(hashlib.sha256('/'.join(map(str, parts)).encode()).digest()[:4], 'little')


def initial_states(count, length, family):
    rng = np.random.default_rng(seed('initial', family))
    s = rng.normal(0, .05, (count, length, 3)); s[..., 2] += 1
    return s / np.linalg.norm(s, axis=-1, keepdims=True)


def graph(length, device='cpu', dtype=None):
    from scripts.model.gate_f import CellGraph
    cell = SimpleNamespace(shape=(length, 1), basis=1, periodic=(False, False),
                           templates=[(0, 0, (1, 0), 1.)], anisotropy=(0., 0., .1))
    return CellGraph(cell).to(device=device, dtype=dtype)


def observables(paths):
    """[path,time,site,xyz] -> E/site, Mxyz, spatial & temporal diagnostics."""
    length = paths.shape[-2]
    e = (-(paths[..., :-1, :]*paths[..., 1:, :]).sum((-2, -1))
         -.1*(paths[..., 2]**2).sum(-1)) / length
    m = paths.mean(-2)
    corr = np.stack([(paths[..., :-r, :]*paths[..., r:, :]).sum(-1).mean(-1)
                     for r in (1, max(2, length//4), length//2)], -1)
    walls = (np.diff(np.sign(paths[..., 2]), axis=-1) != 0).sum(-1)/length
    edge = np.stack((paths[..., 0, 2], paths[..., -1, 2]), -1)
    return np.concatenate((e[..., None], m, corr, walls[..., None], edge), -1)


def run_reference(config, root, index):
    case = config['reference_cases'][index]
    name = case['id']; directory = root/'reference'/name; directory.mkdir(parents=True, exist_ok=True)
    start = time.monotonic(); n = case['paths']; length = case['L']
    dt = case['dt']; stride = round(case['save_dt']/dt)
    block = min(10000., case['duration']); steps = round(block/dt)
    if steps % stride or abs(steps*dt-block)>1e-8 or case['duration'] % block:
        raise ValueError('exact integral block, save and dt grid required')
    state = np.zeros((n, length, 3)); state[..., 2] = 1.
    checkpoint = directory/'checkpoint.npz'; first_block = 0
    if checkpoint.exists():
        with np.load(checkpoint) as c:
            state = c['state']; first_block = int(c['next_block'])
    path = directory/'observables.h5'
    if checkpoint.exists() and not path.exists():
        raise ValueError('checkpoint without matching raw data')
    mode = 'r+' if path.exists() else 'w'
    with h5py.File(path, mode) as h:
        if mode == 'w':
            h.create_dataset('values', (n, 1, 10), maxshape=(n, None, 10), dtype='f8', chunks=(1, 501, 10))
            h['values'][:, :1] = observables(state[:, None])
            h.create_dataset('spins', data=state[:, None].astype('f4'), maxshape=(n, None, length, 3),
                             chunks=(1, 128, length, 3), compression='lzf')
            h.attrs['case_json'] = json.dumps(case); h.attrs['complete'] = False
        if json.loads(h.attrs['case_json']) != case:
            raise ValueError('checkpoint case identity mismatch')
        h['values'].resize(1+first_block*(steps//stride), axis=1)
        h['spins'].resize(1+first_block*(steps//stride), axis=1)
        maximum = float(h.attrs.get('raw_norm_error', 0.))
        for block_id in range(first_block, round(case['duration']/block)):
            samples, errors = ensemble(state, 100000000+index*1000000+block_id*n+np.arange(n, dtype=np.int64),
                                       dt, steps, stride, .1, .1, case['theta'],
                                       int(case.get('method', 'mt') == 'heun'), case.get('project', True))
            if not np.isfinite(samples).all():
                raise ValueError('nonfinite dynamics: numerical failure, not censoring')
            values = observables(samples[:, 1:]); old = h['values'].shape[1]
            h['values'].resize(old+values.shape[1], axis=1); h['values'][:, old:] = values
            h['spins'].resize(old+values.shape[1], axis=1); h['spins'][:, old:] = samples[:, 1:].astype('f4')
            state = samples[:, -1].copy(); maximum = max(maximum, float(errors.max()))
            h.attrs['raw_norm_error'] = maximum; h.flush()
            temp = directory/'checkpoint.tmp.npz'
            np.savez(temp, state=state, next_block=block_id+1); temp.replace(checkpoint)
            print(json.dumps(dict(case=name, completed_duration=(block_id+1)*block, wall_seconds=time.monotonic()-start)), flush=True)
        h.attrs['complete'] = True; h.attrs['wall_seconds'] = time.monotonic()-start
    save(directory/'result.json', dict(status='complete_not_certified', case=case, data_sha256=digest(path),
                                      wall_seconds=time.monotonic()-start, job_id=os.getenv('SLURM_JOB_ID')))


def event_rows(mz, times, threshold, dwell):
    out = []; all_events = []
    horizon = times[-1]
    for path in mz:
        events = reversals(times, path, threshold, dwell)
        first = events[0] if len(events) else horizon
        out.append([float(len(events)>0), float(len(events)>1), first/horizon,
                    *[float(first>t or len(events)==0) for t in np.linspace(0, horizon, 11)[1:]]])
        all_events.append(events)
    return np.asarray(out), all_events


def mean_ci(a, b, repetitions, rng, confidence=1-.05/6):
    """Independent path-cluster bootstrap, simultaneous family across observables."""
    point = a.mean(0)-b.mean(0)
    w1 = rng.multinomial(len(a), np.full(len(a), 1/len(a)), size=repetitions)/len(a)
    w2 = rng.multinomial(len(b), np.full(len(b), 1/len(b)), size=repetitions)/len(b)
    draws = w1@a-w2@b
    radius = np.quantile(np.max(abs(draws-point), axis=1), confidence)
    return dict(confidence=confidence, estimate=point.tolist(), ci=np.stack((point-radius, point+radius), -1).tolist())



def mechanism_summary(file_path, events, save_dt, bootstrap):
    """Operational coarse-wall diagnosis. Resolution sensitivity remains explicit."""
    per_path=[]
    with h5py.File(file_path) as h:
        for i, completions in enumerate(events):
            z=h['spins'][i,...,2]
            width=min(5,z.shape[-1])
            smooth=np.stack([z[:,j:j+width].mean(1) for j in range(z.shape[1]-width+1)],-1)
            m=z.mean(-1);previous=0;basin=1;edge=resolved=0;propagation=[]
            for t in completions:
                end=round(t/save_dt)
                stable=np.flatnonzero(basin*m[previous:end+1]>=.6)
                begin=previous+int(stable[-1]) if len(stable) else previous
                candidate=basin*smooth[begin:end+1]<-.3
                frames=np.flatnonzero(candidate.any(-1))
                if len(frames):
                    first=int(frames[0]);positions=np.flatnonzero(candidate[first])
                    at_edge=bool(positions[0]==0 or positions[-1]==smooth.shape[-1]-1)
                    edge+=at_edge;resolved+=1;propagation.append((end-begin-first)*save_dt)
                previous=end;basin=-basin
            per_path.append([edge,resolved,len(completions),sum(propagation)])
    v=np.asarray(per_path,float);rng=np.random.default_rng(seed('mechanism',str(file_path)))
    w=rng.multinomial(len(v),np.full(len(v),1/len(v)),size=bootstrap);boot=w@v
    good=boot[:,1]>0
    return dict(edge_events=int(v[:,0].sum()),resolved_events=int(v[:,1].sum()),total_events=int(v[:,2].sum()),
                edge_fraction=float(v[:,0].sum()/v[:,1].sum()) if v[:,1].sum() else None,
                edge_fraction_ci95=np.quantile(boot[good,0]/boot[good,1],[.025,.975]).tolist() if good.all() else None,
                unresolved_events=int((v[:,2]-v[:,1]).sum()),
                mean_propagation_time=float(v[:,3].sum()/v[:,1].sum()) if v[:,1].sum() else None,
                estimator='first 5-site mean sz < -0.3 following last stable basin, confirmation-time endpoint',
                limitation='Discrete-frame diagnostic; nucleation can be missed between saved frames. No mechanistic PASS by itself.')


def summarize_reference(root, case, bootstrap):
    directory = root/'reference'/case['id']
    with h5py.File(directory/'observables.h5') as h:
        if not h.attrs['complete']:
            raise ValueError('incomplete reference')
        v = h['values'][:]; raw = float(h.attrs['raw_norm_error'])
    times = np.arange(v.shape[1])*case['save_dt']
    event, events = event_rows(v[..., 3], times, .6, 2*case['save_dt'])
    sums = np.array([np.diff(e).sum() for e in events]); counts = np.array([max(0, len(e)-1) for e in events])
    rng = np.random.default_rng(seed('ref_boot', case['id']))
    w = rng.multinomial(len(v), np.full(len(v), 1/len(v)), size=bootstrap)
    bs_count = w@counts; usable = bs_count>0
    recurrence = float(sums.sum()/counts.sum()) if counts.sum() else None
    ci = np.quantile((w@sums)[usable]/bs_count[usable], [.025, .975]).tolist() if usable.all() else None
    coarse, _ = event_rows(v[:, ::2, 3], times[::2], .6, 2*case['save_dt'])
    dwell_alt, _ = event_rows(v[..., 3], times, .6, 4*case['save_dt'])
    tail = v[:, len(times)//2:]
    row = dict(case=case, completed_events=sum(map(len, events)), completed_intervals=int(counts.sum()),
               recurrent_interval_mean=recurrence, recurrent_interval_ci95=ci,
               interval_sums_per_trajectory=sums.tolist(), interval_counts_per_trajectory=counts.tolist(),
               recurrence_scope='Appendix-A completed-interval statistic; finite-window bias assessed separately',
               rmst=float(event[:, 2].mean()*times[-1]), censor_fraction=float(1-event[:, 0].mean()),
               first_pass_times=(event[:, 2]*times[-1]).tolist(), first_pass_observed=event[:, 0].tolist(),
               km_survival_grid=event[:, 3:].mean(0).tolist(),
               raw_norm_error=raw, norm_projection=case.get('project', True),
               save_event_probability_difference=float(abs(coarse[:, 0].mean()-event[:, 0].mean())),
               dwell_event_probability_difference=float(abs(dwell_alt[:, 0].mean()-event[:, 0].mean())),
               tail_observable_mean=tail.mean((0,1)).tolist(),
               event_times=[e.tolist() for e in events])
    row['mechanism'] = mechanism_summary(directory/'observables.h5', events, case['save_dt'], bootstrap)
    half_events=[e[e<=times[-1]/2] for e in events]
    half_count=sum(max(0,len(e)-1) for e in half_events)
    half_mean=sum(np.diff(e).sum() for e in half_events)/half_count if half_count else None
    row['finite_window_bias_diagnostic']=dict(half_window_completed_interval_mean=half_mean,
        full_window_completed_interval_mean=recurrence,
        warning='Half/full agreement is diagnostic, not an unrestricted-lifetime proof.')
    save(directory/'analysis.json', row)
    return row, np.concatenate((tail[..., :4].mean(1), event), -1)


def analyze_reference(config, root):
    checks = {}; summaries = {}; features = {}; missing = []
    for case in config['reference_cases']:
        try:
            summaries[case['id']], features[case['id']] = summarize_reference(root, case, config['bootstrap'])
        except (OSError, KeyError, ValueError) as exc:
            missing.append(dict(case=case['id'], reason=str(exc)))
    rng = np.random.default_rng(914)
    for scope in ('original', 'model'):
        ids = [f'{scope}_dt{dt}' for dt in ('002', '001', '0005')]
        if all(i in summaries for i in ids):
            a = features[ids[0]]; scale = np.maximum(np.std(features[ids[-1]], axis=0), .1)
            for target in ids[1:]:
                comparison = mean_ci(a/scale, features[target]/scale, config['bootstrap'], rng)
                comparison['tolerance'] = .1
                comparison['pass'] = bool(np.max(np.abs(comparison['ci']))<=.1)
                checks[f'{scope}_weak_convergence_{target}'] = comparison
    # Projection and independent discretization are statistical checks, never common-Wiener MT claims.
    for target in ('model_unprojected', 'model_heun'):
        if target in features and 'model_dt0005' in features:
            scale = np.maximum(np.std(features['model_dt0005'], axis=0), .1)
            cmp = mean_ci(features[target]/scale, features['model_dt0005']/scale, config['bootstrap'], rng)
            cmp.update(tolerance=.1, passed=bool(np.max(np.abs(cmp['ci']))<=.1))
            checks[target] = cmp
    for name, summary in summaries.items():
        checks[name+'_save_and_dwell'] = dict(pass_=max(summary['save_event_probability_difference'],
                                                           summary['dwell_event_probability_difference'])<=.05)
    fits = []
    for length in (25, 50, 100):
        names = [f'life_L{length}_T{t}' for t in ('011', '012', '013')]
        if all(n in summaries and summaries[n]['recurrent_interval_mean'] for n in names):
            rows = [summaries[n] for n in names]
            x = 1/np.array([r['case']['theta'] for r in rows]); y = np.log([r['recurrent_interval_mean'] for r in rows])
            se = np.array([(r['recurrent_interval_ci95'][1]-r['recurrent_interval_ci95'][0])/3.92/r['recurrent_interval_mean']
                           if r['recurrent_interval_ci95'] else np.inf for r in rows])
            coefficients, covariance = np.polyfit(x, y, 1, w=1/np.maximum(se, 1e-8), cov='unscaled')
            barrier = float(coefficients[0]); error = float(np.sqrt(covariance[0,0]))
            draws=[]
            for r in rows:
                counts=np.asarray(r['interval_counts_per_trajectory']);sums=np.asarray(r['interval_sums_per_trajectory'])
                w=rng.multinomial(len(counts),np.full(len(counts),1/len(counts)),size=config['bootstrap'])
                total=w@counts;draws.append(np.where(total>0,(w@sums)/np.maximum(total,1),np.nan))
            boot_y=np.log(np.asarray(draws));valid=np.isfinite(boot_y).all(0)
            boot_fit=np.polyfit(x,boot_y[:,valid],1) if valid.any() else None
            fits.append(dict(L=length, barrier=barrier, intercept=float(coefficients[1]),
                             covariance=covariance.tolist(), barrier_ci95=[barrier-1.96*error,barrier+1.96*error],
                             barrier_bootstrap_ci95=np.quantile(boot_fit[0],[.025,.975]).tolist() if valid.all() else None,
                             intercept_bootstrap_ci95=np.quantile(boot_fit[1],[.025,.975]).tolist() if valid.all() else None,
                             bootstrap_unit='independent parent trajectories within each temperature',
                             wall_energy=float(2*np.sqrt(.2)),
                             barrier_relative_error=abs(barrier/(2*np.sqrt(.2))-1),
                             minimum_500_events=all(r['completed_events']>=500 for r in rows),
                             status='extension_grid_not_published_temperature_points'))
    save(root/'reference_certificate.json', dict(
        status='HOLD', numerical_checks=checks, missing=missing, arrhenius_fits=fits,
        original_case=summaries.get('life_L100_T011'),
        publication_gate='Unresolved: published lifetime-temperature grid is not supplied. No strict P1 release.',
        production_dt=.02, production_dt_certified=False, numerical_comparisons_passed=bool(checks) and not missing and all(
            c.get('pass', c.get('passed', c.get('pass_', False))) for c in checks.values()),
        limitations=['Completed intervals do not identify unrestricted lifetime without censoring/bias assessment.',
                     'Mechanism and length scaling require reviewed literature-matched targets; no automatic waiver.']))


def pilot(config, root):
    cfg = config['model']; out = root/'pilot'; out.mkdir(parents=True, exist_ok=True)
    records = []; start = time.monotonic()
    for theta_index,theta in enumerate(cfg['train_theta']):
        initial = initial_states(cfg['pilot_initials'], cfg['L'], f'pilot/{theta}')
        states = np.repeat(initial, cfg['pilot_noise'], axis=0)
        paths, err = ensemble(states, 200000000+theta_index*1000000+np.arange(len(states)),
                              .02, round(cfg['pilot_horizon']/.02), round(cfg['pilot_save_dt']/.02), .1, .1, theta)
        with h5py.File(out/f'T{theta}.h5', 'w') as h:
            h.create_dataset('spins', data=paths.astype('f4'), compression='lzf'); h.attrs['complete'] = True
        v = observables(paths); times = np.arange(paths.shape[1])*cfg['pilot_save_dt']
        row = dict(theta=theta, raw_norm_error=float(err.max()), windows=[])
        for horizon in cfg['candidate_windows']:
            count = round(horizon/cfg['pilot_save_dt'])+1
            e, _ = event_rows(v[:, :count, 3], times[:count], .6, 2*cfg['pilot_save_dt'])
            # Binomial precision is a lower bound: the conditional-initial variation is also reported.
            p = e[:, 0].mean(); se = np.sqrt(p*(1-p)/len(e))
            row['windows'].append(dict(horizon=horizon, reversal_probability=float(p),
                                        nominal_halfwidth95=float(1.96*se), completed_first_passes=int(e[:,0].sum()),
                                        conditional_p=e[:,0].reshape(cfg['pilot_initials'],cfg['pilot_noise']).mean(1).tolist()))
        late = v[:, len(times)//2:, 3]
        hist, edges = np.histogram(late, bins=np.linspace(-1, 1, 41), density=True)
        positive = hist[edges[:-1]>=.4].max(); negative = hist[edges[1:]<=-.4].max()
        center = hist[(edges[:-1]>=-.2)&(edges[1:]<=.2)].mean()
        row['basin_histogram'] = dict(density=hist.tolist(), edges=edges.tolist(),
                                      two_peak_to_center=float(min(positive,negative)/max(center,1e-12)))
        # Finite-chain orientational correlation, no exponential-fit claim in a magnetized mixture.
        row['spatial_correlation'] = v[:, len(times)//2:, 4:7].mean((0,1)).tolist()
        row['correlation_length_status'] = 'finite-chain correlation diagnostic; xi estimator not certified'
        records.append(row)
    viable = [w for w in cfg['candidate_windows'] if all(any(r['horizon']==w and .1<=r['reversal_probability']<=.8
                    and r['completed_first_passes']>=20 for r in row['windows']) for row in records)]
    save(root/'pilot_result.json', dict(status='HOLD', records=records,
         candidate_window=min(viable) if viable else None,
         reasons=['P2 requires reviewed basin/correlation-length, mechanism resolution and physically justified event margins.',
                  'No default event tolerances or power certificate are substituted for those requirements.'],
         wall_seconds=time.monotonic()-start))

    from scripts.analysis.bauer_campaign_plot import run as plot
    plot(root, pilot_only=True)


def gate(config, root):
    reference = load(root/'reference_certificate.json') if (root/'reference_certificate.json').exists() else {'status':'missing'}
    pilot_result = load(root/'pilot_result.json') if (root/'pilot_result.json').exists() else {'status':'missing'}
    software = load(root/'software/result.json') if (root/'software/result.json').exists() else {'status':'missing'}
    passed = all(r['status']=='PASS' for r in (reference, pilot_result, software))
    save(root/'release.json', dict(status='GO' if passed else 'HOLD',
         stages=dict(P1=reference['status'],P2=pilot_result['status'],software=software['status']),
         training_released=passed,
         reason='Scientific reference and event contract must pass before certified generation research.'))


def report(config, root):
    results = {}
    for name in ('software/result', 'physics/result', 'reference_certificate', 'pilot_result', 'release'):
        p = root/(name+'.json')
        results[name] = load(p) if p.exists() else dict(status='MISSING_JOB_FAILED_OR_NOT_COMPLETED')
    runners={p.name:load(p) for p in (root/'runner_status').glob('*.json')}
    summary = dict(status='HOLD', results=results, runner_exit_codes=runners,
                   completion_claim='Submitted independent tests have evidence only where a result artifact exists.',
                   model_claim='Model software checks do not certify learned path distributions. P3 training and P5 are not submitted while P1/P2 remain unresolved.',
                   config_sha256=digest(root/'config.json'))
    save(root/'final_report.json', summary)
    lines = ['# 经典链独立验证结果', '', '本报告自动汇总；缺失、失败或统计不足不会记为通过。', '',
             '| 项目 | 状态 |', '|---|---|']
    lines.extend(f'| {name} | {result["status"]} |' for name,result in results.items())
    lines += ['', '## 核查入口', '', '`reference/*/analysis.json`：步长、反转、删失与寿命统计；',
              '`pilot_result.json`：固定完整初态下独立噪声、候选窗口、磁盆和空间关联；',
              '`software/result.json`：开放链网络、球面流、协变、梯度与各基线；',
              '`reference_certificate.json`：文献对照限制与逐项数值检查。', '',
              '当前不宣称 P1–P5 全部通过，也不宣称已证明生成模型的物理准确性。']
    from scripts.analysis.bauer_campaign_plot import run as plot
    try:
        plot(root)
    except (OSError,ValueError,KeyError) as exc:
        save(root/'plot_error.json',dict(status='FAILED',reason=str(exc)))
        lines.extend(['', 'Plot generation failed; inspect plot_error.json.'])
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--stage', required=True, choices=('reference','analyze','pilot','software','physics','gate','report'))
    p.add_argument('--index', type=int, default=int(os.getenv('SLURM_ARRAY_TASK_ID', '0')))
    p.add_argument('--device', default='cpu'); args = p.parse_args()
    config = load(args.config); root = Path(config['output']); root.mkdir(parents=True, exist_ok=True)
    if args.stage == 'reference': run_reference(config, root, args.index)
    elif args.stage == 'analyze': analyze_reference(config, root)
    elif args.stage == 'pilot': pilot(config, root)
    elif args.stage == 'software':
        from scripts.validation.bauer_model_checks import run
        run(root, args.device)
    elif args.stage == 'gate': gate(config, root)
    elif args.stage == 'physics':
        from scripts.validation.bauer_physics_checks import run
        run(root)
    elif args.stage == 'report': report(config, root)


if __name__ == '__main__':
    main()
