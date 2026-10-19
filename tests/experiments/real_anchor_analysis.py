"""Complete-batch descriptive restart analysis; no significance/pass threshold."""
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.restart_observables import matched_restart


def validate_report(report, cfg):
    expected = {(f, t, start) for f in range(4) for t in (.11, .13) for start in cfg['start_frames']}
    keys = [(r['family'], r['theta'], r['start_frame']) for r in report['rows']]
    if report['status'] != 'COMPLETE_REAL_ANCHOR_PROBE_NOT_CERTIFIED':
        raise ValueError('complete generation required')
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('missing or duplicated conditional restart')
    if report['config'] != cfg:
        raise ValueError('generation configuration changed')


def main():
    cfg_path = Path('tests/conf/real_anchor_probe.json')
    cfg = json.loads(cfg_path.read_text())
    reports = []
    for seed in cfg['training_seeds']:
        p = Path(f'tests/results/real_anchor_probe/{seed}/summary.json')
        report = json.loads(p.read_text()); validate_report(report, cfg)
        if report['config_sha256'] != digest(cfg_path) or report['audit_sha256'] != digest(cfg['audit']):
            raise ValueError('changed configuration or audit')
        reports.append((seed, p, report))
    out = Path('tests/results/real_anchor_analysis')
    if out.exists():
        raise FileExistsError('preserve previous analysis')
    out.mkdir()
    result = dict(status='RUNNING', rows=[],
        scope='Descriptive transition averages over real intermediate states, not a single-anchor conditional law. No uncertainty intervals, autonomous-rollout comparison or certification in this output.',
        code_sha256={p:digest(p) for p in [__file__, 'tests/experiments/restart_observables.py']},
        generation_reports={str(p):digest(p) for _,p,_ in reports})
    save(out/'summary.json', result)
    for seed, _, report in reports:
        for row in report['rows']:
            if digest(row['path']) != row['sha256'] or digest(row['reference_path']) != row['reference_sha256']:
                raise ValueError('generated/reference file changed')
            n, start, frames = cfg['n_parents'], row['start_frame'], cfg['frames']
            with np.load(row['path']) as d:
                g = d['spins']
                if not np.array_equal(d['parent_index'], np.arange(n)):
                    raise ValueError('parent order changed')
                if not np.array_equal(d['reduced_time'], np.arange(frames)*80):
                    raise ValueError('generated time grid mismatch')
            with np.load(row['reference_path']) as d:
                r = d['spins'][:n, start:start+frames]
                times = d['reduced_time'][start:start+frames]
                if not np.array_equal(times-times[0], np.arange(frames)*80):
                    raise ValueError('reference time grid mismatch')
            if g.shape != (n, frames, 25, 3):
                raise ValueError('unexpected restart shape')
            measures = matched_restart(g, r)
            name = f"seed{seed}_family{row['family']}_T{row['theta']}_start{start}"
            path = out/f'{name}.npz'
            np.savez_compressed(path, parent_index=np.arange(n),
                                **{f'{group}/{k}':v for group, obs in measures.items() for k,v in obs.items()})
            result['rows'].append(dict(seed=seed, family=row['family'], theta=row['theta'], start_frame=start,
                per_parent_path=str(path), sha256=digest(path),
                mean_curves={group:{k:v.mean(axis=0).tolist() for k,v in obs.items()}
                             for group,obs in measures.items()}))
            save(out/'summary.json', result)
    result['status']='COMPLETE_RESTART_MOMENTS_NOT_CERTIFIED'; save(out/'summary.json', result)


if __name__ == '__main__':
    main()
