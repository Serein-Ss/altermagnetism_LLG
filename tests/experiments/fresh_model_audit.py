"""Frozen-model fresh-state rollout comparisons, with sampler refinement."""
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.director_budget_analysis import paired_statistics
from tests.experiments.director_convergence_summary import full_covariance
from tests.experiments.fresh_reference_audit import FRAMES, path_energy


def temporal_covariance(p):
    p = np.asarray(p, dtype=float)
    c = (p-p.mean(0)).reshape(len(p), p.shape[1], -1)
    return np.einsum('ntd,nud->tu', c, c)/((len(p)-1)*p.shape[2])


def sampler_statistics(data, rng, replicates=2000):
    arrays, metrics = paired_statistics(data, rng, replicates)
    labels = {'6000': '32', '24000': '64', 'paired_change': 'paired_change'}
    renamed = {}
    for key, value in arrays.items():
        prefix = next(k for k in labels if key.startswith(k+'_'))
        renamed[labels[prefix]+key[len(prefix):]] = value
    return renamed, {labels[k]: v for k, v in metrics.items()}


def run():
    ref_path = Path('tests/results/fresh_rollout_reference/summary.json')
    ref = json.loads(ref_path.read_text())
    assert ref['status'] == 'COMPLETE_FRESH_ROLLOUT_REFERENCE_NOT_CERTIFIED'
    numerical_path = Path('tests/results/fresh_reference_audit/summary.json')
    numerical = json.loads(numerical_path.read_text())
    assert numerical['status'] == 'COMPLETE_FRESH_REFERENCE_AUDIT_NOT_CERTIFIED'
    assert numerical['source_sha256'] == digest(ref_path) and len(numerical['rows']) == 24
    references = {r['anchor']: r for r in ref['rows'] if r['case'] == 1}
    assert set(references) == set(range(12))
    reports = {}
    for seed in [11,22,33]:
        path = Path(f'tests/results/fresh_rollout_sampling/{seed}/summary.json')
        report = json.loads(path.read_text())
        assert report['status'] == 'COMPLETE_FRESH_ROLLOUT_SAMPLING_NOT_CERTIFIED'
        assert report['manifest_sha256'] == ref['manifest_sha256']
        assert len(report['rows']) == 24
        assert {(r['anchor'],r['steps']) for r in report['rows']} == {(a,s) for a in range(12) for s in [32,64]}
        reports[seed] = report, digest(path)
    root = Path('tests/results/fresh_model_audit')
    root.mkdir(exist_ok=False)
    result = dict(status='RUNNING', rows=[], replicates=2000,
        numerical_audit_sha256=digest(numerical_path), reference_sha256=digest(ref_path),
        code_sha256={p:digest(p) for p in [__file__, 'tests/experiments/director_budget_analysis.py',
            'tests/experiments/director_uncertainty.py', 'tests/experiments/director_convergence_summary.py',
            'tests/experiments/fresh_reference_audit.py']},
        scope='Fresh fixed initial-state stress test, not equilibrium or large-size certification. '
        'Midpoint32 primary, midpoint64 paired numerical control. Independent reference whole-path '
        'resampling. Pointwise intervals, no equivalence decision. Full temporal covariance is '
        'site-averaged same-site dot covariance; selected joint covariance retains all sites and '
        'components at five times. Energy split repeats are descriptive, not independent replicates. '
        'Numerical audit completion is not assumed to imply reference convergence.')
    for seed,(report,source_hash) in reports.items():
        for anchor in range(12):
            rows = [next(r for r in report['rows'] if r['anchor']==anchor and r['steps']==s) for s in [32,64]]
            assert rows[0]['latent_seed'] == rows[1]['latent_seed']
            data, states = [], []
            for r in [*rows,references[anchor]]:
                assert digest(r['path']) == r['sha256']
                with np.load(r['path']) as z:
                    p = z['spins'].astype(float)
                    t = z['reduced_time']
                    if 'rng_state' in z: states.append(z['rng_state'])
                assert p.shape == (256,81,25,3) and np.isfinite(p).all()
                assert np.max(abs(np.linalg.norm(p,axis=-1)-1)) < 1e-5
                np.testing.assert_allclose(t,np.arange(81)*.01,atol=1e-12,rtol=0)
                if data: np.testing.assert_allclose(p[:,0],data[0][:,0],atol=2e-7,rtol=0)
                data.append(p)
            np.testing.assert_array_equal(*states)
            rng_seed = 995000000+seed*1000+anchor
            rng = np.random.default_rng(rng_seed)
            arrays,metrics = sampler_statistics(data,rng)
            arrays['time'] = t[1:]
            arrays['paired_path_rmse'] = np.sqrt(np.mean((data[1]-data[0])**2,axis=(0,2,3)))
            for label,p in zip(['32','64','reference'],data):
                arrays[label+'_full_covariance'] = full_covariance(p[:,np.r_[0,FRAMES]])
                arrays[label+'_temporal_covariance'] = temporal_covariance(p)
                arrays[label+'_selected_joint_covariance'] = np.cov(p[:,FRAMES].reshape(256,-1),rowvar=False)
            selected = [p[:,FRAMES] for p in data]
            energy = {label:path_energy(p,selected[2]) for label,p in zip(['32','64'],selected[:2])}
            baseline, model32, model64 = [], [], []
            for _ in range(20):
                ri,gi = rng.permutation(256),rng.permutation(256)
                a,b = selected[2][ri[:128]],selected[2][ri[128:]]
                baseline.append(path_energy(a,b))
                model32.append(path_energy(selected[0][gi[:128]],a))
                model64.append(path_energy(selected[1][gi[:128]],a))
            arrays.update(reference_split_energy=np.array(baseline), model32_split_energy=np.array(model32), model64_split_energy=np.array(model64))
            assert all(np.isfinite(v).all() for v in arrays.values())
            path = root/f'seed{seed}_anchor{anchor}.npz'
            np.savez_compressed(path,**arrays)
            result['rows'].append(dict(seed=seed,anchor=anchor,state=rows[0]['state'],path=str(path),
                sha256=digest(path),source_sha256=source_hash,rng_seed=rng_seed,metrics=metrics,joint_energy=energy))
            save(root/'summary.json',result)
    result['status'] = 'COMPLETE_FRESH_MODEL_AUDIT_NOT_CERTIFIED'
    save(root/'summary.json',result)


if __name__ == '__main__':
    run()
