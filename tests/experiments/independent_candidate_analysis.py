"""Independent-path comparisons for all frozen candidates and numerical controls."""
import json
from pathlib import Path

import numpy as np

from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.director_uncertainty import weighted_profiles, differences
from tests.experiments.fresh_reference_audit import FRAMES, path_energy
from tests.experiments.fresh_model_audit import temporal_covariance


def compare(candidate, reference, rng, replicates=2000):
    profiles, draws = [], []
    for p in [candidate, reference]:
        n = len(p)
        profiles.append(weighted_profiles(p, np.ones((1, n))))
        weights = rng.multinomial(n, np.full(n, 1/n), size=replicates)
        draws.append(weighted_profiles(p, weights))
    point = differences(*profiles)
    bootstrap = differences(*draws)
    metrics = {key: dict(estimate=value[0].tolist(),
        ci95=np.quantile(bootstrap[key], [.025, .975], axis=0).tolist())
        for key, value in point.items()}
    return bootstrap, metrics


def load_paths(item):
    assert digest(item['path']) == item['sha256']
    with np.load(item['path']) as z:
        paths = z['spins'].astype(float)
        times = z['reduced_time']
    assert paths.shape == (512, 81, 25, 3) and np.isfinite(paths).all()
    np.testing.assert_allclose(times, np.arange(81)*.01, atol=1e-12, rtol=0)
    assert np.max(abs(np.linalg.norm(paths, axis=-1)-1)) < 1e-5
    return paths


def run():
    rp = Path('tests/results/independent_candidate_reference/summary.json')
    reference_report = json.loads(rp.read_text())
    assert reference_report['status'] == 'COMPLETE_INDEPENDENT_CANDIDATE_REFERENCE_NOT_CERTIFIED'
    assert {(r['case'], r['anchor']) for r in reference_report['rows']} == {(c,a) for c in range(3) for a in range(24)}
    manifest_path = rp.parent/'manifest.json'
    assert digest(manifest_path) == reference_report['manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    reports, sources = {}, {str(rp): digest(rp)}
    for seed in [11, 22, 33]:
        p = Path(f'tests/results/independent_candidate_sampling/{seed}/summary.json')
        report = json.loads(p.read_text())
        assert report['status'] == 'COMPLETE_INDEPENDENT_CANDIDATE_SAMPLING_NOT_CERTIFIED'
        assert report['manifest_sha256'] == digest(manifest_path)
        assert len(report['rows']) == 24 and {r['anchor'] for r in report['rows']} == set(range(24))
        reports[seed] = report
        sources[str(p)] = digest(p)
    root = Path('tests/results/independent_candidate_analysis')
    root.mkdir(exist_ok=False)
    result = dict(status='RUNNING', rows=[], sources=sources, replicates=2000,
        manifest_sha256=digest(manifest_path),
        code_sha256={p:digest(p) for p in [__file__, 'tests/experiments/director_uncertainty.py',
            'tests/experiments/fresh_reference_audit.py', 'tests/experiments/fresh_model_audit.py']},
        scope='All 24 conditions: three frozen model seeds and two numerical controls versus fine MT. '
        'Independent whole-path bootstrap, pointwise intervals only; no pooled seed/family test or equivalence gate. '
        'Selected joint covariance/energy use five fixed times; temporal covariance averages same-site dot products. '
        'No full continuous-path, large-size or long-time certification.')
    for anchor in range(24):
        ref_item = next(r for r in reference_report['rows'] if r['case']==1 and r['anchor']==anchor)
        reference = load_paths(ref_item)
        candidates = [(f'reference{c}', next(r for r in reference_report['rows'] if r['case']==c and r['anchor']==anchor)) for c in [0,2]]
        candidates += [(f'model{s}', next(r for r in reports[s]['rows'] if r['anchor']==anchor)) for s in [11,22,33]]
        for index, (label, item) in enumerate(candidates):
            candidate = load_paths(item)
            np.testing.assert_allclose(candidate[:,0], reference[:,0], atol=2e-7, rtol=0)
            rng_seed = 1007000000+anchor*10+index
            arrays, metrics = compare(candidate, reference, np.random.default_rng(rng_seed))
            arrays['time'] = np.arange(1,81)*.01
            for name, paths in [('candidate',candidate),('reference',reference)]:
                arrays[name+'_selected_covariance'] = np.cov(paths[:,FRAMES].reshape(512,-1), rowvar=False)
                arrays[name+'_temporal_covariance'] = temporal_covariance(paths)
            assert all(np.isfinite(x).all() for x in arrays.values())
            p = root/f'{label}_anchor{anchor}.npz'
            np.savez_compressed(p, **arrays)
            result['rows'].append(dict(label=label, anchor=anchor, state=manifest['anchors'][anchor],
                candidate=item, reference=ref_item, path=str(p), sha256=digest(p), rng_seed=rng_seed,
                metrics=metrics, joint_energy=path_energy(candidate[:,FRAMES],reference[:,FRAMES])))
            save(root/'summary.json',result)
    assert len(result['rows']) == 120
    result['status'] = 'COMPLETE_INDEPENDENT_CANDIDATE_ANALYSIS_NOT_CERTIFIED'
    save(root/'summary.json',result)


if __name__ == '__main__':
    run()
