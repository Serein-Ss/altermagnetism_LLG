"""Independent-noise numerical reference comparisons; not continuum certification."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.director_uncertainty import weighted_profiles, differences
from tests.experiments.director_convergence_summary import full_covariance

FRAMES = np.array([1, 8, 20, 40, 80])


def path_energy(a, b):
    """Unbiased energy discrepancy for the supplied joint path vectors; may be negative."""
    x = np.asarray(a, dtype=float).reshape(len(a), -1)
    y = np.asarray(b, dtype=float).reshape(len(b), -1)
    if min(len(x), len(y)) < 2 or x.shape[1] != y.shape[1]:
        raise ValueError('at least two vectors per ensemble, with matching dimension')
    return float((2*cdist(x, y).mean()-pdist(x).mean()-pdist(y).mean())/np.sqrt(x.shape[1]))


def run():
    source = Path('tests/results/fresh_rollout_reference/summary.json')
    report = json.loads(source.read_text())
    assert report['status'] == 'COMPLETE_FRESH_ROLLOUT_REFERENCE_NOT_CERTIFIED'
    assert len(report['rows']) == 36
    assert {(r['case'], r['anchor']) for r in report['rows']} == {(c,a) for c in range(3) for a in range(12)}
    manifest_path = source.parent/'manifest.json'
    assert digest(manifest_path) == report['manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    root = Path('tests/results/fresh_reference_audit')
    root.mkdir(exist_ok=False)
    result = dict(status='RUNNING', rows=[], source_sha256=digest(source), replicates=2000,
        frames=FRAMES.tolist(), code_sha256={p: digest(p) for p in [__file__,
            'tests/experiments/director_uncertainty.py', 'tests/experiments/director_convergence_summary.py']},
        scope='All 12 fixed initial-temperature conditions, independent-noise MT coarse and Heun '
        'versus MT dt.00125. Whole-path pointwise percentile intervals, no equivalence decision. '
        'Energy discrepancy is on five selected joint times, not full continuous path law; '
        '20 matched128-path partitions provide descriptive reference noise context, not CIs. '
        'First adjacent-time connected covariance is structurally zero.')
    for anchor in range(12):
        data, hashes, seeds = {}, {}, {}
        for row in [r for r in report['rows'] if r['anchor'] == anchor]:
            assert digest(row['path']) == row['sha256']
            with np.load(row['path']) as z:
                p = z['spins'].astype(float)
                t = z['reduced_time']
                noise = z['seeds']
            assert p.shape == (256,81,25,3) and np.isfinite(p).all()
            assert np.max(abs(np.linalg.norm(p, axis=-1)-1)) < 1e-10
            np.testing.assert_allclose(t, np.arange(81)*.01, atol=1e-12, rtol=0)
            assert len(np.unique(noise)) == 256
            for previous in seeds.values():
                assert not np.intersect1d(previous, noise).size
            if data:
                np.testing.assert_array_equal(p[:,0], next(iter(data.values()))[:,0])
            data[row['case']], hashes[row['case']], seeds[row['case']] = p, row['sha256'], noise
        rng_seed = 994000000+anchor
        rng = np.random.default_rng(rng_seed)
        reference = data[1]
        rb = weighted_profiles(reference, rng.multinomial(256, np.full(256,1/256), size=2000))
        rp = weighted_profiles(reference, np.ones((1,256)))
        rc = full_covariance(reference[:, np.r_[0,FRAMES]])
        for case in [0,2]:
            p = data[case]
            pb = weighted_profiles(p, rng.multinomial(256, np.full(256,1/256), size=2000))
            draws = differences(pb, rb)
            estimates = differences(weighted_profiles(p,np.ones((1,256))), rp)
            arrays = dict(time=t[1:], **draws)
            arrays['full_covariance_rmse'] = np.sqrt(np.mean((full_covariance(p[:,np.r_[0,FRAMES]])-rc)**2,axis=(1,2)))
            selected, ref_selected = p[:,FRAMES], reference[:,FRAMES]
            split_energy, cross_energy = [], []
            for _ in range(20):
                ri, pi = rng.permutation(256), rng.permutation(256)
                split_energy.append(path_energy(ref_selected[ri[:128]],ref_selected[ri[128:]]))
                cross_energy.append(path_energy(selected[pi[:128]],ref_selected[ri[:128]]))
            arrays.update(reference_split_energy=np.array(split_energy), cross_energy=np.array(cross_energy))
            assert all(np.isfinite(v).all() for v in arrays.values())
            path = root/f'case{case}_anchor{anchor}.npz'
            np.savez_compressed(path, **arrays)
            metrics = {k: dict(estimate=v[0].tolist(), ci95=np.quantile(draws[k],[.025,.975],axis=0).tolist()) for k,v in estimates.items()}
            result['rows'].append(dict(case=case, anchor=anchor, state=manifest['anchors'][anchor],
                path=str(path), sha256=digest(path), source_sha256=hashes, rng_seed=rng_seed,
                metrics=metrics, joint_energy=path_energy(selected,ref_selected)))
            save(root/'summary.json',result)
    result['status'] = 'COMPLETE_FRESH_REFERENCE_AUDIT_NOT_CERTIFIED'
    save(root/'summary.json', result)


if __name__ == '__main__':
    run()
