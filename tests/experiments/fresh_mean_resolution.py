"""Locate fresh-rollout mean bias without conflating it with ensemble noise."""
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.midpoint_mean_resolution import corrected_mean_mse


def load(row):
    assert digest(row['path']) == row['sha256']
    with np.load(row['path']) as z:
        p = z['spins'].astype(float)
        np.testing.assert_allclose(z['reduced_time'],np.arange(81)*.01,atol=1e-12,rtol=0)
    assert p.shape == (256,81,25,3) and np.isfinite(p).all()
    return p


def run():
    source = Path('tests/results/fresh_model_audit/summary.json')
    audit = json.loads(source.read_text())
    verified = json.loads(source.with_name('verified_aggregate.json').read_text())
    assert verified['source_sha256'] == digest(source)
    assert verified['status'] == 'VERIFIED_RAW_POINTS_AND_STORED_INTERVALS_NOT_CERTIFIED'
    ref_path = Path('tests/results/fresh_rollout_reference/summary.json')
    assert digest(ref_path) == audit['reference_sha256']
    refs = {(r['case'],r['anchor']):r for r in json.loads(ref_path.read_text())['rows']}
    root = Path('tests/results/fresh_mean_resolution')
    root.mkdir(exist_ok=False)
    rows = []
    for anchor in range(12):
        reference = load(refs[1,anchor])
        sources = [('mt_coarse',refs[0,anchor]),('heun_fine',refs[2,anchor])]
        for seed in [11,22,33]:
            report = Path(f'tests/results/fresh_rollout_sampling/{seed}/summary.json')
            row = next(r for r in audit['rows'] if r['seed']==seed and r['anchor']==anchor)
            assert digest(report) == row['source_sha256']
            sources.append((f'model{seed}',next(r for r in json.loads(report.read_text())['rows'] if r['anchor']==anchor and r['steps']==32)))
        for label,row in sources:
            candidate = load(row)
            arrays = dict(time=np.arange(1,81)*.01)
            for region,index in [('all',np.arange(25)),('boundary',np.r_[0:3,22:25]),('interior',np.arange(3,22))]:
                raw,noise,corrected = corrected_mean_mse(candidate[:,:,index],reference[:,:,index])
                arrays.update({region+'_raw_mse':raw,region+'_mean_noise':noise,
                    region+'_corrected_mse':corrected,region+'_raw_noise_ratio':raw/noise})
            path = root/f'{label}_anchor{anchor}.npz'
            np.savez_compressed(path,**arrays)
            rows.append(dict(label=label,anchor=anchor,state=next(r['state'] for r in audit['rows'] if r['anchor']==anchor),
                path=str(path),sha256=digest(path),source_sha256=row['sha256'],reference_sha256=refs[1,anchor]['sha256'],
                selected={k:v[np.array([0,7,19,39,79])].tolist() for k,v in arrays.items()}))
    save(root/'summary.json',dict(status='COMPLETE_EXPLORATORY_MEAN_RESOLUTION_NOT_CERTIFIED',
        source_sha256=digest(source),code_sha256={p:digest(p) for p in [__file__,'tests/experiments/midpoint_mean_resolution.py']},rows=rows,
        scope='Fixed-state independent path ensembles. Subtract sample-mean variance from squared mean error; negative corrected estimates retained. No clipping, significance test, equivalence claim or independence across times/seeds. Boundary first/last three sites reflects model radius, not a fitted partition. Reference integrator comparisons include independent sampling noise.'))
    for kind in ['aligned','tilted','wall']:
        for label in ['model11','model22','model33','mt_coarse','heun_fine']:
            group=[r for r in rows if r['state']['kind']==kind and r['label']==label]
            ratios=np.array([r['selected']['all_raw_noise_ratio'] for r in group])
            print(kind,label,'mean raw/noise at .01,.08,.2,.4,.8:',np.round(ratios.mean(0),3))


if __name__ == '__main__':
    run()
