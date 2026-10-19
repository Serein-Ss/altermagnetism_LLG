"""Full-coverage descriptive integration refinement summary, not certification."""
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.local_transition_baselines import profiles, compare


def full_covariance(paths):
    p = np.asarray(paths, dtype=float)
    centered = p[:, 1:]-p[:, 1:].mean(0)
    flat = centered.reshape(len(p), p.shape[1]-1, -1).transpose(1, 0, 2)
    return flat.transpose(0, 2, 1) @ flat/(len(p)-1)


def run():
    reports = {}
    for seed in [11,22,33]:
        source = Path(f'tests/results/director_sampling_convergence/{seed}/summary.json')
        report = json.loads(source.read_text())
        if report['status'] != 'COMPLETE_SAMPLING_REFINEMENT_NOT_CERTIFIED':
            raise ValueError(f'incomplete seed {seed}; do not summarize partial results')
        assert len(report['rows']) == 48
        assert {(r['anchor'],r['steps']) for r in report['rows']} == {(a,s) for a in range(12) for s in [32,64,128,256]}
        reports[seed] = report, digest(source)
    root = Path('tests/results/director_convergence_summary')
    root.mkdir(exist_ok=False)
    result = dict(status='RUNNING',rows=[],code_sha256=digest(__file__),
        scope='Descriptive paired integration differences, no formal error tolerance or distribution certification. '
        'Full covariance reference errors include finite-sample noise. Raw paired-path errors apply only between sampler resolutions, not model versus LLG noise.')
    for seed,(report,source_hash) in reports.items():
        for anchor in range(12):
            rows = {r['steps']:r for r in report['rows'] if r['anchor']==anchor}
            reference_path=rows[64]['reference_path']
            assert digest(reference_path)==rows[64]['reference_sha256']
            with np.load(reference_path) as d:
                reference=d['spins'][:,::4]; times=d['reduced_time'][::4]
            rp=profiles(reference); rc=full_covariance(reference)
            paths={}; statistics={}; covariances={}; metrics={}
            for steps,row in rows.items():
                assert row['reference_path']==reference_path
                assert row['latent_seed']==rows[64]['latent_seed']
                assert digest(row['path'])==row['sha256']
                with np.load(row['path']) as d:
                    p=d['spins'].astype(float)
                    np.testing.assert_allclose(d['reduced_time'],times,atol=1e-12,rtol=0)
                assert p.shape==reference.shape==(256,9,25,3) and np.isfinite(p).all()
                paths[steps]=p; statistics[steps]=profiles(p); covariances[steps]=full_covariance(p)
                metrics[str(steps)]=dict(errors=compare(statistics[steps],rp),
                    variance_ratio=(statistics[steps]['site_variance'].mean(-1)[1:]/rp['site_variance'].mean(-1)[1:]).tolist(),
                    full_covariance_rmse=np.sqrt(np.mean((covariances[steps]-rc)**2,axis=(1,2))).tolist())
            changes={}
            for low,high in [(32,64),(64,128),(128,256),(64,256)]:
                changes[f'{low}_to_{high}']=dict(
                    paired_path_rmse=np.sqrt(np.mean((paths[high]-paths[low])**2,axis=(0,2,3))).tolist(),
                    variance_change=(statistics[high]['site_variance']-statistics[low]['site_variance']).mean(-1).tolist(),
                    full_covariance_change_rmse=np.sqrt(np.mean((covariances[high]-covariances[low])**2,axis=(1,2))).tolist())
            result['rows'].append(dict(seed=seed,anchor=anchor,split=rows[64]['split'],state=rows[64]['state'],
                source_sha256=source_hash,metrics=metrics,changes=changes))
            save(root/'summary.json',result)
    result['status']='COMPLETE_DESCRIPTIVE_CONVERGENCE_NOT_CERTIFIED'
    save(root/'summary.json',result)


if __name__=='__main__':
    run()
