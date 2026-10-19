"""Descriptive localization of conditional mean error; no new model fitting."""
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest, save


def spatial_error(generated, reference):
    g = generated-generated[:, :1]
    r = reference-reference[:, :1]
    difference = g.mean(0)-r.mean(0)
    raw = (difference**2).mean(-1)
    noise = (g.var(0, ddof=1)/len(g)+r.var(0, ddof=1)/len(r)).mean(-1)
    return difference[1:], raw[1:], noise[1:], (raw-noise)[1:]


def run():
    root = Path('tests/results/midpoint_spatial_mean')
    root.mkdir(exist_ok=False)
    result = dict(status='RUNNING', rows=[], code_sha256=digest(__file__),
        scope='Exploratory fixed states, midpoint32. Per-site squared vector mean error '
        'minus independent-ensemble sample-mean noise. Signed corrections retained. '
        'Boundary means first/last three sites, interior remaining 19, defined before analysis '
        'from network radius. No independent-site inference or causal coverage claims. '
        'Train reused data and development reused in model design; not certification.')
    for seed in [11, 22, 33]:
        source = Path(f'tests/results/director_midpoint_sampling/{seed}/summary.json')
        report = json.loads(source.read_text())
        assert report['status'] == 'COMPLETE_MIDPOINT_REFINEMENT_NOT_CERTIFIED'
        rows = [r for r in report['rows'] if r['steps'] == 32]
        assert len(rows) == 12 and {r['anchor'] for r in rows} == set(range(12))
        for row in rows:
            data = []
            for path, sha, stride in [(row['path'], row['sha256'], 1),
                    (row['reference_path'], row['reference_sha256'], 4)]:
                assert digest(path) == sha
                with np.load(path) as z:
                    p = z['spins'][:, ::stride].astype(float)
                    t = z['reduced_time'][::stride]
                assert p.shape == (256, 9, 25, 3) and np.isfinite(p).all()
                if data:
                    np.testing.assert_allclose(t, times, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(p[:, 0], data[0][:, 0], atol=2e-7, rtol=0)
                times = t
                data.append(p)
            difference, raw, noise, corrected = spatial_error(*data)
            initial = data[1][0, 0]
            bond = np.sum(initial[:-1]*initial[1:], axis=-1)
            path = root/f"seed{seed}_anchor{row['anchor']}.npz"
            np.savez_compressed(path, time=times[1:], mean_difference=difference,
                raw_mse=raw, estimated_mean_noise=noise, corrected_mse=corrected,
                initial=initial, initial_bond=bond)
            boundary = np.r_[0:3, 22:25]
            result['rows'].append(dict(seed=seed, anchor=row['anchor'], split=row['split'],
                state=row['state'], path=str(path), sha256=digest(path),
                source_report_sha256=digest(source),
                raw_mse=raw.mean(-1).tolist(), noise=noise.mean(-1).tolist(),
                boundary_corrected_mse=corrected[:, boundary].mean(-1).tolist(),
                interior_corrected_mse=corrected[:, 3:22].mean(-1).tolist(),
                corrected_mse_per_time_squared=(corrected.mean(-1)/times[1:]**2).tolist()))
            save(root/'summary.json', result)
    result['status'] = 'COMPLETE_SPATIAL_MEAN_LOCALIZATION_NOT_CERTIFIED'
    save(root/'summary.json', result)


if __name__ == '__main__':
    run()
