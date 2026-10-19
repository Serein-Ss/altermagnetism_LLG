"""Same-state, equal-branch descriptive diagnostics; no acceptance claim."""
import json
from pathlib import Path
import h5py
import numpy as np
from tests.bauer_publication_20260920.audit import mean_bias_squared

HERE = Path(__file__).resolve().parent


def moments(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 4 or a.shape[1] != 2 or a.shape[-1] != 3:
        raise ValueError('Require equal branch counts and two frames')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite input')
    if not np.allclose(a[:, 0], b[0, 0], atol=2e-7) or not np.allclose(b[:, 0], b[0, 0], atol=2e-7):
        raise ValueError('Initial states do not match')
    da, db = a[:, 1]-a[:, 0], b[:, 1]-b[:, 0]
    ca, cb = da-da.mean(0), db-db.mean(0)
    va = np.einsum('bni,bnj->nij', ca, ca)/(len(a)-1)
    vb = np.einsum('bni,bnj->nij', cb, cb)/(len(b)-1)
    scale = max(float(np.mean(np.sum(db*db, axis=-1))), 1e-12)
    return dict(mean_increment=mean_bias_squared(da, db),
                signed_mz_increment=float(da[..., 2].mean()-db[..., 2].mean()),
                signed_power_relative=float(np.mean(np.sum(da*da, axis=-1))/scale-1),
                local_covariance_relative=float(np.sqrt(np.mean(np.sum((va-vb)**2, axis=(1, 2))))/scale))


def main():
    spec = json.loads((HERE/'data/specification.json').read_text()); rows = []
    with h5py.File(HERE/'results/reference.h5', 'r') as ref:
        assert ref.attrs['complete'] and len(ref) == spec['conditions']
        for seed in spec['model_seeds']:
            with h5py.File(HERE/'results'/f'model_{seed}.h5', 'r') as model:
                assert model.attrs['complete'] and set(model) == set(ref)
                for state in spec['rows']:
                    a, b = model[state['key']][:], ref[state['key']][:]
                    assert len(a) == 64 and len(b) == 128
                    rows.append(dict(seed=seed, L=state['L'], parent=state['parent'], time_ps=state['time_ps'],
                                     model_ref_a=moments(a, b[:64]), model_ref_b=moments(a, b[64:]),
                                     ref_b_ref_a=moments(b[64:], b[:64])))
    result = dict(rows=rows, scientific_certification=False,
                  interpretation='Exploratory, 64 versus 64 branches. Negative corrected mean-square estimates retained. '
                  'Local covariance covers spin components at one site, not cross-site covariance. '
                  'Two model comparisons share generated samples; times share parent trajectories. No p-values or equivalence claim.')
    (HERE/'results/moments.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for seed in spec['model_seeds']:
        selected = [r for r in rows if r['seed'] == seed]
        print(seed, 'corrected_mean_mse/model,reference',
              np.mean([r[g]['mean_increment']['corrected_mse'] for r in selected for g in ['model_ref_a', 'model_ref_b']]),
              np.mean([r['ref_b_ref_a']['mean_increment']['corrected_mse'] for r in selected]))


if __name__ == '__main__':
    main()
