"""Exploratory diagnosis of frozen two-stage outputs, not a new acceptance gate."""
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import wasserstein_distance

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'bauer_two_stage_20260919'


def mean_bias_squared(a, b):
    """Unbiased squared difference of conditional means; axis 0 is independent branches.

    Remaining sites/times/components are averaged, never treated as replicates.
    A negative estimate is retained: sampling variation, not negative physical bias.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if min(len(a), len(b)) < 2:
        raise ValueError('Need at least two independent branches in each group')
    raw = np.mean((a.mean(0) - b.mean(0)) ** 2)
    noise = np.mean(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b))
    return dict(raw_mse=float(raw), sampling_variance=float(noise), corrected_mse=float(raw-noise))


def phase_statistics(x, block_frames=10):
    """Increments at phase 0 start a block; pair phase 9 straddles a seam."""
    d = np.diff(np.asarray(x, dtype=float), axis=1)
    power = np.mean(np.sum(d*d, axis=-1), axis=(0, 2))
    adjacent = np.mean(np.sum(d[:, :-1]*d[:, 1:], axis=-1), axis=(0, 2))
    def phases(values):
        return [float(values[k::block_frames].mean()) if len(values[k::block_frames]) else None
                for k in range(block_frames)]
    return dict(increment_power=phases(power), adjacent_increment_dot=phases(adjacent))


def initial_features(x):
    z = x[..., 2]
    bond = np.sum(x[:, :-1]*x[:, 1:], axis=-1)
    return dict(min_sz=z.min(1), min_bond=bond.min(1), mean_sz=z.mean(1),
                fraction_sz_below_0=(z < 0).mean(1), fraction_sz_below_half=(z < .5).mean(1),
                fraction_bond_below_half=(bond < .5).mean(1),
                edge_sz=z[:, [0, -1]].mean(1), interior_sz=z[:, 1:-1].mean(1))


def bootstrap_parents(values):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260920)
    means = a[rng.integers(len(a), size=(2000, len(a)))].mean(1)
    return dict(parents=len(a), mean=float(a.mean()),
                exploratory_parent_bootstrap_95=np.quantile(means, [.025, .975]).tolist())


def main():
    rows, initials, summaries = [], [], []
    for length in [16, 25, 32, 50, 100]:
        with h5py.File(SOURCE/'data'/f'L{length}'/'test_initial.h5') as h:
            real = initial_features(h['spins'][:].astype(float))
        with h5py.File(SOURCE/'generated'/f'L{length}.h5') as h:
            assert h.attrs['complete']
            fake = initial_features(h['initial'][:].astype(float))
            for name, a in real.items():
                b = fake[name]
                # Equal-n disjoint reference-reference comparisons. Repeated partitions
                # describe a finite-sample floor, NOT independent experiments or a p value.
                rng = np.random.default_rng(20260920 + length)
                rr, gr = [], []
                for _ in range(200):
                    r, g = rng.permutation(a), rng.permutation(b)
                    rr.append(wasserstein_distance(r[:32], r[32:64]))
                    gr.append(wasserstein_distance(g[:32], r[:32]))
                initials.append(dict(L=length, feature=name, reference_n=len(a), generated_n=len(b),
                    reference_mean=float(a.mean()), generated_mean=float(b.mean()),
                    signed_difference=float(b.mean()-a.mean()), reference_sd=float(a.std(ddof=1)),
                    generated_sd=float(b.std(ddof=1)), reference_quantiles=np.quantile(a,[0,.25,.5,.75,1]).tolist(),
                    generated_quantiles=np.quantile(b,[0,.25,.5,.75,1]).tolist(),
                    w1_equal32_rr_quantiles=np.quantile(rr,[.05,.5,.95]).tolist(),
                    w1_equal32_gr_quantiles=np.quantile(gr,[.05,.5,.95]).tolist()))
            for parent in range(6):
                with h5py.File(SOURCE/'data'/f'L{length}'/f'test_future_{parent}.h5') as ref:
                    r = ref['spins'][:].astype(float)
                    times = ref['time_ps'][:]
                g = h[f'RG_{parent}'][:].astype(float)
                assert np.allclose(times, h['time_ps'][:])
                assert np.allclose(r[:, 0], g[0, 0], atol=2e-7)
                assert np.isfinite(r).all() and np.isfinite(g).all()
                for end in [11, 101, 1001]:
                    dr, dg = np.diff(r[:, :end], axis=1), np.diff(g[:, :end], axis=1)
                    rows.append(dict(L=length, parent=parent, horizon_ps=float(times[end-1]),
                                     model_vs_reference=mean_bias_squared(dg, dr),
                                     reference_vs_reference=mean_bias_squared(dr[:16], dr[16:]),
                                     reference_branches=len(r), model_branches=len(g)))
                rows.append(dict(L=length, parent=parent, phase_model=phase_statistics(g),
                                 phase_reference=phase_statistics(r)))
        for horizon in [1., 10., 100.]:
            selected = [v for v in rows if v['L']==length and v.get('horizon_ps')==horizon]
            summaries.append(dict(L=length, horizon_ps=horizon,
                corrected_mse=bootstrap_parents([v['model_vs_reference']['corrected_mse'] for v in selected]),
                reference_corrected_mse=bootstrap_parents([v['reference_vs_reference']['corrected_mse'] for v in selected])))
    result = dict(exploratory=True, scientific_certification=False, initial=initials,
                  conditional=rows, summary=summaries,
                  source=str(SOURCE), numpy_version=np.__version__,
                  cautions=['Existing test outputs are now diagnostic/development evidence; use fresh final holdout.',
                            'Six parent states per size; percentile bootstrap is descriptive and not simultaneous.',
                            'No hypothesis tests, no new pass/fail thresholds, no proof of equivalence.',
                            'Tail cutoffs 0 and 0.5 are diagnostic bins, not completed-switch definitions.',
                            'Phase 0 increment starts a 1ps block; adjacent-dot phase 9 crosses a block seam.'])
    out = HERE/'results'; out.mkdir(exist_ok=True)
    (out/'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for row in summaries:
        print(row, flush=True)


if __name__ == '__main__':
    main()
