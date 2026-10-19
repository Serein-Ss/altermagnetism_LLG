"""Per-parent descriptive path comparisons, never a universal distribution proof."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


def observables(s):
    z = s[..., 2]
    mz = z.mean(-1)
    bond = (s[:, :, 1:]*s[:, :, :-1]).sum(-1).mean(-1)
    energy = -bond*(s.shape[2]-1)/s.shape[2]-.1*(z*z).mean(-1)
    temporal = (s*s[:, :1]).sum(-1).mean(-1)
    spatial = np.stack([(z[:, :, :-lag]*z[:, :, lag:]).mean(-1) for lag in [1, 2, 4]], -1)
    delta = s[:, 1:]-s[:, :-1]
    mean = delta.mean(0)
    centered = delta-mean
    covariance = np.einsum('btni,btnj->tnij', centered, centered)/len(s)
    walls = (z[:, :, 1:]*z[:, :, :-1] < 0).mean(-1)
    features = np.concatenate((mz[:, 1:], energy[:, 1:], temporal[:, 1:], spatial[:, 1:].reshape(len(s), -1)), -1)
    features = features/np.sqrt(features.shape[-1])
    return dict(mz=mz, energy=energy, temporal=temporal, spatial=spatial,
                delta_mean=mean, delta_cov=covariance, walls=walls, features=features)


def compare(a, b):
    x, y = observables(a), observables(b)
    rmse = lambda p, q: float(np.sqrt(np.mean((p-q)**2)))
    f, g = x['features'], y['features']
    return dict(mz_mean=rmse(x['mz'].mean(0), y['mz'].mean(0)),
                mz_var=rmse(x['mz'].var(0), y['mz'].var(0)),
                energy_mean=rmse(x['energy'].mean(0), y['energy'].mean(0)),
                spatial_corr=rmse(x['spatial'].mean(0), y['spatial'].mean(0)),
                temporal_corr=rmse(x['temporal'].mean(0), y['temporal'].mean(0)),
                increment_mean=rmse(x['delta_mean'], y['delta_mean']),
                increment_cov=rmse(x['delta_cov'], y['delta_cov']),
                endpoint_w1=float(wasserstein_distance(x['mz'][:, -1], y['mz'][:, -1])),
                wall_proxy=rmse(x['walls'].mean(0), y['walls'].mean(0)),
                path_energy_distance=float(max(0., 2*cdist(f, g).mean()-cdist(f, f).mean()-cdist(g, g).mean())))


def summary(rows, margins, repeats=1000):
    rng = np.random.default_rng(20260919)
    result = {}
    # Stratified parent bootstrap: preserve the balanced L/family design.
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r['L'], r['family']), []).append(i)
    indices = np.concatenate([rng.choice(ix, (repeats, len(ix)), replace=True) for ix in groups.values()], axis=1)
    for key, margin in margins.items():
        a = np.asarray([r[key] for r in rows])
        lo, hi = np.quantile(a[indices].mean(1), [.025, .975])
        status = 'WITHIN_PILOT_MARGIN' if hi <= margin else ('OUTSIDE_PILOT_MARGIN' if lo > margin else 'INCONCLUSIVE')
        result[key] = dict(mean=float(a.mean()), parent_bootstrap_95=[float(lo), float(hi)], margin=margin, status=status)
    return result
