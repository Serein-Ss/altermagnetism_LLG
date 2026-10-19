"""Time-resolved open-chain observables for ONE exact initial/condition family.

No stationarity assumption, significance threshold, or distribution certificate.
The caller must keep physical conditions and time grids matched. Ensemble moments
use population normalization; frames/sites are not independent replicates.
"""
import numpy as np


def path_observables(paths, temporal_lags=(1, 2, 4, 8, 16, 32), spatial_lags=(1, 2, 4, 8, 12)):
    s = np.asarray(paths, dtype=np.float64)
    if s.ndim != 4 or s.shape[-1] != 3 or min(s.shape[:3]) < 2:
        raise ValueError('expected at least two paths, frames and sites: [B,T,L,3]')
    if not np.isfinite(s).all() or np.max(np.abs(np.linalg.norm(s, axis=-1)-1)) > 1e-5:
        raise ValueError('finite unit spins required')
    if not np.all(s[:, 0] == s[0, 0]):
        raise ValueError('do not pool distinct initial families')
    temporal_lags = tuple(temporal_lags); spatial_lags = tuple(spatial_lags)
    if any(not isinstance(k, (int, np.integer)) or k <= 0 or k >= s.shape[1] for k in temporal_lags):
        raise ValueError('invalid temporal lag')
    if any(not isinstance(k, (int, np.integer)) or k <= 0 or k >= s.shape[2] for k in spatial_lags):
        raise ValueError('invalid spatial lag')
    mean = s.mean(axis=0)
    centered = s-mean[None]
    magnetization = s.mean(axis=2)
    result = dict(magnetization_per_path=magnetization,
                  magnetization_mean=magnetization.mean(axis=0),
                  magnetization_variance=magnetization.var(axis=0),
                  spin_ensemble_variance=np.square(centered).sum(-1).mean(axis=(0, 2)),
                  spatial_spin_variance=np.square(s-magnetization[:, :, None]).sum(-1).mean(axis=(0, 2)))
    for lag in temporal_lags:
        # Preserve start time: no averaging over a possibly nonstationary path.
        result[f'temporal_dot_lag{lag}'] = (s[:, :-lag]*s[:, lag:]).sum(-1).mean(axis=(0, 2))
        result[f'temporal_connected_lag{lag}'] = (centered[:, :-lag]*centered[:, lag:]).sum(-1).mean(axis=(0, 2))
        result[f'increment_second_moment_lag{lag}'] = np.square(s[:, lag:]-s[:, :-lag]).sum(-1).mean(axis=(0, 2))
    for lag in spatial_lags:
        # Open boundary: never wrap the last spin back onto the first.
        result[f'spatial_dot_lag{lag}'] = (s[:, :, :-lag]*s[:, :, lag:]).sum(-1).mean(axis=(0, 2))
        result[f'spatial_connected_lag{lag}'] = (centered[:, :, :-lag]*centered[:, :, lag:]).sum(-1).mean(axis=(0, 2))
    return result
