import numpy as np


def patches(state, radius):
    state = np.asarray(state, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != 3 or radius < 1:
        raise ValueError('chain spins and positive radius required')
    sites = np.arange(len(state))[:, None] + np.arange(-radius, radius + 1)
    mask = (sites >= 0) & (sites < len(state))
    return state[np.clip(sites, 0, len(state)-1)] * mask[..., None], mask


def nearest(query, query_mask, bank, bank_mask):
    """Nearest axial-aligned spin patch with exactly matching open-edge mask.

    Global rotations about registered z are minimized analytically. No reflection,
    z inversion or independent per-spin rotation is allowed. RMS chordal distance
    is averaged over valid spins (not vector components).
    """
    distance = np.full(len(query), np.nan)
    for i, (q, mask) in enumerate(zip(query, query_mask)):
        b = bank[np.all(bank_mask == mask, axis=1)]
        if not len(b):
            continue
        planar = np.sum((b[..., 0] + 1j*b[..., 1]) *
                        (q[..., 0] - 1j*q[..., 1]), axis=1)
        dot = np.abs(planar) + np.sum(b[..., 2]*q[..., 2], axis=1)
        norm = np.sum(b*b, axis=(1, 2)) + np.sum(q*q)
        distance[i] = np.sqrt(np.maximum(0, np.min(norm-2*dot))/mask.sum())
    return distance


def bias(generated, reference):
    """Fixed-state endpoint increment bias; unbiased squared-bias estimate.

    Independent whole-future branch ensembles; correction can be negative.
    No assumption of independent sites or acceptance threshold.
    """
    g, r = [np.asarray(x, dtype=np.float64) for x in (generated, reference)]
    if (g.ndim != 4 or r.ndim != 4 or g.shape[1:] != r.shape[1:] or
            g.shape[1] != 2 or min(len(g), len(r)) < 2):
        raise ValueError('matching two-frame ensembles required')
    initial = r[0, 0]
    if not np.allclose(g[:, 0], initial, atol=2e-6, rtol=0) or not np.allclose(r[:, 0], initial, atol=2e-6, rtol=0):
        raise ValueError('actual conditional initial state differs')
    dg, dr = g[:, 1]-g[:, 0], r[:, 1]-r[:, 0]
    error = dg.mean(0)-dr.mean(0)
    variance = dg.var(0, ddof=1)/len(g) + dr.var(0, ddof=1)/len(r)
    return dict(error=error, mean_rmse=np.sqrt(np.mean(error**2)),
                site_rmse=np.sqrt(np.mean(error**2, axis=-1)),
                corrected_mse=np.mean(error**2-variance),
                longitudinal_rmse=np.sqrt(np.mean(error[:, 2]**2)),
                transverse_rmse=np.sqrt(np.mean(error[:, :2]**2)),
                estimator_variance=np.mean(variance))
