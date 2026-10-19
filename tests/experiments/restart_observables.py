"""Parent-preserving observables for short continuations with distinct anchors.

Unlike path_observables, these arrays do not average over parents or claim a
single-initial ensemble variance. A subsequent analysis must retain parent
pairing across restart times and trained seeds.
"""
import numpy as np


def per_parent(paths):
    s = np.asarray(paths, dtype=np.float64)
    if s.ndim != 4 or s.shape[-1] != 3 or s.shape[0] < 1 or min(s.shape[1:3]) < 2:
        raise ValueError('expected [parents, frames>=2, sites>=2, 3]')
    if not np.isfinite(s).all() or np.max(np.abs(np.linalg.norm(s, axis=-1)-1)) > 1e-5:
        raise ValueError('finite unit spins required')
    mag = s.mean(axis=2)
    return dict(magnetization=mag, magnetization_square=mag**2,
                spatial_dot_lag1=(s[:, :, :-1]*s[:, :, 1:]).sum(-1).mean(-1),
                increment_second_moment_lag1=np.square(s[:, 1:]-s[:, :-1]).sum(-1).mean(-1),
                anchor_dot=(s*s[:, :1]).sum(-1).mean(-1))


def matched_restart(generated, reference):
    """Signed observable differences, not a paired spin-path RMSE.

    The two continuations share the anchor but NOT the physical noise. Individual
    differences are noisy paired contributions, not per-path prediction errors.
    The caller must verify parent IDs and matching physical time/conditions.
    """
    g, r = np.asarray(generated), np.asarray(reference)
    if g.shape != r.shape or g.ndim != 4:
        raise ValueError('matching continuation shapes required')
    if not np.array_equal(g[:, 0].astype('f4'), r[:, 0].astype('f4')):
        raise ValueError('paired parents must have identical anchors at inference precision')
    go, ro = per_parent(g), per_parent(r)
    return dict(generated=go, reference=ro,
                differences={k: go[k]-ro[k] for k in go})
