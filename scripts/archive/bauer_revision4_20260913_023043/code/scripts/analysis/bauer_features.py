"""Revision-4 44-dimensional development features, no automatic GO."""
import numpy as np


def primary_features(energy_per_spin, magnetization, indices):
    e, m = np.asarray(energy_per_spin), np.asarray(magnetization)
    if e.ndim != 2 or m.shape != (*e.shape, 3):
        raise ValueError('expected [paths,frames] energy and [paths,frames,3] M')
    if len(indices) != 11 or len(set(indices)) != 11 or min(indices) < 0 or max(indices) >= e.shape[1]:
        raise ValueError('eleven distinct frozen frame indices required')
    features = np.concatenate((e[..., None], m), -1)[:, indices].reshape(len(e), 44)
    if not np.isfinite(features).all():
        raise ValueError('nonfinite primary feature')
    return features
