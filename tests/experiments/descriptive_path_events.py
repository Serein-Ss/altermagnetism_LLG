"""Sampled-time sign events, not magnetic-basin or committed-switch labels."""
import numpy as np


def sign_events(mz, reduced_time):
    """One row per trajectory initially strictly positive.

    Crossings between saved frames can be missed. First-negative times are
    observation times, not interpolated first-passage times; NaN means no
    observed crossing, not an infinite physical first-passage time.
    """
    mz = np.asarray(mz, dtype=float)
    time = np.asarray(reduced_time, dtype=float)
    if mz.ndim != 2 or mz.shape[0] < 1 or mz.shape[1] < 2:
        raise ValueError('expected nonempty [paths, frames] with at least two frames')
    if time.shape != (mz.shape[1],) or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError('finite strictly increasing matching time grid required')
    if not np.isfinite(mz).all() or np.any(np.abs(mz) > 1 + 1e-5):
        raise ValueError('finite normalized magnetization required')
    if np.any(mz[:, 0] <= 0):
        raise ValueError('these directed events require positive initial magnetization')
    negative = mz < 0
    crossed = negative.any(axis=1)
    first = negative.argmax(axis=1)
    first_time = np.where(crossed, time[first], np.nan)
    return dict(negative_endpoint=negative[:, -1], crossed_zero=crossed,
                first_observed_negative_time=first_time,
                observation_end=np.full(mz.shape[0], time[-1]))
