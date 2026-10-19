"""Directional derivative of open-chain nearest-neighbor correlation.

This is a derivative with respect to FLOW time when v is an RFM velocity,
not a physical-time LLG derivative. Keep signed per-parent contributions;
neither pointwise agreement nor a mean residual certifies the path law.
"""
import torch


def correlation_rate(spins, velocity):
    """Return [parents, frames] for inputs [parents, frames, sites, 3]."""
    if spins.shape != velocity.shape or spins.ndim != 4 or spins.shape[-1] != 3 or spins.shape[2] < 2:
        raise ValueError('matching [parents, frames, sites>=2, 3] tensors required')
    if not torch.isfinite(spins).all() or not torch.isfinite(velocity).all():
        raise ValueError('finite states and velocities required')
    # C = sum_{i=0}^{L-2} dot(s_i,s_{i+1})/(L-1): no wrapped bond.
    return ((velocity[:, :, :-1]*spins[:, :, 1:]
             + spins[:, :, :-1]*velocity[:, :, 1:]).sum(-1)).mean(-1)
