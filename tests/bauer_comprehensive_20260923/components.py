"""Controlled diagnostic candidates; never used to replace frozen production models."""
import numpy as np
import h5py
import torch
from torch import nn
from torch.nn import functional as F
from scripts.model.sphere import sphere_exp, tangent_project
from tests.bauer_p2_20260923.common import Model
from tests.mdgen_path_refactor_20260918.data import PathWindows


class CachedWindows(PathWindows):
    """Same window/index contract, one read per HDF5 file, no data transformation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
        for path, _, _, _ in self.records:
            with h5py.File(path) as h:
                s = h['spins'][:]
                if not np.isfinite(s).all() or not np.allclose(np.linalg.norm(s, axis=-1), 1, atol=1e-6, rtol=0):
                    raise ValueError(f'invalid cached spins: {path}')
                self.cache[path] = (s, h['time_ps'][:], h['seeds'][:])

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        group = int(np.searchsorted(self.ends, index, side='right'))
        offset = index - (self.ends[group-1] if group else 0)
        path, row, starts, steps = self.records[group]
        branch, window = divmod(offset, len(starts))
        start = starts[window]
        s, t, seeds = self.cache[path]
        spins = s[branch, start:start+steps+1].copy()
        times = t[start:start+steps+1]
        return dict(spins=spins, initial=spins[0].copy(), time_ps=times-times[0],
                    origin_ps=float(times[0]), parent_id=row['parent_id'], family=row['family'],
                    split=row['split'], branch=branch, seed=int(seeds[branch]))


class VectorTemporalModel(Model):
    """P2 graph plus invariant-attention/vector-value residual.

    The local-extra control has identical parameters and only removes cross-time
    mixing in this residual. It does NOT remove the old backbone's time attention.
    This is a diagnostic residual, not a completed multiscale architecture.
    """
    def __init__(self, config, mixing=True, global_pool=False):
        super().__init__(config)
        self.mixing = mixing
        self.global_pool = global_pool
        width = config['width']
        self.temporal_input = nn.Sequential(nn.Linear(9, width), nn.SiLU(), nn.LayerNorm(width))
        self.temporal_qk = nn.Linear(width, 2*width)
        self.temporal_gate = nn.Linear(width, 8)
        nn.init.zeros_(self.temporal_gate.weight)
        nn.init.zeros_(self.temporal_gate.bias)

    def get_extra_state(self):
        return dict(periodic=self.periodic, symmetry='SO(3)', version=2,
                    residual='vector_time' if self.mixing else 'local_extra', global_pool=self.global_pool)

    def velocity(self, x, tau, initial, times, temperature=None, frame=None):
        base = super().velocity(x, tau, initial, times, temperature, frame)
        frame = torch.eye(3, device=x.device, dtype=x.dtype) if frame is None else frame
        anchor = initial[:, None].expand_as(x)
        meta = torch.stack((tau[:, None, None].expand(x.shape[:-1]),
                            times[None, :, None].expand(x.shape[:-1])), -1)
        features = torch.cat(((x*anchor).sum(-1, keepdim=True), x@frame.T, anchor@frame.T, meta), -1)
        h = self.temporal_input(features).transpose(1, 2)  # B,N,T,C
        if self.global_pool:
            h = h + h.mean(1, keepdim=True)
        q, k = self.temporal_qk(h).chunk(2, -1)
        b, n, t, c = q.shape
        q = q.reshape(b, n, t, 4, c//4).transpose(2, 3)
        k = k.reshape(b, n, t, 4, c//4).transpose(2, 3)
        weights = torch.softmax((q@k.transpose(-1, -2))/(c//4)**.5, -1)
        if not self.mixing:
            # q/k remain active and parameter matched, without remote value access.
            values = x.transpose(1, 2)[:, :, None].expand(b, n, 4, t, 3)
            values = values*weights.diagonal(dim1=-2, dim2=-1)[..., None]
        else:
            vector_input = x
            if self.global_pool:
                vector_input = .5*(x+x.mean(2, keepdim=True))
            values = weights@vector_input.transpose(1, 2)[:, :, None]
        values = values.permute(0, 3, 1, 2, 4)  # B,T,N,heads,xyz
        direct, cross = self.temporal_gate(h.transpose(1, 2)).chunk(2, -1)
        correction = (direct[..., None]*values + cross[..., None]*torch.linalg.cross(x[..., None, :], values)).sum(-2)
        correction = tangent_project(x, correction)
        correction = torch.cat((torch.zeros_like(correction[:, :1]), correction[:, 1:]), 1)
        return base + correction


def make_model(config, arm):
    arm = arm.removesuffix('_long')
    if arm in ('temporal', 'temporal_source', 'local_extra', 'global_temporal'):
        return VectorTemporalModel(config, mixing=arm != 'local_extra', global_pool=arm == 'global_temporal')
    return Model(config)


def source_from_noise(noise, initial, arm):
    # Shifted Gaussian normalized on S2 has full support; not physical heat noise.
    shift = 2. if arm.removesuffix('_long') in ('source', 'temporal_source') else 0.
    z = F.normalize(noise + shift*initial[:, None], dim=-1)
    z[:, 0] = initial
    return z


def synthetic_parents(split):
    g = torch.Generator().manual_seed(923610 if split == 'train' else 923611)
    return F.normalize(torch.randn(24 if split == 'train' else 12, 16, 3, generator=g), dim=-1)


def synthetic_paths(initial, kind, generator):
    """Known conditional laws, NOT LLG. All random futures are unselected.

    Deterministic z rotation; correlated tangent Gaussian perturbations; or a
    balanced mixture of opposite z rotations with weaker correlated noise.
    """
    b, n, _ = initial.shape
    t = torch.linspace(0, 1, 11, device=initial.device, dtype=initial.dtype)
    sign = torch.ones(b, device=initial.device, dtype=initial.dtype)
    if kind == 'mixture':
        sign = 2*torch.randint(2, (b,), device=initial.device, generator=generator)-1
    angle = sign[:, None]*t[None]*.8
    c, s = angle.cos()[:, :, None], angle.sin()[:, :, None]
    x, y, z = initial.unbind(-1)
    rotated = torch.stack((c*x[:, None]-s*y[:, None], s*x[:, None]+c*y[:, None],
                           z[:, None].expand(b, 11, n)), -1)
    if kind != 'deterministic':
        noise = torch.randn((b, 11, n, 3), device=initial.device, dtype=initial.dtype, generator=generator)
        correlated = [torch.zeros_like(noise[:, 0])]
        for j in range(1, 11):
            correlated.append(.8*correlated[-1]+.6*noise[:, j])
        scale = .25 if kind == 'gaussian' else .12
        rotated = sphere_exp(rotated, scale*torch.stack(correlated, 1))
    rotated[:, 0] = initial
    return rotated


def diagnostic_arrays(s):
    """Fixed-parent whole-path metrics; covariance is unbiased across branches."""
    s = np.asarray(s, dtype=np.float64)
    d = np.diff(s, axis=1)
    centered = d-d.mean(0)
    variance = (centered*centered).sum(-1).sum(0).mean(-1)/(len(s)-1)
    covariance = (centered[:, 1:]*centered[:, :-1]).sum(-1).sum(0).mean(-1)/(len(s)-1)
    # Entire time-by-time spin correlation, not just adjacent or selected lags.
    correlation = np.einsum('btni,buni->tu', s, s)/(s.shape[0]*s.shape[2])
    return dict(increment_variance=variance, increment_cross_covariance=covariance,
                two_time_spin_correlation=correlation)
