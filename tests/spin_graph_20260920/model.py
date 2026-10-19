"""SO(3) scalar/vector graph path candidate; not a pretrained dynamics model.

Coordinates stay fixed. Crystal frame and spins transform jointly. Only graph
connectivity is read from CellHamiltonian; no J, anisotropy or field is consumed.
"""
import itertools

import torch
from torch import nn
from torch.nn import functional as F

from scripts.model.sphere import tangent_project


def cell_edges(cell):
    """Directed multigraph, preserving periodic image multiplicity and open edges."""
    def index(coord, basis):
        out = 0
        for value, length in zip(coord, cell.shape):
            out = out*length+value
        return out*cell.basis+basis

    edges = []
    for coord in itertools.product(*(range(n) for n in cell.shape)):
        for a, b, delta, _ in cell.templates:
            target = tuple(c+d for c, d in zip(coord, delta))
            if any(not wrap and not 0 <= t < n
                   for t, n, wrap in zip(target, cell.shape, cell.periodic)):
                continue
            target = tuple(t % n for t, n in zip(target, cell.shape))
            i, j = index(coord, a), index(target, b)
            edges.extend(((i, j), (j, i)))
    return torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous()


def aggregate(values, receiver, nodes):
    out = values.new_zeros(*values.shape[:-2], nodes, values.shape[-1])
    out.index_add_(-2, receiver, values)
    degree = torch.bincount(receiver, minlength=nodes).to(values).clamp_min(1)
    return out/degree[:, None]


class SpinGraphNet(nn.Module):
    """Same forward/spherical sampler interface as the existing cell PathNet.

An explicit graph candidate with local vector updates and per-site temporal
attention. Multiscale pooling is intentionally NOT claimed by this first patch.
"""
    def __init__(self, periodic, width=32, depth=3):
        super().__init__()
        self.periodic = tuple(periodic)
        self.input = nn.Linear(49+3, width)
        self.messages = nn.ModuleList([
            nn.Sequential(nn.Linear(2*width+3, width), nn.SiLU(),
                          nn.Linear(width, width+2)) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        self.film = nn.ModuleList([nn.Linear(2, 2*width) for _ in range(depth)])
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(width, 4, batch_first=True) for _ in range(depth)])
        self.output = nn.Linear(width, 14)
        self._edges = {}

    def get_extra_state(self):
        return {'periodic': self.periodic, 'symmetry': 'SO(3)', 'version': 1}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('checkpoint geometry/symmetry contract mismatch')

    def forward(self, x, tau, initial, cell, frame, temperature, time):
        if tuple(cell.periodic) != self.periodic:
            raise ValueError('boundary differs from model package')
        if tuple(x.shape[2:-1]) != (*cell.shape, cell.basis):
            raise ValueError('spin array does not match cell geometry')
        if time.ndim != 1 or len(time) != x.shape[1] or not torch.all(time[1:] > time[:-1]):
            raise ValueError('physical frame times must be strictly increasing')
        key = (cell.shape, cell.basis, tuple((a, b, d) for a, b, d, _ in cell.templates), str(x.device))
        if key not in self._edges:
            self._edges[key] = cell_edges(cell).to(x.device)
        receiver, sender = self._edges[key]
        b, nt = x.shape[:2]
        s = x.reshape(b, nt, -1, 3)
        anchor = initial.reshape(b, 1, -1, 3).expand_as(s)
        previous = torch.cat((s[:, :1], s[:, :-1]), 1)
        following = torch.cat((s[:, 1:], s[:, -1:]), 1)
        axes = frame.reshape(1, 1, 1, 3, 3).expand(*s.shape[:-1], 3, 3)
        vectors = torch.cat((torch.stack((s, anchor, previous, following), -2), axes), -2)
        gram = (vectors@vectors.transpose(-1, -2)).flatten(-2)
        times = time.reshape(1, nt, 1).expand(s.shape[:-1])
        temp = torch.as_tensor(temperature, device=x.device, dtype=x.dtype).expand(b)
        meta = torch.stack((tau[:, None, None].expand_as(times), times,
                            temp[:, None, None].expand_as(times)), -1)
        h = F.silu(self.input(torch.cat((gram, meta), -1)))
        cond = torch.stack((tau, temp), -1)
        v = s
        for message, norm, film, attention in zip(self.messages, self.norms, self.film, self.attention):
            y = norm(h)
            vi, vj = v[:, :, receiver], v[:, :, sender]
            invariants = torch.stack(((vi*vj).sum(-1), vi.square().sum(-1), vj.square().sum(-1)), -1)
            m = message(torch.cat((y[:, :, receiver], y[:, :, sender], invariants), -1))
            scalar = aggregate(m[..., :-2], receiver, s.shape[2])
            vector = m[..., -2:-1]*vj+m[..., -1:]*torch.linalg.cross(vi, vj)
            v = v+.1*aggregate(vector, receiver, s.shape[2])
            sequence = y.transpose(1, 2).reshape(-1, nt, y.shape[-1])
            temporal = attention(sequence, sequence, sequence, need_weights=False)[0]
            temporal = temporal.reshape(b, -1, nt, y.shape[-1]).transpose(1, 2)
            scale, bias = film(cond).chunk(2, -1)
            h = h+.1*F.silu((scalar+temporal)*(1+scale[:, None, None])+bias[:, None, None])
        # Learned vector channel replaces only the current-state basis vector.
        vectors = torch.cat((v[..., None, :], vectors[..., 1:, :]), -2)
        direct, cross = self.output(h).chunk(2, -1)
        velocity = (direct[..., None]*vectors + cross[..., None]*torch.linalg.cross(s[..., None, :], vectors)).sum(-2)
        velocity = tangent_project(s, velocity).reshape_as(x)
        return torch.cat((torch.zeros_like(velocity[:, :1]), velocity[:, 1:]), 1)
