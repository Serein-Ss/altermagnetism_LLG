"""Experimental symmetry repair and multidimensional space-time backbone.

The repaired chain uses group averaging of the entire learned vector field.
No frozen code/checkpoint is modified. The multidimensional model consumes
connectivity and a crystal frame, never Hamiltonian coefficients or fields.
"""
import torch
from torch import nn
from torch.nn import functional as F
from scripts.core.cell_hamiltonian import shifted
from scripts.model.sphere import tangent_project
from tests.bauer_closed_loop_20260919.model import Net
from tests.path_smoke_20260917.model import neighbors


class ReversalNet(Net):
    def forward(self, x, tau, initial, times, temperature, axis=None):
        direct = super().forward(x, tau, initial, times, temperature, axis)
        reverse = super().forward(x.flip(-2), tau, initial.flip(-2), times, temperature, axis)
        return .5*(direct+reverse.flip(-2))


def scale_context(x, cell, distance):
    y = x.flatten(0, 1)
    total = y.clone(); count = torch.ones_like(y[..., :1])
    for axis in range(len(cell.shape)):
        for sign in (-1, 1):
            delta = [0]*len(cell.shape); delta[axis] = sign*distance
            total = total+shifted(y, delta, cell.periodic)
            count = count+shifted(torch.ones_like(y[..., :1]), delta, cell.periodic)
    return (total/count).reshape_as(x)


def directional_context(x, cell, axis):
    """Oriented cell-coordinate message, not a Hamiltonian-derived effective field."""
    y = x.flatten(0, 1); delta = [0]*len(cell.shape); delta[axis] = 1
    forward = shifted(y, delta, cell.periodic)
    backward = shifted(y, [-v for v in delta], cell.periodic)
    return (.5*(forward-backward)).reshape_as(x)


class SpaceTimeNet(nn.Module):
    def __init__(self, width=32, depth=3):
        super().__init__()
        self.input = nn.Linear(169+3, width)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        self.local = nn.ModuleList([nn.Linear(width*3, width) for _ in range(depth)])
        self.attn = nn.ModuleList([nn.MultiheadAttention(width, 4, batch_first=True) for _ in range(depth)])
        self.film = nn.ModuleList([nn.Linear(2, 2*width) for _ in range(depth)])
        self.output = nn.Linear(width, 26)

    def forward(self, x, tau, initial, cell, frame, temperature, time):
        anchor = initial[:, None].expand_as(x)
        near = neighbors(x, cell)
        previous = torch.cat((x[:, :1], x[:, :-1]), 1)
        following = torch.cat((x[:, 1:], x[:, -1:]), 1)
        global_mean = x.flatten(2, -2).mean(2, keepdim=True)
        global_mean = global_mean.reshape(*x.shape[:2], *([1]*(x.ndim-3)), 3).expand_as(x)
        axes = frame.reshape(*([1]*(x.ndim-1)), 3, 3).expand(*x.shape[:-1], 3, 3)
        vectors = torch.cat((torch.stack((x, anchor, near, neighbors(near, cell), previous,
                             following, scale_context(x, cell, 4), global_mean,
                             directional_context(x, cell, 0), directional_context(x, cell, 1)), -2), axes), -2)
        gram = (vectors@vectors.transpose(-1, -2)).flatten(-2)
        shape = [len(x)]+[1]*(x.ndim-2)
        t = time.reshape(1, len(time), *([1]*(x.ndim-3))).expand(x.shape[:-1])
        meta = torch.stack((tau.reshape(shape).expand_as(t), t, torch.full_like(t, temperature)), -1)
        h = F.silu(self.input(torch.cat((gram, meta), -1)))
        cond = torch.stack((tau, torch.full_like(tau, temperature)), -1)
        for j, (norm, local, attn, film) in enumerate(zip(self.norms, self.local, self.attn, self.film)):
            y = norm(h)
            spatial = local(torch.cat((y, neighbors(y, cell), scale_context(y, cell, 2**(j+1))), -1))
            b, nt = y.shape[:2]; w = y.shape[-1]
            seq = y.reshape(b, nt, -1, w).transpose(1, 2).reshape(-1, nt, w)
            temporal = attn(seq, seq, seq, need_weights=False)[0]
            temporal = temporal.reshape(b, -1, nt, w).transpose(1, 2).reshape_as(y)
            scale, bias = film(cond).chunk(2, -1)
            view = [b]+[1]*(h.ndim-2)+[w]
            h = h+.1*F.silu((spatial+temporal)*(1+scale.reshape(view))+bias.reshape(view))
        direct, cross = self.output(h).chunk(2, -1)
        v = (direct[..., None]*vectors+cross[..., None]*torch.linalg.cross(x[..., None, :], vectors)).sum(-2)
        v = tangent_project(x, v)
        return torch.cat((torch.zeros_like(v[:, :1]), v[:, 1:]), 1)


def observed_context(truth, known):
    """No hidden-frame information is passed to a future masked conditioner."""
    if known.shape != truth.shape[:-1] or known.dtype != torch.bool:
        raise ValueError('Boolean mask must have one entry per space-time spin')
    return torch.where(known[..., None], truth, torch.zeros_like(truth))


def masked_flow_loss(prediction, target, known):
    free = ~known
    if not free.any():
        raise ValueError('At least one unobserved token is required')
    return (prediction-target).square().sum(-1)[free].mean()
