"""Minimal joint-path scalar/vector prototype, not the final multiscale model.

No Hamiltonian coefficients or effective fields enter the network. Cell bond
connectivity and a jointly transformed crystal frame are structural inputs.
"""
import torch
from torch import nn
from scripts.core.cell_hamiltonian import shifted
from scripts.model.sphere import tangent_project, sphere_exp


def neighbors(x, cell):
    # [batch, frame, *cell_shape, basis, channel]
    shape = x.shape
    y = x.flatten(0, 1)
    out = torch.zeros_like(y)
    degree = torch.zeros_like(y[..., :1])
    for a, b, delta, _ in cell.templates:
        for u, v, d in ((a, b, delta), (b, a, tuple(-q for q in delta))):
            out[..., u, :] += shifted(y[..., v, :], d, cell.periodic)
            degree[..., u, :] += shifted(torch.ones_like(y[..., v, :1]), d, cell.periodic)
    return (out / degree.clamp_min(1)).reshape(shape)


class PathNet(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.input = nn.Linear(81 + 3, width)
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.film = nn.ModuleList([nn.Linear(2, 2 * width) for _ in range(2)])
        self.output = nn.Linear(width, 18)

    def forward(self, x, tau, initial, cell, frame, temperature, time):
        anchor = initial[:, None].expand_as(x)
        near = neighbors(x, cell)
        previous = torch.cat((x[:, :1], x[:, :-1]), 1)
        following = torch.cat((x[:, 1:], x[:, -1:]), 1)
        axes = frame.reshape(*([1] * (x.ndim - 1)), 3, 3).expand(*x.shape[:-1], 3, 3)
        vectors = torch.cat((torch.stack((x, anchor, near, neighbors(near, cell), previous, following), -2), axes), -2)
        gram = (vectors @ vectors.transpose(-1, -2)).flatten(-2)
        scalar_shape = [x.shape[0]] + [1] * (x.ndim - 2)
        t = time.reshape(1, len(time), *([1] * (x.ndim - 3))).expand(x.shape[:-1])
        meta = torch.stack((tau.reshape(scalar_shape).expand_as(t), t, torch.full_like(t, temperature)), -1)
        h = torch.nn.functional.silu(self.input(torch.cat((gram, meta), -1)))
        cond = torch.stack((tau, torch.full_like(tau, temperature)), -1)
        for layer, norm, film in zip(self.layers, self.norms, self.film):
            scale, bias = film(cond).chunk(2, -1)
            view = [len(tau)] + [1] * (h.ndim - 2) + [h.shape[-1]]
            h = h + .1 * torch.nn.functional.silu(norm(layer(h)) * (1 + scale.reshape(view)) + bias.reshape(view))
        direct, cross = self.output(h).chunk(2, -1)
        v = (direct[..., None] * vectors + cross[..., None] * torch.linalg.cross(x[..., None, :], vectors)).sum(-2)
        v = tangent_project(x, v)
        return torch.cat((torch.zeros_like(v[:, :1]), v[:, 1:]), 1)


@torch.no_grad()
def generate(net, initial, cell, frame, temperature, time, seed=72, steps=32, source=None):
    if source is None:
        rng = torch.Generator(device=initial.device).manual_seed(seed)
        x = torch.randn((len(initial), len(time), *initial.shape[1:]), device=initial.device, dtype=initial.dtype, generator=rng)
        x = torch.nn.functional.normalize(x, dim=-1)
        x[:, 0] = initial
    else:
        x = source.clone()
    for i in range(steps):
        x = sphere_exp(x, net(x, x.new_full((len(x),), i / steps), initial, cell, frame, temperature, time) / steps)
        x[:, 0] = initial
    return x
