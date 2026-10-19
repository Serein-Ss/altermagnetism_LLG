"""Open-chain scalar/vector path networks; no analytic field or Hamiltonian input."""
import torch
from torch import nn
from torch.nn import functional as F
from scripts.model.sphere import tangent_project, sphere_exp


def neighbor(x):
    left = torch.cat((torch.zeros_like(x[..., :1, :]), x[..., :-1, :]), -2)
    right = torch.cat((x[..., 1:, :], torch.zeros_like(x[..., :1, :])), -2)
    degree = x.new_full((x.shape[-2], 1), 2.)
    degree[0] = degree[-1] = 1.
    return (left+right)/degree


def coarse_mean(x):
    """Dyadic block means, without normalizing vectors or including padded zeros."""
    n = x.shape[-2]
    if n == 1:
        return x
    if n % 2:
        x = torch.cat((x, x[..., -1:, :]), -2)
    return (x[..., ::2, :]+x[..., 1::2, :])/2


def hierarchy(x):
    # Shared operation at every scale, preserving vector magnitudes.
    n = x.shape[-2]
    if n <= 2:
        return x.mean(-2, keepdim=True).expand_as(x)
    pooled = coarse_mean(x)
    return (.5*pooled+.5*hierarchy(pooled)).repeat_interleave(2, -2)[..., :n, :]


class Block(nn.Module):
    def __init__(self, width, temporal, multiscale):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.local = nn.Linear(2*width, width)
        self.film = nn.Linear(2, 2*width)
        self.temporal = temporal
        self.multiscale = multiscale
        if temporal:
            self.attn = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        if multiscale:
            self.scale = nn.Linear(width, width)

    def forward(self, h, cond):
        y = self.norm(h)
        out = self.local(torch.cat((y, neighbor(y)), -1))
        if self.multiscale:
            out = out+self.scale(hierarchy(y))
        if self.temporal:
            b, t, n, w = y.shape
            seq = y.transpose(1, 2).reshape(b*n, t, w)
            attended = self.attn(seq, seq, seq, need_weights=False)[0]
            out = out+attended.reshape(b, n, t, w).transpose(1, 2)
        scale, bias = self.film(cond).chunk(2, -1)
        return h+.2*F.silu(out*(1+scale[:, None, None])+bias[:, None, None])


class Net(nn.Module):
    def __init__(self, kind, width=64, depth=3):
        super().__init__()
        self.kind = kind
        self.input = nn.Linear(64+5, width)
        self.blocks = nn.ModuleList([Block(width, kind != 'A', kind == 'C') for _ in range(depth)])
        self.output = nn.Linear(width, 16)
        self.register_buffer('temperature_mean', torch.tensor(0.))
        self.register_buffer('temperature_scale', torch.tensor(1.))

    def forward(self, x, tau, initial, times, temperature, axis=None):
        axis = x.new_tensor([0., 0., 1.]) if axis is None else axis
        anchor = initial[:, None].expand_as(x)
        proj = lambda v: (v*axis).sum(-1, keepdim=True)*axis
        coarse = hierarchy(x) if self.kind == 'C' else torch.zeros_like(x)
        coarse0 = hierarchy(anchor) if self.kind == 'C' else torch.zeros_like(x)
        vectors = torch.stack((x, anchor, neighbor(x), neighbor(anchor), proj(x), proj(anchor), coarse, coarse0), -2)
        gram = (vectors@vectors.transpose(-1, -2)).flatten(-2)
        b, t, n, _ = x.shape
        physical = times[None, :, None].expand(b, t, n)  # fixed 1 ps time scale, not per-window rescaling
        temp = (torch.as_tensor(temperature, device=x.device, dtype=x.dtype)-self.temperature_mean)/self.temperature_scale
        meta = torch.stack((tau[:, None, None].expand_as(physical), physical,
                            torch.sin(physical*torch.pi), torch.cos(physical*torch.pi),
                            torch.ones_like(physical)*temp), -1)
        h = F.silu(self.input(torch.cat((gram, meta), -1)))
        cond = torch.stack((tau, torch.ones_like(tau)*temp), -1)
        for block in self.blocks:
            h = block(h, cond)
        direct, cross = self.output(h).chunk(2, -1)
        v = (direct[..., None]*vectors+cross[..., None]*torch.linalg.cross(x[..., None, :], vectors)).sum(-2)
        v = tangent_project(x, v)
        return torch.cat((torch.zeros_like(v[:, :1]), v[:, 1:]), 1)


@torch.no_grad()
def generate(net, initial, times, temperature, seed, steps=32):
    if net.kind == 'A' and len(times) > 2:
        frames = [initial]
        for j in range(1, len(times)):
            pair = generate(net, frames[-1], times[j-1:j+1]-times[j-1], temperature, seed+100003*j, steps)
            frames.append(pair[:, -1])
        return torch.stack(frames, 1)
    anchor = initial[:, None].expand(-1, len(times), -1, -1)
    if net.kind == 'D':
        v = net(anchor, initial.new_ones(len(initial)), initial, times, temperature)
        y = sphere_exp(anchor, v)
        y[:, 0] = initial
        return y
    rng = torch.Generator(device=initial.device).manual_seed(seed)
    x = F.normalize(torch.randn(anchor.shape, device=initial.device, dtype=initial.dtype, generator=rng), dim=-1)
    x[:, 0] = initial
    for k in range(steps):
        v = net(x, x.new_full((len(x),), k/steps), initial, times, temperature)
        x = sphere_exp(x, v/steps)
        x[:, 0] = initial
    return x
