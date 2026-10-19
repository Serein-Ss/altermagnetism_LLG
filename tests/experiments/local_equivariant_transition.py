"""Sample-trained SO(3)-equivariant open-chain conditional sphere flow prototype."""
import math
import torch
from torch import nn
from scripts.model.sphere import tangent_project, sphere_exp
from scripts.model.gate_f import path_interpolate


def neighbors(x):
    zero = torch.zeros_like(x[:, :1])
    return torch.cat((zero, x[:, :-1]), 1), torch.cat((x[:, 1:], zero), 1)


class LocalEquivariantTransition(nn.Module):
    def __init__(self, theta_mean, theta_std, width=64):
        super().__init__()
        if theta_std <= 0:
            raise ValueError('positive training temperature scale required')
        self.register_buffer('theta_mean', torch.tensor(float(theta_mean)))
        self.register_buffer('theta_std', torch.tensor(float(theta_std)))
        self.input = nn.Linear(51, width)
        self.convs = nn.ModuleList([nn.Conv1d(width, width, 3, padding=1) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.conditions = nn.ModuleList([nn.Linear(5, 2*width) for _ in range(2)])
        self.output = nn.Linear(width, 12)

    def forward(self, x, tau, initial, theta, lag, axis):
        left, right = neighbors(x)
        al, ar = neighbors(initial)
        axis = axis[:, None].expand_as(x)
        vectors = torch.stack((x, initial, left, right, al, ar, axis), -2)
        gram = (vectors @ vectors.transpose(-1, -2)).flatten(-2)
        masks = x.new_ones((*x.shape[:2], 2))
        masks[:, 0, 0] = 0; masks[:, -1, 1] = 0
        condition = torch.stack(((theta-self.theta_mean)/self.theta_std, lag/.01,
                                 tau, torch.sin(math.pi*tau), torch.cos(math.pi*tau)), -1)
        h = torch.nn.functional.silu(self.input(torch.cat((gram, masks), -1)))
        for conv, norm, inject in zip(self.convs, self.norms, self.conditions):
            scale, shift = inject(condition).chunk(2, -1)
            residual = norm(conv(h.transpose(1, 2)).transpose(1, 2))
            h = h + .1*torch.nn.functional.silu(residual*(1+scale[:, None])+shift[:, None])
        direct, crossed = self.output(h).chunk(2, -1)
        basis = vectors[..., 1:, :]
        velocity = (direct[..., None]*basis + crossed[..., None]*torch.linalg.cross(x[..., None, :], basis)).sum(-2)
        return tangent_project(x, velocity)


def source(initial, generator, scale):
    if scale <= 0:
        raise ValueError('positive latent scale required')
    noise = torch.randn(initial.shape, dtype=initial.dtype, device=initial.device, generator=generator)
    return torch.nn.functional.normalize(initial+scale*noise, dim=-1)


def loss(model, initial, target, theta, lag, axis, generator, scale):
    z = source(initial, generator, scale)
    tau = torch.rand(initial.shape[0], dtype=initial.dtype, device=initial.device, generator=generator)
    x, velocity, diagnostics = path_interpolate(torch.stack((initial, z), 1),
        torch.stack((initial, target), 1), tau, generator)
    predicted = model(x[:, 1], tau, initial, theta, lag, axis)
    return (predicted-velocity[:, 1]).square().sum(-1).mean(), diagnostics


@torch.no_grad()
def sample(model, initial, theta, lag, axis, generator, scale, steps):
    if steps < 1:
        raise ValueError('positive integration step count required')
    x = source(initial, generator, scale)
    for step in range(steps):
        tau = initial.new_full((len(initial),), step/steps)
        x = sphere_exp(x, model(x, tau, initial, theta, lag, axis)/steps)
    return x
