"""Versioned 2D graph adapter: geometric directions, no Hamiltonian coefficients."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from scripts.model.sphere import tangent_project, sphere_exp
from tests.spin_graph_20260920.model import SpinGraphNet, cell_edges, aggregate


class GeometryGraph(SpinGraphNet):
    def __init__(self, arm, width=64):
        super().__init__((False, False), width=width, depth=3)
        if arm not in ('shifted', 'uniform', 'global'):
            raise ValueError(arm)
        self.arm = arm
        self.geometry_message = nn.Sequential(nn.Linear(8, width), nn.SiLU(), nn.Linear(width, 6))
        if arm == 'global':
            self.global_gate = nn.Sequential(nn.Linear(5, width), nn.SiLU(), nn.Linear(width, 4))
        self._geometry = {}

    def get_extra_state(self):
        return dict(version=1, arm=self.arm, periodic=self.periodic, system='miltat',
                    temperature_K=12.5, alpha=.5, block_ps=1., save_ps=.1,
                    symmetry='joint SO(3)', source_shift=0. if self.arm=='uniform' else 2.)

    def geometry(self, cell, x, frame):
        if cell.basis != 2 or not hasattr(cell, 'positions'):
            raise ValueError('expected triangular monolayer geometry')
        key = (cell.shape, str(x.device), x.dtype)
        if key not in self._geometry:
            edges = cell_edges(cell).to(x.device)
            pos = cell.positions(x).reshape(-1, 3)
            # No periodic wrapping here: this package is strictly open/open.
            d = pos[edges[1]]-pos[edges[0]]
            if not torch.allclose(d.norm(dim=-1), torch.ones_like(d[:, 0]), atol=1e-6):
                raise ValueError('non-nearest or incorrectly oriented bond')
            self._geometry[key] = (edges, d)
        edges, direction = self._geometry[key]
        return edges, direction@frame

    def correction(self, x, tau, initial, cell, frame, time):
        (receiver, sender), direction = self.geometry(cell, x, frame)
        s = x.flatten(2, -2); a = initial.flatten(1, -2)[:, None].expand_as(s)
        u, v = s[:, :, receiver], s[:, :, sender]
        ai, aj = a[:, :, receiver], a[:, :, sender]
        d = direction[None, None].expand_as(u)
        normal = frame[2].expand_as(u)
        dm_direction = torch.linalg.cross(normal, d)
        meta = torch.stack((tau[:, None, None].expand(u.shape[:-1]),
                            time[None, :, None].expand(u.shape[:-1])), -1)
        inv = torch.stack(((u*d).sum(-1), (v*d).sum(-1),
                           (ai*d).sum(-1), (aj*d).sum(-1),
                           (u*v).sum(-1), (torch.linalg.cross(u,v)*dm_direction).sum(-1)), -1)
        weights = self.geometry_message(torch.cat((inv, meta), -1))
        basis = torch.stack((v, ai, d, dm_direction,
                             torch.linalg.cross(v, dm_direction), torch.linalg.cross(ai, v)), -2)
        result = aggregate((weights[..., None]*basis).sum(-2), receiver, s.shape[2])
        if self.arm == 'global':
            mean_s = s.mean(2, keepdim=True).expand_as(s)
            mean_a = a.mean(2, keepdim=True).expand_as(s)
            meta = torch.stack((tau[:, None, None].expand(s.shape[:-1]),
                                time[None, :, None].expand(s.shape[:-1])), -1)
            features = torch.cat((torch.stack(((s*mean_s).sum(-1), (a*mean_a).sum(-1),
                                               (s*a).sum(-1)), -1), meta), -1)
            basis = torch.stack((mean_s, mean_a, torch.linalg.cross(s,mean_s),
                                 torch.linalg.cross(s,mean_a)), -2)
            result = result+(self.global_gate(features)[..., None]*basis).sum(-2)
        result = tangent_project(s, result).reshape_as(x)
        return torch.cat((torch.zeros_like(result[:, :1]), result[:, 1:]), 1)

    def forward(self, x, tau, initial, cell, frame, temperature, time):
        if abs(float(temperature)) > 1e-8:
            raise ValueError('constant training-temperature token must be zero')
        return super().forward(x,tau,initial,cell,frame,temperature,time)+self.correction(x,tau,initial,cell,frame,time)


def source(initial, frames, generator, shift):
    z = torch.randn((len(initial),frames,*initial.shape[1:]), device=initial.device,
                    dtype=initial.dtype,generator=generator)
    z = F.normalize(z+shift*initial[:,None],dim=-1)
    z[:,0] = initial
    return z


@torch.no_grad()
def sample(net, initial, cell, times, steps, generator):
    x = source(initial,len(times),generator,0. if net.arm=='uniform' else 2.)
    frame = torch.eye(3,device=x.device,dtype=x.dtype)
    for k in range(steps):
        v = net(x,x.new_full((len(x),),k/steps),initial,cell,frame,0.,times)
        x = sphere_exp(x,v/steps); x[:,0] = initial
    return x
