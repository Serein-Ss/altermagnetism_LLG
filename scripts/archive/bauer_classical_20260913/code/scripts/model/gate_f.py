"""Gate-F development models; no legacy checkpoint or physical certification claim.

Paths use [B,F,A,Nx,Ny,3]. Explicit edges are compiled from the existing
CellHamiltonian templates (including periodic images), not a second material
stencil. Supported physics: exchange + a uniform uniaxial anisotropy, zero drive.
"""
import math
import torch
from torch import nn
from scripts.model.sphere import tangent_project, sphere_exp


class CellGraph(nn.Module):
    """One undirected edge per template/cell, preserving open boundaries."""
    def __init__(self, cell):
        super().__init__()
        if len(cell.shape) != 2 or tuple(cell.anisotropy[:2]) != (0., 0.):
            raise ValueError('Gate-F supports 2D cells with uniaxial anisotropy')
        self.shape = (cell.basis, *cell.shape)
        self.periodic = cell.periodic
        nx, ny = cell.shape
        edges, weights = [], []
        for a, b, (dx, dy), j in cell.templates:
            for x in range(nx):
                for y in range(ny):
                    xx, yy = x+dx, y+dy
                    if (not cell.periodic[0] and not 0 <= xx < nx) or (not cell.periodic[1] and not 0 <= yy < ny):
                        continue
                    edges.append((a*nx*ny+x*ny+y, b*nx*ny+(xx % nx)*ny+yy % ny))
                    weights.append(j)
        self.register_buffer('edges', torch.tensor(edges, dtype=torch.long).reshape(-1, 2))
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float64))
        self.register_buffer('kappa', torch.tensor(cell.anisotropy[2], dtype=torch.float64))

    def aggregate(self, x, weighted=False):
        # x is [B,F,N,C]; normalization is local, never across space.
        i, j = self.edges.unbind(-1)
        w = self.weights.to(x) if weighted else torch.ones_like(self.weights).to(x)
        if weighted == 'absolute':
            w = w.abs()
        out = torch.zeros_like(x)
        out.index_add_(2, i, x[:, :, j]*w[None, None, :, None])
        out.index_add_(2, j, x[:, :, i]*w[None, None, :, None])
        return out

    def field(self, x, axis):
        axis = axis[:, None, None, :]
        return self.aggregate(x, True) + 2*self.kappa.to(x)*(x*axis).sum(-1, keepdim=True)*axis

    def energy(self, x, axis):
        i, j = self.edges.unbind(-1)
        exchange = -(self.weights.to(x)*(x[:, :, i]*x[:, :, j]).sum(-1)).sum(-1)
        return exchange-self.kappa.to(x)*(x*axis[:, None, None]).sum(-1).square().sum(-1)


class ScalarBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.local = nn.Linear(width, width)
        self.neighbor = nn.Linear(width, width, bias=False)
        self.temporal = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)
        self.film = nn.Linear(2, 2*width)

    def forward(self, h, condition, graph, degree):
        temporal = (torch.cat((h[:, :1], h[:, :-1]), 1)+torch.cat((h[:, 1:], h[:, -1:]), 1))/2
        z = self.local(h)+self.neighbor(graph.aggregate(h)/degree)+self.temporal(temporal)
        scale, shift = self.film(condition).chunk(2, -1)
        return h+torch.nn.functional.silu(self.norm(z)*(1+scale[:, None, None])+shift[:, None, None])


class GateFNet(nn.Module):
    """Shared invariant scalar backbone with explicit covariant vector bases.

    Condition order is [theta, alpha]; frozen train-only statistics are persistent
    buffers. Couplings, anisotropy and boundary conditions enter via CellGraph;
    the easy axis rotates jointly with spins. No crop/tile or material inference.
    """
    def __init__(self, width=32, blocks=3, condition_mean=(0., 0.), condition_std=(1., 1.)):
        super().__init__()
        mean, std = torch.tensor(condition_mean), torch.tensor(condition_std)
        if mean.shape != (2,) or std.shape != (2,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError('finite theta/alpha training statistics with positive scales required')
        self.register_buffer('condition_mean', mean)
        self.register_buffer('condition_std', std)
        self.input = nn.Linear(21, width)
        self.blocks = nn.ModuleList([ScalarBlock(width) for _ in range(blocks)])
        self.output = nn.Linear(width, 10)

    def forward(self, state, tau, initial, condition, time, graph, axis, *, tangent=True):
        b, f, a, nx, ny, xyz = state.shape
        if xyz != 3 or (a, nx, ny) != graph.shape or initial.shape != state[:, 0].shape:
            raise ValueError('full-cell path/initial/graph shape mismatch')
        if f < 2 or time.shape != (b, f) or not torch.isfinite(time).all() or (time[:, 1:] <= time[:, :-1]).any():
            raise ValueError('strictly increasing reduced physical time required')
        if condition.shape != (b, 2) or not torch.isfinite(condition).all() or (condition < 0).any():
            raise ValueError('condition must be finite nonnegative [theta, alpha]')
        if axis.shape != (b, 3) or not torch.allclose(axis.norm(dim=-1), torch.ones(b, device=axis.device, dtype=axis.dtype), atol=1e-6):
            raise ValueError('unit easy axis required')
        x = state.reshape(b, f, -1, 3)
        anchor = initial.reshape(b, 1, -1, 3).expand_as(x)
        degree = graph.aggregate(torch.ones_like(x[..., :1])).clamp_min(1)
        field = graph.field(x, axis)
        previous = torch.cat((x[:, :1], x[:, :-1]), 1)-x
        following = torch.cat((x[:, 1:], x[:, -1:]), 1)-x
        basis = torch.stack((anchor, field, graph.aggregate(x)/degree, previous, following), -2)
        inv = torch.cat(((x.unsqueeze(-2)*basis).sum(-1), basis.square().sum(-1)), -1)
        cond = (condition-self.condition_mean.to(condition))/self.condition_std.to(condition)
        t = time-time[:, :1]
        spacing = torch.cat((time[:, 1:2]-time[:, :1], time[:, 1:]-time[:, :-1]), 1)
        meta = torch.stack((tau[:, None].expand_as(t), t, time[:, -1:].sub(time[:, :1]).expand_as(t), spacing), -1)
        meta = torch.cat((meta, cond[:, None].expand(-1, f, -1)), -1)
        ones = torch.ones_like(x[..., :1])
        physics = torch.cat((ones*graph.kappa.to(x), degree,
                             graph.aggregate(ones, True),
                             graph.aggregate(ones, 'absolute'), ones*float(all(graph.periodic))), -1)
        h = torch.nn.functional.silu(self.input(torch.cat((inv, meta[:, :, None].expand(-1, -1, x.shape[2], -1), physics), -1)))
        for block in self.blocks:
            h = block(h, cond, graph, degree)
        direct, crossed = self.output(h).chunk(2, -1)
        v = (direct[..., None]*basis+crossed[..., None]*torch.linalg.cross(x.unsqueeze(-2), basis)).sum(-2)
        if tangent:
            v = tangent_project(x, v)
        v = torch.cat((torch.zeros_like(v[:, :1]), v[:, 1:]), 1)
        return v.reshape_as(state)


def reference_path(initial, frames, generator):
    """Full-support IID uniform sphere source; no correspondence to LLG noise."""
    if frames < 2:
        raise ValueError('at least two frames required')
    z = torch.randn((initial.shape[0], frames, *initial.shape[1:]), device=initial.device,
                    dtype=initial.dtype, generator=generator)
    z = z/z.norm(dim=-1, keepdim=True)
    return torch.cat((initial[:, None], z[:, 1:]), 1)


def path_interpolate(z, y, tau, generator):
    """Analytic shortest arcs, with explicit random cut-locus direction.

    Exact antipodes have no unique log. A projected isotropic random vector
    chooses an arc only when direction is numerically unresolved. Near-antipodal
    resolved arcs are retained. Both counts are returned, never filtered out.
    """
    if z.shape != y.shape or not torch.equal(z[:, 0], y[:, 0]):
        raise ValueError('matching paths with identical initial frame required')
    dot = (z*y).sum(-1, keepdim=True).clamp(-1, 1)
    q = y-dot*z
    length = q.norm(dim=-1, keepdim=True)
    eps = 16*torch.finfo(z.dtype).eps
    cut = (dot < 0) & (length <= eps)
    near = dot < -1+1e-5
    random = tangent_project(z, torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator))
    direction = torch.where(cut, random/random.norm(dim=-1, keepdim=True).clamp_min(eps), q/length.clamp_min(eps))
    angle = torch.atan2(length, dot)
    angle = torch.where(cut, torch.full_like(angle, math.pi), angle)
    s = tau.reshape(-1, *([1]*(z.ndim-1)))*angle
    x = s.cos()*z+s.sin()*direction
    v = angle*(-s.sin()*z+s.cos()*direction)
    x = torch.cat((y[:, :1], x[:, 1:]), 1)
    v = torch.cat((torch.zeros_like(v[:, :1]), v[:, 1:]), 1)
    return x, v, {'near_cut_count': int(near[:, 1:].sum()), 'unresolved_cut_count': int(cut[:, 1:].sum()),
                  'free_spin_count': y[:, 1:, ..., 0].numel()}


def matching_loss(net, y, condition, time, graph, axis, generator, *, kind='rfm'):
    initial = y[:, 0]
    tau = torch.rand(y.shape[0], device=y.device, dtype=y.dtype, generator=generator)
    diagnostics = {}
    if kind == 'deterministic':
        anchor = initial[:, None].expand_as(y)
        pred = sphere_exp(anchor, net(anchor, torch.ones_like(tau), initial, condition, time, graph, axis))
        error = pred-y
    elif kind == 'autoregressive':
        previous = y[:, :-1]
        # Teacher forcing: each frame is a separate two-frame conditional sample.
        loss = []
        for f in range(1, y.shape[1]):
            anchor = previous[:, f-1:f].expand(-1, 2, -1, -1, -1, -1)
            v = net(anchor, torch.ones_like(tau), previous[:, f-1], condition, time[:, f-1:f+1], graph, axis)[:, 1]
            mean = sphere_exp(previous[:, f-1], v)
            concentration = net.log_concentration.exp().clamp(1e-3, 100.)
            # vMF likelihood on S2, including the concentration-dependent normalizer.
            log_sinh = concentration+torch.log1p(-torch.exp(-2*concentration))-math.log(2)
            loss.append((log_sinh-torch.log(concentration)-concentration*(mean*y[:, f]).sum(-1)).mean())
        return torch.stack(loss).mean(), diagnostics
    elif kind in ('rfm', 'euclidean'):
        z = reference_path(initial, y.shape[1], generator)
        if kind == 'rfm':
            x, target, diagnostics = path_interpolate(z, y, tau, generator)
        else:
            # Ambient Gaussian flow baseline: no sphere projection during transport.
            z = torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=generator)
            z[:, 0] = initial
            t = tau.reshape(-1, *([1]*(y.ndim-1)))
            x, target = (1-t)*z+t*y, y-z
        error = net(x, tau, initial, condition, time, graph, axis, tangent=kind == 'rfm')-target
    else:
        raise ValueError('unknown model kind')
    return error[:, 1:].square().sum(-1).mean(), diagnostics


def build_model(kind, **kwargs):
    if kind not in ('rfm', 'deterministic', 'euclidean', 'autoregressive'):
        raise ValueError('unknown model kind')
    net = GateFNet(**kwargs)
    if kind == 'autoregressive':
        net.log_concentration = nn.Parameter(torch.tensor(0.))
    return net


@torch.no_grad()
def sample(net, initial, condition, time, graph, axis, generator, *, steps=32, kind='rfm', source=None):
    if steps < 1:
        raise ValueError('positive transport steps required')
    b, f = time.shape
    tau = torch.ones(b, device=initial.device, dtype=initial.dtype)
    if kind == 'deterministic':
        anchor = initial[:, None].expand(-1, f, -1, -1, -1, -1)
        prediction = sphere_exp(anchor, net(anchor, tau, initial, condition, time, graph, axis))
        return torch.cat((initial[:, None], prediction[:, 1:]), 1)
    if kind == 'autoregressive':
        frames = [initial]
        k = net.log_concentration.exp().clamp(1e-3, 100.)
        for index in range(1, f):
            prev = frames[-1]
            anchor = prev[:, None].expand(-1, 2, -1, -1, -1, -1)
            mean = sphere_exp(prev, net(anchor, tau, prev, condition, time[:, index-1:index+1], graph, axis)[:, 1])
            u = torch.rand(mean.shape[:-1]+(1,), device=mean.device, dtype=mean.dtype, generator=generator)
            w = 1+torch.log(u+(1-u)*torch.exp(-2*k))/k
            noise = tangent_project(mean, torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator))
            direction = noise/noise.norm(dim=-1, keepdim=True)
            frames.append(w*mean+(1-w.square()).clamp_min(0).sqrt()*direction)
        return torch.stack(frames, 1)
    if kind not in ('rfm', 'euclidean'):
        raise ValueError('unknown model kind')
    if source is not None:
        x = source.clone()
        if x.shape != (b, f, *initial.shape[1:]) or not torch.equal(x[:, 0], initial):
            raise ValueError('source must match shape and initial state')
    elif kind == 'rfm':
        x = reference_path(initial, f, generator)
    else:
        x = torch.randn((b, f, *initial.shape[1:]), device=initial.device, dtype=initial.dtype, generator=generator)
        x[:, 0] = initial
    for index in range(steps):
        v = net(x, tau*(index/steps), initial, condition, time, graph, axis, tangent=kind == 'rfm')
        x = sphere_exp(x, v/steps) if kind == 'rfm' else x+v/steps
        x[:, 0] = initial
    # Ambient flow is intentionally not silently normalized; report norm errors.
    return x
