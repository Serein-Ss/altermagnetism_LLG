import json
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.nn import functional as F
from scripts.core.cell_hamiltonian import CellHamiltonian
from scripts.model.sphere import sphere_exp
from tests.spin_graph_20260920.model import SpinGraphNet

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def cfg():
    return yaml.safe_load((HERE/'config.yaml').read_text())


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def energy(s, kappa=.1):
    """H/(N J), open chain, each bond counted once; torch or numpy."""
    return -(s[..., :-1, :]*s[..., 1:, :]).sum(-1).sum(-1)/s.shape[-2]-kappa*(s[..., 2]**2).mean(-1)


class Model(SpinGraphNet):
    def __init__(self, config):
        super().__init__((False,), config['width'], config['depth'])
        self.register_buffer('temperature_mean', torch.tensor(config['temperature_K']))
        self.register_buffer('temperature_scale', torch.tensor(1.))
        self.config = config

    def velocity(self, x, tau, initial, times, temperature=None, frame=None):
        temp = self.temperature_mean if temperature is None else torch.as_tensor(temperature, device=x.device, dtype=x.dtype)
        if not torch.allclose(temp, self.temperature_mean.expand_as(temp), atol=1e-4, rtol=0):
            raise ValueError('single-temperature package: unsupported temperature')
        token = (temp-self.temperature_mean)/self.temperature_scale
        cell = CellHamiltonian((x.shape[2],), 1, [(0, 0, (1,), 1.)], periodic=(False,))
        frame = torch.eye(3, device=x.device, dtype=x.dtype) if frame is None else frame
        return self(x.unsqueeze(-2), tau, initial.unsqueeze(-2), cell, frame, token, times).squeeze(-2)


@torch.no_grad()
def generate(net, initial, times, generator, steps, source=None):
    if steps < 1:
        raise ValueError('positive flow steps required')
    x = F.normalize(torch.randn((len(initial), len(times), *initial.shape[1:]),
                                device=initial.device, dtype=initial.dtype, generator=generator), dim=-1) if source is None else source.clone()
    x[:, 0] = initial
    for i in range(steps):
        velocity = net.velocity(x, x.new_full((len(x),), i/steps), initial, times)
        x = sphere_exp(x, velocity/steps)
        x[:, 0] = initial
    return x


@torch.no_grad()
def compose(net, initial, total_ps, generator, steps, elapsed_ps=0.):
    """Software-only block interface. No long-time accuracy claim."""
    config = net.config
    count = round(total_ps/config['save_ps'])
    if total_ps < 0 or not np.isclose(count*config['save_ps'], total_ps):
        raise ValueError('duration must be on physical output grid')
    block = round(config['duration_ps']/config['save_ps'])
    times = torch.arange(block+1, device=initial.device, dtype=initial.dtype)*config['save_ps']
    frames = [initial[:, None]]
    state = initial
    remaining = count
    while remaining:
        path = generate(net, state, times, generator, steps)
        take = min(remaining, block)
        frames.append(path[:, 1:take+1])
        state = path[:, take]
        remaining -= take
    # Resume only at full-block boundaries: partial block needs its unused tail.
    return torch.cat(frames, 1), dict(rng=generator.get_state(), state=state,
        elapsed_ps=elapsed_ps+total_ps, resumable=count % block == 0)


def observable_samples(s):
    """Whole-trajectory observables [branch,...]; sites never become replicates."""
    s = np.asarray(s, dtype=np.float64)
    increment = np.diff(s, axis=1)
    return dict(mz=s[..., 2].mean(-1), energy=energy(s),
        temporal=(s*s[:, :1]).sum(-1).mean(-1),
        temporal_lag=np.stack([(s[:, lag:]*s[:, :-lag]).sum(-1).mean((1,2)) for lag in (1,2,5)],-1),
        spatial=np.stack([(s[:, :, :-lag]*s[:, :, lag:]).sum(-1).mean(-1) for lag in (1, 2, 4)], -1),
        increment=increment)


def statistics(s):
    v = observable_samples(s)
    d = v['increment']
    centered = d-d.mean(0)
    return dict(mz_mean=v['mz'].mean(0), mz_var=v['mz'].var(0, ddof=1),
        energy_mean=v['energy'].mean(0), energy_var=v['energy'].var(0, ddof=1),
        temporal=v['temporal'].mean(0), temporal_lag=v['temporal_lag'].mean(0), spatial=v['spatial'].mean(0),
        increment_mean=d.mean(0),
        increment_cov=np.einsum('btni,btnj->tnij', centered, centered)/(len(s)-1))


def comparison(generated, reference, repeats=0, seed=19):
    a, b = statistics(generated), statistics(reference)
    result = {k: dict(rmse=float(np.sqrt(np.mean((a[k]-b[k])**2))),
                      mean_difference=float((a[k]-b[k]).mean())) for k in a}
    if repeats:
        rng = np.random.default_rng(seed)
        boot = {k: [] for k in a}
        for _ in range(repeats):
            # Fixed parent, independent whole branches in each ensemble.
            ga = statistics(generated[rng.integers(len(generated), size=len(generated))])
            rb = statistics(reference[rng.integers(len(reference), size=len(reference))])
            for k in a:
                boot[k].append(float((ga[k]-rb[k]).mean()))
        for k in a:
            result[k]['mean_difference_ci95'] = np.quantile(boot[k], [.025, .975]).tolist()
    return result
