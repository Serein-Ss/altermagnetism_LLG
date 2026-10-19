"""New-library full-domain pilot using the frozen local multidimensional prototype.

This tests the data/model interface, not a final multiscale model or certification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import h5py
import numpy as np
import torch
import yaml
from torch.nn import functional as F
from tests.path_smoke_20260917.model import PathNet, generate
from tests.path_smoke_20260917.run import contracts
from tests.literature_preflight_20260917.models import build
from tests.bauer_two_stage_20260919.common import allocation, dump
from scripts.model.gate_f import path_interpolate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT/'tests/miltat_library_20260920/data'
SHAPES = [(64, 38), (96, 56), (128, 74)]


def read_window(row, branch, start, frames=11):
    with h5py.File(DATA/row['file']) as h:
        assert h.attrs['complete'] and h.attrs['split'] == row['split']
        x = h['spins'][branch, start:start+frames]
        t = h['time_ps'][start:start+frames]
    assert len(x) == frames and np.isfinite(x).all()
    assert np.allclose(np.linalg.norm(x, axis=-1), 1, atol=1e-6)
    assert np.allclose(np.diff(t), .1)
    return x, t-t[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=int, choices=range(3), required=True)
    p.add_argument('--device', choices=['cuda'], required=True)
    p.add_argument('--benchmark', action='store_true')
    p.add_argument('--updates', type=int, default=200)
    a = p.parse_args(); allocation(a.device); torch.set_num_threads(1)
    torch.manual_seed(260920); rng = np.random.default_rng(260920)
    out = HERE/('benchmark' if a.benchmark else 'miltat')/str(a.index)
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((DATA/'manifest.json').read_text())
    rows = [r for r in manifest['parents'] if tuple(r['cell_shape']) == SHAPES[a.index]]
    train = [r for r in rows if r['split'] == 'train']
    dev = [r for r in rows if r['split'] == 'dev']
    assert len(train) == 12 and len(dev) == 4
    assert not {r['preparation_group'] for r in train} & {r['preparation_group'] for r in dev}
    cfg = yaml.safe_load((ROOT/'tests/literature_preflight_20260917/config.yaml').read_text())
    cfg['systems']['miltat']['physics'].update(temperature_K=12.5, alpha=.5)
    cell, reduced, _ = build(cfg, 'miltat', shape=SHAPES[a.index])
    net = PathNet().cuda(); frame = torch.eye(3, device='cuda')
    opt = torch.optim.Adam(net.parameters(), lr=.001)
    generator = torch.Generator(device='cuda').manual_seed(260921)
    times = torch.arange(11, device='cuda')*.1
    losses = []; step_seconds = []; updates = 8 if a.benchmark else a.updates
    start_all = time.perf_counter()
    for step in range(updates):
        torch.cuda.synchronize(); start = time.perf_counter()
        row = train[int(rng.integers(len(train)))]
        raw, _ = read_window(row, int(rng.integers(8)), int(rng.integers(91)))
        y = F.normalize(torch.tensor(raw[None], device='cuda'), dim=-1)
        z = F.normalize(torch.randn(y.shape, device='cuda', generator=generator), dim=-1)
        z[:, 0] = y[:, 0]
        tau = torch.rand(1, device='cuda', generator=generator)
        state, target, _ = path_interpolate(z, y, tau, generator)
        prediction = net(state, tau, y[:, 0], cell, frame, reduced['theta'], times)
        loss = (prediction[:, 1:]-target[:, 1:]).square().sum(-1).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in net.parameters() if v.grad is not None)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.); opt.step()
        torch.cuda.synchronize(); step_seconds.append(time.perf_counter()-start); losses.append(float(loss))
    if a.benchmark:
        dump(out/'report.json', dict(step_seconds=step_seconds, median_warm_seconds=float(np.median(step_seconds[2:])),
             peak_allocated_bytes=torch.cuda.max_memory_allocated(), device=torch.cuda.get_device_name(),
             status='BENCHMARK_ONLY', shape=SHAPES[a.index]))
        return
    net.eval(); checks = contracts(net, cfg, 'miltat', 'cuda')
    norm = 0.; anchor = 0.; finite = True
    with h5py.File(out/'generated.h5', 'x') as h:
        h.attrs['complete'] = False
        for j, row in enumerate(dev):
            with h5py.File(DATA/row['file']) as source:
                ref = source['spins'][:, :11]
                positions = source['positions_nm'][:]
            initial = F.normalize(torch.tensor(ref[0, 0][None], device='cuda'), dim=-1)
            generated = []
            for branch in range(8):
                generated.append(generate(net, initial, cell, frame, reduced['theta'], times,
                                          seed=270000+j*100+branch, steps=32).cpu().numpy()[0])
            x = np.stack(generated)
            g = h.create_group(row['parent_id'])
            for name, val in [('generated', x), ('reference', ref), ('positions_nm', positions), ('time_ps', np.arange(11)*.1)]:
                g.create_dataset(name, data=val, compression='lzf')
            norm = max(norm, float(abs(np.linalg.norm(x, axis=-1)-1).max()))
            anchor = max(anchor, float(abs(x[:, 0]-ref[:, 0]).max()))
            finite &= bool(np.isfinite(x).all())
        h.attrs['complete'] = True
    torch.save(dict(model=net.state_dict(), width=24, shape=SHAPES[a.index], updates=updates), out/'last.pt')
    dump(out/'report.json', dict(shape=SHAPES[a.index], loss=losses, updates=updates, checks=checks,
         norm_error=norm, initial_error=anchor, finite=finite, seconds=time.perf_counter()-start_all,
         step_seconds_median=float(np.median(step_seconds[2:])), manifest_sha256=hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest(),
         training_parents=[r['parent_id'] for r in train], development_parents=[r['parent_id'] for r in dev],
         final_test_used=False, scientific_certification=False,
         architecture='Frozen local two-hop PathNet prototype; no temporal attention or multiscale branch; not final architecture',
         geometry='2D triangular monolayer with two coplanar basis sites per rectangular cell, NOT two layers'))
    assert checks['passed'] and finite and norm < 2e-6 and anchor < 2e-6


if __name__ == '__main__': main()
