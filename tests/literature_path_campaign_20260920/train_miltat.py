"""Paired local/full space-time pilots on full domains; same data and FM objective."""
import argparse
import json
import os
import time
from pathlib import Path
import h5py
import numpy as np
import torch
import yaml
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.path_smoke_20260917.model import PathNet, generate
from tests.path_smoke_20260917.run import contracts
from tests.literature_preflight_20260917.models import build
from tests.cross_system_audit_20260920.miltat import DATA, SHAPES, read_window
from tests.bauer_two_stage_20260919.common import allocation, dump
from .models import SpaceTimeNet

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def physical_config():
    cfg = yaml.safe_load((ROOT/'tests/literature_preflight_20260917/config.yaml').read_text())
    cfg['systems']['miltat']['physics'].update(temperature_K=12.5, alpha=.5)
    return cfg


@torch.no_grad()
def evaluate(net, rows, out, branches=8, steps=32):
    """Per-parent conditional evaluation, not independent per-site replicates."""
    net.eval(); metrics = []; frame = torch.eye(3, device='cuda'); times = torch.arange(11, device='cuda')*.1
    with h5py.File(out, 'x') as h:
        h.attrs['complete'] = False
        for j, row in enumerate(rows):
            cell, reduced, _ = build(physical_config(), 'miltat', shape=row['cell_shape'])
            with h5py.File(DATA/row['file']) as source:
                ref = source['spins'][:, :11]; positions = source['positions_nm'][:]
            initial = F.normalize(torch.tensor(ref[:1, 0], device='cuda'), dim=-1)
            x = np.stack([generate(net, initial, cell, frame, reduced['theta'], times,
                         seed=270000+j*100+b, steps=steps).cpu().numpy()[0] for b in range(branches)])
            group = h.create_group(row['parent_id']); group.attrs['split'] = row['split']
            for key, value in [('generated', x), ('reference', ref), ('time_ps', np.arange(11)*.1), ('positions_nm', positions)]:
                group.create_dataset(key, data=value, compression='lzf')
            dg, dr = np.diff(x, axis=1), np.diff(ref, axis=1)
            mean_mse = float(((x.mean(0)-ref.mean(0))**2).mean())
            power_ratio = float((dg**2).mean()/(dr**2).mean())
            metrics.append(dict(parent_id=row['parent_id'], shape=row['cell_shape'], mean_mse=mean_mse,
                                power_ratio=power_ratio, norm_error=float(abs(np.linalg.norm(x, axis=-1)-1).max())))
        h.attrs['complete'] = True
    # Fixed development heuristic; not a physical acceptance threshold.
    return dict(rows=metrics, score=float(np.mean([r['mean_mse']/.01+abs(np.log(r['power_ratio'])) for r in metrics])),
                selection='Mean field MSE / .01 plus absolute log increment-power ratio; development only',
                scientific_certification=False)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--device', required=True, choices=['cuda'])
    p.add_argument('--index', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 3)))
    p.add_argument('--benchmark', action='store_true'); a = p.parse_args()
    allocation(a.device); torch.set_num_threads(1)
    if a.index not in range(6): raise ValueError('Expected index 0..5')
    kind = ['local', 'full'][a.index//3]; seed = [11,22,33][a.index%3]
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    trng = torch.Generator(device='cuda').manual_seed(seed+2000)
    out = HERE/('benchmark_miltat' if a.benchmark else 'miltat')/f'{kind}_{seed}'
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((DATA/'manifest.json').read_text())['parents']
    train = [[r for r in manifest if r['split']=='train' and tuple(r['cell_shape'])==shape] for shape in SHAPES]
    dev = [r for r in manifest if r['split']=='dev']
    assert all(len(rows)==12 for rows in train) and len(dev)==12
    assert not {r['preparation_group'] for rows in train for r in rows} & {r['preparation_group'] for r in dev}
    net = (PathNet() if kind=='local' else SpaceTimeNet()).cuda()
    checks = contracts(net, physical_config(), 'miltat', 'cuda')
    dump(out/'contracts.json', checks)
    if not checks['passed']: raise RuntimeError('Architecture contract failed')
    cells = [build(physical_config(), 'miltat', shape=shape) for shape in SHAPES]
    opt = torch.optim.Adam(net.parameters(), lr=.001); frame = torch.eye(3, device='cuda')
    times = torch.arange(11, device='cuda')*.1; seconds = []; history = []; best = float('inf')
    started = time.perf_counter()
    for step in range(1, (8 if a.benchmark else 1500)+1):
        net.train(); torch.cuda.synchronize(); ts = time.perf_counter()
        i = 2 if a.benchmark else int(rng.integers(3)); rows = train[i]
        row = rows[int(rng.integers(len(rows)))]; cell, reduced, _ = cells[i]
        raw, _ = read_window(row, int(rng.integers(8)), int(rng.integers(91)))
        y = F.normalize(torch.tensor(raw[None], device='cuda'), dim=-1)
        z = F.normalize(torch.randn(y.shape, device='cuda', generator=trng), dim=-1); z[:, 0] = y[:, 0]
        tau = torch.rand(1, device='cuda', generator=trng)
        state, target, _ = path_interpolate(z, y, tau, trng)
        loss = (net(state, tau, y[:, 0], cell, frame, reduced['theta'], times)[:, 1:]-target[:, 1:]).square().sum(-1).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
        if not torch.isfinite(loss) or not torch.isfinite(grad): raise RuntimeError('Nonfinite optimization')
        opt.step(); torch.cuda.synchronize(); seconds.append(time.perf_counter()-ts)
        if step % 100 == 0: print(kind, seed, step, float(loss), flush=True)
        if not a.benchmark and step % 500 == 0:
            # Fixed two-branch screening saves memory/time; full eight-branch assessment after selection.
            result = evaluate(net, dev, out/f'dev_{step}.h5', branches=2)
            dump(out/f'dev_{step}.json', result); history.append(dict(step=step, loss=float(loss), score=result['score']))
            state = dict(model=net.state_dict(), kind=kind, seed=seed, step=step, dev_score=result['score'])
            if result['score'] < best: best = result['score']; torch.save(state, out/'best.pt')
            torch.save(state, out/'last.pt'); dump(out/'history.json', history)
    if a.benchmark:
        ts = time.perf_counter(); evaluate(net, dev[-1:], out/'dev.h5', branches=2); torch.cuda.synchronize()
        eval_seconds = time.perf_counter()-ts
        dump(out/'report.json', dict(warm_step_seconds=float(np.median(seconds[2:])),
             one_parent_validation_seconds=eval_seconds, peak_memory_bytes=torch.cuda.max_memory_allocated(),
             estimated_1500_updates_plus_3_dev_seconds=float(np.median(seconds[2:]))*1500+eval_seconds*12*3))
    else:
        dump(out/'complete.json', dict(best_dev_score=best, seconds=time.perf_counter()-started,
             heldout_test_used=False, scientific_certification=False, training_shapes=SHAPES))


if __name__ == '__main__': main()
