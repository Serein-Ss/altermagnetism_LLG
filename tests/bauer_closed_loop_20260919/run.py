"""Bounded train/dev/select/test experiment. All artifacts stay in tests/."""
import argparse
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch.nn import functional as F
import yaml
from scripts.model.gate_f import path_interpolate
from scripts.model.sphere import sphere_exp, tangent_project
from tests.mdgen_path_refactor_20260918.data import PathWindows
from .model import Net, generate
from .metrics import compare, summary

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def config():
    return yaml.safe_load((HERE/'config.yaml').read_text())


def records(cfg, split):
    result = []
    for length in cfg['sizes']:
        root = ROOT/cfg['data']/f'L{length}'
        for row in json.loads((root/'manifest.json').read_text()):
            if row['split'] == split and row['arm'] == 'base':
                result.append(dict(row, L=length, path=root/row['file']))
    return result


def reference(row, cfg):
    frames = round(cfg['duration_ps']/cfg['save_ps'])+1
    with h5py.File(row['path']) as h:
        s = h['spins'][:, :frames].astype('f8')
        times = h['time_ps'][:frames]
        initial = h['initial'][:]
    return s, times, initial


def evaluate(net, cfg, split, device, steps, seed, out=None, subset=None):
    rows = []
    selected = records(cfg, split)
    if subset:
        selected = selected[:subset]
    started = time.perf_counter()
    net.eval()
    for j, row in enumerate(selected):
        truth, times, initial = reference(row, cfg)
        init = F.normalize(torch.tensor(initial, dtype=torch.float32, device=device), dim=-1)
        init = init[None].expand(cfg['branches'], -1, -1)
        physical = torch.tensor(times-times[0], device=device, dtype=torch.float32)
        pathseed = seed+1000000*j
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        pred = generate(net, init, physical, cfg['temperature_K'], pathseed, steps)
        if device == 'cuda':
            torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        s = pred.cpu().numpy()
        norm = float(abs(np.linalg.norm(s, axis=-1)-1).max())
        anchor = float(abs(s[:, 0]-initial).max())
        if not np.isfinite(s).all() or norm > 2e-6 or anchor > 2e-6:
            raise RuntimeError('Generated trajectory integrity failure')
        item = dict(parent_id=row['parent_id'], family=row['family'], L=row['L'],
                    **compare(s.astype('f8'), truth), max_norm_error=norm, initial_error=anchor,
                    generation_seconds=seconds, seed=pathseed)
        rows.append(item)
        if out:
            out.mkdir(exist_ok=True, parents=True)
            np.savez_compressed(out/f'{row["parent_id"]}.npz', generated=s, reference=truth.astype('f4'),
                                time_ps=times, seed=pathseed, flow_steps=steps, initial=initial,
                                branch_rng='one seeded generator, separate sequential draws per branch')
    stats = summary(rows, cfg['margins'], cfg['bootstrap_repeats'])
    score = float(np.mean([r['mean']/r['margin'] for r in stats.values()]))
    return dict(split=split, steps=steps, score=score, parents=rows, statistics=stats,
                seconds=time.perf_counter()-started,
                score_definition='equal-weight mean of metric/margin, averaged over balanced parents')


def batch(datasets, rng, cfg, device, kind):
    ds = datasets[int(rng.integers(len(datasets)))]
    samples = []
    for _ in range(cfg['batch']):
        i = int(rng.integers(len(ds)))
        if rng.random() < .5:
            # Half exposure from actual branch anchors; half uniformly selected windows.
            g = int(np.searchsorted(ds.ends, i, side='right'))
            start = ds.ends[g-1] if g else 0
            nstarts = len(ds.records[g][2])
            i = start+((i-start)//nstarts)*nstarts
        samples.append(ds[i]['spins'])
    y = F.normalize(torch.tensor(np.stack(samples), device=device), dim=-1)
    t = torch.arange(y.shape[1], device=device)*cfg['save_ps']
    if kind == 'A':
        offsets = rng.integers(y.shape[1]-1, size=len(y))
        y = torch.stack([y[j, k:k+2] for j, k in enumerate(offsets)])
        t = t[:2]
    return y, t


def loss_for(net, y, times, cfg, rng):
    if net.kind == 'D':
        anchor = y[:, :1].expand_as(y)
        pred = sphere_exp(anchor, net(anchor, y.new_ones(len(y)), y[:, 0], times, cfg['temperature_K']))
        return (pred[:, 1:]-y[:, 1:]).square().sum(-1).mean()
    z = F.normalize(torch.randn(y.shape, device=y.device, generator=rng), dim=-1)
    z[:, 0] = y[:, 0]
    tau = torch.rand(len(y), device=y.device, generator=rng)
    x, v, _ = path_interpolate(z, y, tau, rng)
    pred = net(x, tau, y[:, 0], times, cfg['temperature_K'])
    return (pred[:, 1:]-v[:, 1:]).square().sum(-1).mean()


def run(kind, seed, device, smoke=False):
    cfg = config()
    if smoke:
        cfg.update(updates=3, validate_every=3, batch=2, dev_flow_steps=2, bootstrap_repeats=20)
    if device != 'cuda' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Training requires Slurm --device cuda')
    audit = json.loads((HERE/'audit.json').read_text())
    if not audit['integrity_pass']:
        raise RuntimeError('Reference data integrity gate failed')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    trng = torch.Generator(device=device).manual_seed(seed+999)
    out = HERE/('smoke_runs' if smoke else 'runs')/f'{kind}_{seed}'
    out.mkdir(parents=True, exist_ok=False)
    datasets = [PathWindows([ROOT/cfg['data']/f'L{n}'], 'train', cfg['duration_ps'], stride_frames=10) for n in cfg['sizes']]
    net = Net(kind, cfg['width'], cfg['depth']).to(device)
    net.temperature_mean.fill_(audit['train_temperature_mean'])
    net.temperature_scale.fill_(audit['train_temperature_scale'])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['learning_rate'], weight_decay=1e-5)
    best = float('inf')
    history = []
    started = time.perf_counter()
    train_seconds = 0.
    future_spins_seen = 0
    for step in range(1, cfg['updates']+1):
        net.train()
        ts = time.perf_counter()
        y, physical = batch(datasets, rng, cfg, device, kind)
        future_spins_seen += len(y)*(y.shape[1]-1)*y.shape[2]
        opt.zero_grad(set_to_none=True)
        loss = loss_for(net, y, physical, cfg, trng)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
        if not torch.isfinite(grad):
            raise RuntimeError('Nonfinite gradient')
        opt.step()
        train_seconds += time.perf_counter()-ts
        history.append(dict(step=step, loss=float(loss)))
        if step % 100 == 0:
            print(json.dumps(dict(kind=kind, seed=seed, **history[-1])), flush=True)
        if step % cfg['validate_every'] == 0 or step == cfg['updates']:
            dev = evaluate(net, cfg, 'dev', device, cfg['dev_flow_steps'], 73000, subset=2 if smoke else None)
            dump(out/f'dev_{step}.json', dev)
            if dev['score'] < best:
                best = dev['score']
                torch.save(dict(model=net.state_dict(), kind=kind, seed=seed, step=step, config=cfg,
                                selection_score=best), out/'best.pt')
            torch.save(dict(model=net.state_dict(), optimizer=opt.state_dict(), step=step,
                            config=cfg, torch_rng=trng.get_state(), numpy_rng=rng.bit_generator.state), out/'last.pt')
            dump(out/'history.json', history)
            print(json.dumps(dict(kind=kind, seed=seed, step=step, dev_score=dev['score'])), flush=True)
    checkpoint = torch.load(out/'best.pt', map_location=device, weights_only=False)
    net.load_state_dict(checkpoint['model'])
    dump(out/'selection.json', dict(selected_step=checkpoint['step'], dev_score=best,
                                    parameters=sum(p.numel() for p in net.parameters()),
                                    updates=cfg['updates'], windows_seen=cfg['updates']*cfg['batch'],
                                    target_future_spins_seen=future_spins_seen,
                                    exposure_note='same updates/windows, not compute or target-frame matched; compare recorded costs',
                                    training_and_io_seconds=train_seconds, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                                    gpu=torch.cuda.get_device_name(), config=cfg))
    if not smoke:
        # Freeze selection above before the first test-future read.
        for steps in ([cfg['test_flow_steps'][-1]] if kind == 'D' else cfg['test_flow_steps']):
            report = evaluate(net, cfg, 'test', device, steps, 83000, out/'generated'/f'steps_{steps}')
            report['model_status'] = 'PILOT_METRICS_ONLY_REFERENCE_AND_POWER_LIMITS_APPLY'
            dump(out/f'test_{steps}.json', report)
        if kind != 'D':
            paired = []
            for row in records(cfg, 'test'):
                with np.load(out/'generated'/'steps_128'/f'{row["parent_id"]}.npz') as data:
                    fine = data['generated'].astype('f8')
                for n in cfg['test_flow_steps'][:-1]:
                    with np.load(out/'generated'/f'steps_{n}'/f'{row["parent_id"]}.npz') as data:
                        coarse = data['generated'].astype('f8')
                    paired.append(dict(parent_id=row['parent_id'], L=row['L'], coarse_steps=n,
                                       fine_steps=128, paired_spin_rmse=float(np.sqrt(np.mean((coarse-fine)**2)))))
            dump(out/'sampler_paired.json', paired)
    dump(out/'complete.json', dict(status='SOFTWARE_COMPLETE_NOT_AUTOMATIC_PHYSICS_PASS', smoke=smoke,
                                   kind=kind, seed=seed, total_seconds=time.perf_counter()-started))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--device', required=True, choices=['cuda'])
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    cfg = config()
    if args.smoke:
        for kind in cfg['models']:
            run(kind, cfg['seeds'][0], args.device, True)
    else:
        run(cfg['models'][args.index//len(cfg['seeds'])], cfg['seeds'][args.index % len(cfg['seeds'])], args.device)


if __name__ == '__main__':
    main()
