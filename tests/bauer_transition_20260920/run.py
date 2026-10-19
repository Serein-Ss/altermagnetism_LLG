"""Same spatial backbone; compare joint 1ps paths with composed 0.1ps transitions."""
import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch.nn import functional as F
import yaml

from scripts.model.gate_f import path_interpolate
from tests.bauer_closed_loop_20260919.model import Net
from tests.bauer_closed_loop_20260919.run import config as old_config, records, reference, batch
from tests.bauer_closed_loop_20260919.metrics import compare
from tests.bauer_two_stage_20260919.common import allocation, dump, latent_seed
from tests.bauer_two_stage_20260919.models import base_path, sample_path
from tests.mdgen_path_refactor_20260918.data import PathWindows
from tests.bauer_publication_20260920.audit import mean_bias_squared, phase_statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = HERE.parent/'bauer_two_stage_20260919'


def config():
    return yaml.safe_load((HERE/'config.yaml').read_text())


def identity(index, cfg):
    return cfg['variants'][index//3], cfg['seeds'][index%3]


@torch.no_grad()
def rollout(net, initial, duration, seed, steps, variant, cfg):
    window = cfg['duration_ps'][variant]
    count = round(duration/window)
    if count < 1 or not np.isclose(count*window, duration):
        raise ValueError('Duration must be a positive integer multiple of model window')
    times = torch.arange(round(window/cfg['save_ps'])+1, device=initial.device)*cfg['save_ps']
    frames = [initial[:, None]]
    for k in range(count):
        x = sample_path(net, initial, times, latent_seed(seed, k), steps, 'anchored',
                        dict(anchored_sigma=.7, anchored_time_rho=.8))
        frames.append(x[:, 1:]); initial = x[:, -1]
    out = torch.cat(frames, 1)
    if not torch.isfinite(out).all() or float(abs(out.norm(dim=-1)-1).max()) > 2e-6:
        raise RuntimeError('Generated path integrity failed')
    return out


def temporal_errors(a, b):
    da, db = np.diff(a, axis=1), np.diff(b, axis=1)
    pa = (da*da).sum(-1).mean((0, 2))
    pb = (db*db).sum(-1).mean((0, 2))
    scale = max(float(pb.mean()), 1e-12)
    ca = (da[:, :-1]*da[:, 1:]).sum(-1).mean((0, 2))
    cb = (db[:, :-1]*db[:, 1:]).sum(-1).mean((0, 2))
    if len(ca) == 0:
        raise ValueError('At least three frames required for temporal diagnostics')
    return dict(increment_power_relative=float(np.sqrt(np.mean((pa-pb)**2))/scale),
                adjacent_dot_relative=float(np.sqrt(np.mean((ca-cb)**2))/scale))


def development(net, variant, cfg, smoke=False):
    oc = old_config(); selected = records(oc, 'dev')
    if smoke:
        selected = selected[:1]
    rows = []; net.eval()
    scales = dict(oc['margins'], **cfg['extra_score_scales'])
    for j, row in enumerate(selected):
        truth, times, s0 = reference(row, oc)
        s = F.normalize(torch.tensor(s0, device='cuda', dtype=torch.float32), dim=-1)
        s = s[None].repeat(cfg['branches'], 1, 1)
        x = rollout(net, s, 1., 2026092000+j, cfg['dev_flow_steps'], variant, cfg).cpu().numpy().astype(float)
        metrics = dict(compare(x, truth), **temporal_errors(x, truth))
        rows.append(dict(parent_id=row['parent_id'], L=row['L'], family=row['family'], **metrics))
    score = float(np.mean([r[k]/v for r in rows for k, v in scales.items()]))
    return dict(score=score, parents=rows, score_scales=scales,
                note='Development ranking heuristic, not a physical equivalence threshold')


def train(index, smoke=False):
    allocation('cuda'); torch.set_num_threads(1); cfg = config()
    variant, seed = identity(index, cfg)
    if smoke:
        cfg.update(updates=2, validate_every=2, batch=2, branches=2, dev_flow_steps=2)
    out = HERE/('smoke_runs' if smoke else 'runs')/f'{variant}_{seed}'
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    trng = torch.Generator(device='cuda').manual_seed(seed+2000)
    net = Net('C', cfg['width'], cfg['depth']).cuda()
    warm = PREVIOUS/'runs'/f'anchored_{seed}'/'best.pt'
    net.load_state_dict(torch.load(warm, map_location='cuda', weights_only=False)['model'])
    oc = old_config()
    datasets = [PathWindows([ROOT/oc['data']/f'L{n}'], 'train', cfg['duration_ps'][variant], stride_frames=1)
                for n in cfg['sizes']]
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['learning_rate'], weight_decay=1e-5)
    best = float('inf'); history = []; seen = 0; started = time.perf_counter()
    train_seconds = 0.
    for step in range(1, cfg['updates']+1):
        net.train(); ts = time.perf_counter()
        y, times = batch(datasets, rng, dict(oc, batch=cfg['batch']), 'cuda', 'C')
        seen += len(y)*(y.shape[1]-1)*y.shape[2]
        z = base_path(y[:, 0], y.shape[1], trng, 'anchored')
        tau = torch.rand(len(y), device='cuda', generator=trng)
        x, target, _ = path_interpolate(z, y, tau, trng)
        pred = net(x, tau, y[:, 0], times, cfg['temperature_K'])
        loss = (pred[:, 1:]-target[:, 1:]).square().sum(-1).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        opt.zero_grad(set_to_none=True); loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
        if not torch.isfinite(grad):
            raise RuntimeError('Nonfinite gradient')
        opt.step(); torch.cuda.synchronize(); train_seconds += time.perf_counter()-ts
        if step % 100 == 0:
            print(variant, seed, step, float(loss), flush=True)
        if step % cfg['validate_every'] == 0:
            dev = development(net, variant, cfg, smoke)
            dump(out/f'dev_{step}.json', dev)
            history.append(dict(step=step, loss=float(loss), dev_score=dev['score']))
            state = dict(model=net.state_dict(), variant=variant, seed=seed, step=step,
                         config=cfg, dev_score=dev['score'], warm_start=str(warm.relative_to(ROOT)))
            if dev['score'] < best:
                best = dev['score']; torch.save(state, out/'best.pt')
            torch.save(dict(state, optimizer=opt.state_dict(), numpy_rng=rng.bit_generator.state,
                            latent_rng=trng.get_state(), torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state()), out/'last.pt')
            dump(out/'history.json', history)
    dump(out/'complete.json', dict(variant=variant, seed=seed, best_dev_score=best,
         updates=cfg['updates'], future_spins_seen=seen, train_seconds=train_seconds,
         total_seconds=time.perf_counter()-started, gpu=torch.cuda.get_device_name(), smoke=smoke,
         scientific_certification=False))


def evaluate(index):
    allocation('cuda'); torch.set_num_threads(1); cfg = config()
    variant, seed = identity(index, cfg); name = f'{variant}_{seed}'
    ck = torch.load(HERE/'runs'/name/'best.pt', map_location='cuda', weights_only=False)
    net = Net('C', cfg['width'], cfg['depth']).cuda(); net.load_state_dict(ck['model']); net.eval()
    out = HERE/'evaluation'/name; out.mkdir(parents=True, exist_ok=False)
    results = []
    for n in cfg['evaluation_sizes']:
        with h5py.File(out/f'L{n}.h5', 'x') as dest:
            dest.attrs['complete'] = False; dest.attrs['model'] = name
            dest.attrs['config_json'] = json.dumps(cfg)
            dest.attrs['checkpoint_step'] = ck['step']
            for j in range(6):
                end = round(cfg['evaluation_duration_ps']/cfg['save_ps'])+1
                with h5py.File(PREVIOUS/'data'/f'L{n}'/f'test_future_{j}.h5') as h:
                    truth = h['spins'][:, :end].astype(float); s0 = h['initial'][:]
                    times = h['time_ps'][:end]
                s = F.normalize(torch.tensor(s0, device='cuda', dtype=torch.float32), dim=-1)
                s = s[None].repeat(cfg['branches'], 1, 1)
                pathseed = 260920000+n*100+j
                torch.cuda.synchronize(); start = time.perf_counter()
                x = rollout(net, s, cfg['evaluation_duration_ps'], pathseed,
                            cfg['evaluation_flow_steps'], variant, cfg).cpu().numpy()
                torch.cuda.synchronize(); elapsed = time.perf_counter()-start
                d = dest.create_dataset(f'parent_{j}', data=x, compression='lzf')
                d.attrs['seed'] = pathseed
                for horizon in [1., cfg['evaluation_duration_ps']]:
                    frames = round(horizon/cfg['save_ps'])+1
                    a, b = x[:, :frames].astype(float), truth[:, :frames]
                    results.append(dict(L=n, parent=j, horizon_ps=horizon,
                        raw=compare(a, b[:16]), reference_floor=compare(b[16:], b[:16]),
                        temporal=temporal_errors(a, b),
                        conditional_increment=mean_bias_squared(np.diff(a, axis=1), np.diff(b, axis=1)),
                        model_phase=phase_statistics(a), reference_phase=phase_statistics(b)))
                print(name, n, j, elapsed, flush=True)
                results[-1]['generation_seconds_full_horizon'] = elapsed
            dest.create_dataset('time_ps', data=times); dest.attrs['complete'] = True
        dump(out/'metrics.json', dict(model=name, selected_step=ck['step'], rows=results,
             all_seeds_reported=True, scientific_certification=False,
             note='Previously inspected test parents, diagnostic only; not a fresh final holdout'))
    dump(out/'complete.json', dict(model=name, cases=len(results), scientific_certification=False))


def report():
    import html
    rows = []
    for index in range(6):
        variant, seed = identity(index, config()); name = f'{variant}_{seed}'
        item = dict(model=name)
        for stage, folder in [('training', HERE/'runs'/name), ('evaluation', HERE/'evaluation'/name)]:
            p = folder/'complete.json'
            item[stage] = json.loads(p.read_text()) if p.exists() else {'status': 'NOT_COMPLETED'}
        rows.append(item)
    out = HERE/'results'; out.mkdir(exist_ok=True)
    dump(out/'status.json', dict(models=rows, scientific_certification=False))
    body = '<!doctype html><meta charset="utf-8"><title>Bauer 时间因子化受控实验</title>'
    body += '<h1>Bauer 时间因子化受控实验</h1><p>比较相同空间骨干的 1 ps 整段与 0.1 ps 转移组合，各三个种子。完成运行不代表通过物理验证。</p>'
    body += '<p>仅使用开发集选检查点。现有测试样本已经用于诊断，不能再宣称全新盲测。所有模型结果均保留，不按测试表现挑选。</p>'
    for r in rows:
        body += '<h2>'+r['model']+'</h2><pre>'+html.escape(json.dumps(r, ensure_ascii=False, indent=2))+'</pre>'
        if 'status' not in r['evaluation']:
            body += f'<a href="../evaluation/{r["model"]}/metrics.json">完整诊断指标</a>'
    (out/'index.html').write_text(body)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['train', 'evaluate', 'report'])
    p.add_argument('--index', type=int, default=0, choices=range(6)); p.add_argument('--smoke', action='store_true')
    p.add_argument('--device', choices=['cuda']); a = p.parse_args()
    if a.phase != 'report' and a.device != 'cuda':
        p.error('GPU phases require --device cuda')
    if a.phase == 'train': train(a.index, a.smoke)
    elif a.phase == 'evaluate': evaluate(a.index)
    else: report()
