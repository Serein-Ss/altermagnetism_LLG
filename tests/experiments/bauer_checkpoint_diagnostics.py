"""Paired, development-only diagnostics of frozen Bauer RFM checkpoints.

Raw trajectories go to data, derived statistics to assets, execution logs to logs.
No fitting, test-set selection or physical-generalization certificate is performed.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from scripts.model.gate_f import build_model, reference_path, sample, matching_loss
from tests.experiments.bauer_window import normalized_window, sample_chunks
from scripts.workflow.bauer_campaign import graph, save, digest, event_rows, initial_states


def initial_families(pool, count, reference_count):
    """Group exact saved initial states, retaining first-occurrence order."""
    flat = pool[:, 0].reshape(len(pool), -1)
    _, first = np.unique(flat, axis=0, return_index=True)
    first = np.sort(first)
    if len(first) < count:
        raise ValueError('not enough distinct initial states')
    groups = []
    for index in first[:count]:
        group = np.flatnonzero(np.all(flat == flat[index], axis=1))
        if len(group) < reference_count:
            raise ValueError('not enough real trajectories for matched initial family')
        groups.append(group[:reference_count])
    return first[:count], np.stack(groups)


def sampling_benchmark(net, initial, condition, graph, axis, cfg, device):
    """Warm, synchronized sampling only; excludes load, I/O and Slurm waiting."""
    rows = []
    for batch in (1, cfg['initials'] * cfg['latent_seeds']):
        x = initial[:1].expand(batch, -1, -1, -1, -1).contiguous()
        c = condition[:1].expand(batch, -1)
        a = axis[:1].expand(batch, -1)
        t = normalized_window(x, cfg['frames'], cfg['physical_window'])
        for steps in cfg['integration_steps']:
            durations = []
            for repeat in range(cfg['benchmark_repeats'] + 1):
                gen = torch.Generator(device=device).manual_seed(914000 + repeat)
                if device == 'cuda':
                    torch.cuda.synchronize()
                started = time.perf_counter()
                sample(net, x, c, t, graph, a, gen, steps=steps)
                if device == 'cuda':
                    torch.cuda.synchronize()
                if repeat:
                    durations.append(time.perf_counter() - started)
            median = float(np.median(durations))
            rows.append(dict(batch=batch, steps=steps, seconds=durations,
                median_batch_seconds=median, amortized_seconds_per_path=median/batch))
    return dict(device=torch.cuda.get_device_name() if device == 'cuda' else 'cpu',
        scope='warm sampling including source draw; excludes checkpoint load, I/O, queue and training', rows=rows)


def statistics(paths, dt, threshold, dwell):
    s = np.asarray(paths, dtype=np.float64)
    mz = s[..., 2].mean(-1)
    rows, _ = event_rows(mz, np.arange(s.shape[1]) * dt, threshold, dwell)
    returned = [bool(np.any(p[np.flatnonzero(p < 0)[0]+1:] >= threshold))
                if np.any(p < 0) else False for p in mz]
    energy = (-(s[:, :, :-1] * s[:, :, 1:]).sum((-2, -1))
        - .1 * np.square(s[..., 2]).sum(-1)) / s.shape[-2]
    return dict(negative_endpoint=float((mz[:, -1] < 0).mean()),
                crossed_zero=float((mz < 0).any(-1).mean()),
                crossing_return=float(np.mean(returned)),
                committed_switch=float(rows[:, 0].mean()),
                unresolved_transition=float((np.abs(mz[:, -1]) < threshold).mean()),
                endpoint_variance=float(mz[:, -1].var()),
                endpoint_mean=float(mz[:, -1].mean()),
                mean_energy=float(energy.mean()),
                neighbor_correlation=float((s[:, :, :-1]*s[:, :, 1:]).sum(-1).mean()),
                temporal_increment_mse=float(np.square(np.diff(s, axis=1)).mean()),
                norm_error=float(np.abs(np.linalg.norm(s, axis=-1)-1).max()))


def run(config, seed, device):
    cfg = json.loads(Path(config).read_text())
    if seed not in cfg['training_seeds']:
        raise ValueError('unregistered training seed')
    if device == 'cuda' and (not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available()):
        raise RuntimeError('CUDA requires a Slurm GPU allocation')
    raw = Path(cfg['raw_output']) / str(seed)
    out = Path(cfg['analysis_output']) / str(seed)
    if raw.exists() or out.exists():
        raise FileExistsError('preserve prior diagnostics; use a new output version')
    raw.mkdir(parents=True); out.mkdir(parents=True)
    checkpoint_path = Path(cfg['checkpoints']) / f'rfm_{seed}.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = checkpoint['config']
    if model_cfg['frames'] != cfg['frames'] or model_cfg['duration'] != cfg['physical_window']:
        raise ValueError('legacy checkpoint time contract mismatch')
    net = build_model('rfm', width=model_cfg['width'], blocks=model_cfg['blocks'],
                      condition_mean=(.12, .1), condition_std=(.01, 1.)).to(device)
    net.load_state_dict(checkpoint['model']); net.eval()
    manifest = dict(status='RUNNING', scope=cfg['scope'], config=cfg,
                    checkpoint_sha256=digest(checkpoint_path), data_sha256=digest(cfg['data']),
                    code_sha256={p: digest(p) for p in [__file__, 'scripts/model/gate_f.py',
                                     'tests/experiments/bauer_window.py']},
                    job_id=os.getenv('SLURM_JOB_ID'), training_seed=seed,
                    normalization='legacy physical reduced time / 16000',
                    existing_features=['condition standardization', 'per-block FiLM', 'pointwise LayerNorm'],
                    cases=[], pairs=[], interfaces=[], losses=[])
    training = json.loads((Path(cfg['checkpoints'])/'training.json').read_text())
    manifest['training_history'] = [r for r in training['models']
                                    if r['kind'] == 'rfm' and r['seed'] == seed]
    save(out/'summary.json', manifest)
    with np.load(cfg['data']) as d:
        development = d['development']; conditions = d['development_condition']
    n = cfg['latent_seeds']; ni = cfg['initials']; start = time.monotonic()

    def generator(offset=0):
        return torch.Generator(device=device).manual_seed(cfg['latent_seed_base'] + offset)

    def tensor(a):
        return torch.as_tensor(a, device=device, dtype=torch.float32)

    def record(name, pred, initial):
        if not torch.isfinite(pred).all() or not torch.equal(pred[:, 0], initial):
            raise RuntimeError('nonfinite sample or initial mismatch')
        a = pred[:, :, 0, :, 0].cpu().numpy()
        np.save(raw/f'{name}.npy', a)
        row = dict(name=name, shape=list(a.shape), file=str(raw/f'{name}.npy'),
                   **statistics(a, cfg['physical_window']/(cfg['frames']-1), cfg['threshold'], cfg['dwell']))
        manifest['cases'].append(row)
        # Keep nesting explicit: report diversity separately within each initial.
        if len(a) == ni*n:
            row['within_initial_endpoint_variance'] = a[..., 2].mean(-1)[:, -1].reshape(ni, n).var(-1).tolist()
        save(out/'summary.json', manifest)
        print(json.dumps(row), flush=True)
        return a

    common_initial = None
    for ti, theta in enumerate(cfg['theta']):
        pool = development[np.isclose(conditions[:, 0], theta)]
        selected, groups = initial_families(pool, ni, n)
        manifest.setdefault('initial_families', []).append(dict(theta=theta,
            selected_pool_indices=selected.tolist(), reference_pool_indices=groups.tolist()))
        x = tensor(np.repeat(pool[selected, 0], n, axis=0)[:, None, :, None])
        if common_initial is None:
            common_initial = x.clone()
        t = normalized_window(x, cfg['frames'], cfg['physical_window'])
        g = graph(25, device, torch.float32)
        axis = x.new_tensor([0., 0., 1.]).expand(len(x), -1)
        c = x.new_tensor([theta, cfg['lambda']]).expand(len(x), -1)
        # Independent per-latent streams, identical across steps and conditions.
        source = torch.cat([reference_path(x[j:j+1], cfg['frames'], generator(j)) for j in range(len(x))])
        previous = None
        for steps in cfg['integration_steps']:
            pred = sample(net, x, c, t, g, axis, generator(), steps=steps, source=source)
            a = record(f'T{theta}_steps{steps}', pred, x)
            if previous is not None:
                manifest['pairs'].append(dict(theta=theta, steps=[previous[0], steps],
                    paired_path_rmse=float(np.sqrt(np.square(a-previous[1]).mean()))))
            previous = (steps, a)
        y = tensor(pool[selected, :, None, :, None])
        for loss_seed in range(4):
            with torch.no_grad():
                loss, diagnostic = matching_loss(net, y, c[:ni], t[:ni], g, axis[:ni], generator(1000+loss_seed))
            manifest['losses'].append(dict(theta=theta, draw=loss_seed, loss=float(loss), diagnostics=diagnostic))
        reference = pool[groups.reshape(-1)]
        manifest.setdefault('development_reference', []).append(dict(theta=theta,
            **statistics(reference, 80., cfg['threshold'], cfg['dwell']),
            per_initial=[statistics(pool[group], 80., cfg['threshold'], cfg['dwell']) for group in groups],
            caveat='same exact initial states and equal family weights; real and latent noise are not paired'))
        for row in manifest['cases']:
            if row['name'].startswith(f'T{theta}_steps'):
                generated = np.load(row['file'])
                row['per_initial'] = [statistics(part, 80., cfg['threshold'], cfg['dwell'])
                                      for part in generated.reshape(ni, n, *generated.shape[1:])]
        # Temperature intervention holds the low-temperature initial states fixed.
        cs = torch.cat([reference_path(common_initial[j:j+1], cfg['frames'], generator(j)) for j in range(len(x))])
        p = sample(net, common_initial, c, t, g, axis, generator(), steps=64, source=cs)
        a = record(f'paired_condition_T{theta}', p, common_initial)
        if ti == 0:
            low = a
        else:
            manifest['condition_response'] = dict(paired_path_rmse=float(np.sqrt(np.square(a-low).mean())),
                endpoint_mz_change=(a[..., 2].mean(-1)[:, -1]-low[..., 2].mean(-1)[:, -1]).tolist(),
                caveat='nonzero response does not by itself establish correct conditional dynamics')

    # Shape tests only; do not use these trajectories to tune the checkpoint.
    for length in cfg['interface_lengths']:
        x = tensor(initial_states(1, length, 'diagnostic-interface')[:, None, :, None])
        g = graph(length, device, torch.float32)
        c = x.new_tensor([[cfg['theta'][0], cfg['lambda']]])
        axis = x.new_tensor([[0., 0., 1.]])
        t = normalized_window(x, cfg['frames'])
        p = sample(net, x, c, t, g, axis, generator(), steps=cfg['interface_steps'])
        record(f'interface_L{length}', p, x)
        manifest['interfaces'].append(dict(L=length, finite=bool(torch.isfinite(p).all()),
            initial_exact=bool(torch.equal(p[:, 0], x)), scope='software_only_not_physical_validation'))
    x = common_initial[:1]; g = graph(25, device, torch.float32)
    c = x.new_tensor([[cfg['theta'][0], cfg['lambda']]])
    axis = x.new_tensor([[0., 0., 1.]])
    t = normalized_window(x, cfg['frames'])
    p = sample_chunks(net, x, c, t, g, axis, generator(), chunks=cfg['rollout_chunks'], steps=cfg['rollout_steps'])
    record('chunk_rollout_software_only', p, x)
    tlong = normalized_window(x, 2*cfg['frames']-1, 2*cfg['physical_window'])
    p = sample(net, x, c, tlong, g, axis, generator(), steps=cfg['interface_steps'])
    record('direct_long_window_software_only', p, x)
    if cfg.get('benchmark_repeats', 0):
        manifest['sampling_benchmark'] = sampling_benchmark(net, x, c, g, axis, cfg, device)
    manifest.update(status='COMPLETE_DIAGNOSTICS_NOT_CERTIFIED', wall_seconds=time.monotonic()-start)
    save(out/'summary.json', manifest)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='tests/conf/bauer_diagnostics_20260913_v2.json')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = p.parse_args()
    torch.set_num_threads(1)
    run(args.config, args.seed, args.device)
