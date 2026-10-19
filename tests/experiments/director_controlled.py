"""Concurrent original/director comparison; frozen budget and paired RNG."""
import argparse
import copy
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.fine_transition_dataset import FineTransitionDataset
from tests.experiments.local_equivariant_transition import LocalEquivariantTransition, loss, sample

from tests.experiments.director_transition import DirectorTransition

CONFIG = Path('tests/conf/local_transition_training.json')


def training_arrays(dataset):
    # Pilot is small: explicit materialization for random GPU batches, not a claim of lazy GPU training.
    rows = [dataset[i] for i in range(len(dataset))]
    initial = np.stack([r['initial'] for r in rows])
    target = np.stack([r['target'] for r in rows])
    theta = np.array([r['theta'] for r in rows])
    delta = target-initial
    tangent = delta-(delta*initial).sum(-1, keepdims=True)*initial
    scale = float(np.sqrt(np.mean(np.sum(tangent*tangent, axis=-1))/2))
    return initial, target, theta, scale


def train_step(model, optimizer, initial, target, theta, lag, axis, generator, scale):
    optimizer.zero_grad(set_to_none=True)
    value, diagnostics = loss(model, initial, target, theta, lag, axis, generator, scale)
    if not torch.isfinite(value):
        raise ValueError('nonfinite training loss')
    value.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return float(value.detach()), float(grad), diagnostics


def run(seed, device, arm):
    if arm not in ("control", "director"):
        raise ValueError("unknown arm")
    if device != 'cuda' or not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Slurm CUDA allocation required')
    cfg = json.loads(CONFIG.read_text())
    if seed not in cfg['seeds']:
        raise ValueError('unplanned seed')
    root = Path(f'tests/results/director_controlled/{arm}/{seed}')
    output = Path(f'tests/output/director_controlled/{arm}/{seed}')
    raw = Path(f'tests/data/director_controlled/{arm}/{seed}')
    if any(p.exists() for p in (root, output, raw)):
        raise FileExistsError('preserve prior experiment')
    for p in (root, output, raw):
        p.mkdir(parents=True)
    train = FineTransitionDataset('train', lag=cfg['lag'])
    development = FineTransitionDataset('development', lag=cfg['lag'])
    assert set(train.parents).isdisjoint(development.parents)
    initial, target, theta, scale = training_arrays(train)
    mean, std = float(theta.mean()), float(theta.std())
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_class = LocalEquivariantTransition if arm == 'control' else DirectorTransition
    model = model_class(mean, std, cfg['width']).to(device)
    ema = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['updates'], eta_min=cfg['final_learning_rate'])
    initial, target, theta = [torch.as_tensor(a, device=device, dtype=torch.float32) for a in (initial, target, theta)]
    initial = torch.nn.functional.normalize(initial, dim=-1)
    target = torch.nn.functional.normalize(target, dim=-1)
    rng = torch.Generator(device=device).manual_seed(950000000+seed)
    axis = torch.tensor(cfg['axis'], device=device).expand(cfg['batch_size'], -1)
    lag = theta.new_full((cfg['batch_size'],), cfg['lag'])
    report = dict(arm=arm, status='TRAINING', config=cfg, config_sha256=digest(CONFIG), seed=seed,
        job_id=os.environ['SLURM_JOB_ID'], theta_mean=mean, theta_std=std, latent_scale=scale,
        train_parents=train.parents, development_parents=development.parents,
        source_sha256=digest('tests/results/fine_transition_pilot/summary.json'),
        code_sha256={p:digest(p) for p in [__file__, 'tests/experiments/director_transition.py', 'tests/experiments/local_equivariant_transition.py',
            'tests/experiments/fine_transition_dataset.py', 'scripts/model/gate_f.py', 'scripts/model/sphere.py']},
        history=[], rows=[])
    save(root/'summary.json', report)
    started = time.monotonic()
    for step in range(1, cfg['updates']+1):
        ids = torch.randint(len(initial), (cfg['batch_size'],), device=device, generator=rng)
        value, grad, diagnostics = train_step(model, optimizer, initial[ids], target[ids], theta[ids], lag, axis, rng, scale)
        scheduler.step()
        with torch.no_grad():
            for ep, p in zip(ema.parameters(), model.parameters()):
                ep.lerp_(p, 1-cfg['ema_decay'])
        if step == 1 or step % 100 == 0:
            report['history'].append(dict(step=step, loss=value, grad_norm=grad, diagnostics=diagnostics))
            save(root/'summary.json', report)
    torch.cuda.synchronize()
    report['training_seconds'] = time.monotonic()-started
    checkpoint = output/'final.pt'
    torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), rng=rng.get_state(), config=cfg, step=cfg['updates'],
        theta_mean=mean, theta_std=std, latent_scale=scale), checkpoint)
    report.update(status='GENERATING', checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint))
    save(root/'summary.json', report)
    ema.eval()
    for split, dataset in [('train', train), ('development', development)]:
        for row in dataset.rows:
            assert digest(row['path']) == row['sha256']
            with np.load(row['path']) as d:
                original = d['spins'][0, 0]
            batches = []
            latent_seed = 960000000+seed*1000+row['anchor']
            generate_rng = torch.Generator(device=device).manual_seed(latent_seed)
            for offset in range(0, cfg['generated_paths'], cfg['generation_batch']):
                n = min(cfg['generation_batch'], cfg['generated_paths']-offset)
                x = torch.tensor(original, device=device, dtype=torch.float32).expand(n, -1, -1)
                x = torch.nn.functional.normalize(x, dim=-1)
                frames = [x.cpu().numpy()]
                temperature = x.new_full((n,), row['state']['theta'])
                interval = x.new_full((n,), cfg['lag'])
                direction = x.new_tensor(cfg['axis']).expand(n, -1)
                for _ in range(cfg['rollout_steps']):
                    x = sample(ema, x, temperature, interval, direction, generate_rng, scale, cfg['flow_steps'])
                    frames.append(x.cpu().numpy())
                batches.append(np.stack(frames, axis=1))
            paths = np.concatenate(batches)
            if not np.isfinite(paths).all() or np.max(np.abs(np.linalg.norm(paths, axis=-1)-1)) > 1e-5:
                raise ValueError('invalid generated paths')
            path = raw/f"{split}_anchor{row['anchor']}.npz"
            np.savez_compressed(path, spins=paths, reduced_time=np.arange(cfg['rollout_steps']+1)*cfg['lag'])
            report['rows'].append(dict(split=split, anchor=row['anchor'], state=row['state'],
                path=str(path), sha256=digest(path), reference_path=row['path'], reference_sha256=row['sha256'],
                latent_seed=latent_seed, generation_batch=cfg['generation_batch']))
            save(root/'summary.json', report)
    report['status'] = 'COMPLETE_TRAINING_AND_GENERATION_NOT_CERTIFIED'
    save(root/'summary.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True, choices=['control', 'director'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    run(args.seed, args.device, args.arm)
