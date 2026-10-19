"""Fixed-checkpoint, fixed-latent integration refinement; no training."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.director_transition import DirectorTransition
from tests.experiments.local_equivariant_transition import sample


def rollout(model, original, theta, cfg, scale, steps, seed, device):
    rng = torch.Generator(device=device).manual_seed(seed)
    batches = []
    for offset in range(0, cfg['generated_paths'], cfg['generation_batch']):
        n = min(cfg['generation_batch'], cfg['generated_paths']-offset)
        x = torch.tensor(original, device=device, dtype=torch.float32).expand(n, -1, -1)
        x = torch.nn.functional.normalize(x, dim=-1)
        frames = [x.cpu().numpy()]
        temperature = x.new_full((n,), theta)
        interval = x.new_full((n,), cfg['lag'])
        direction = x.new_tensor(cfg['axis']).expand(n, -1)
        for _ in range(cfg['rollout_steps']):
            x = sample(model, x, temperature, interval, direction, rng, scale, steps)
            frames.append(x.cpu().numpy())
        batches.append(np.stack(frames, axis=1))
    return np.concatenate(batches), rng.get_state().cpu().numpy()


def run(seed, device):
    if device != 'cuda' or not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Slurm CUDA allocation required')
    source = Path(f'tests/results/director_controlled/director/{seed}/summary.json')
    previous = json.loads(source.read_text())
    assert previous['status'] == 'COMPLETE_TRAINING_AND_GENERATION_NOT_CERTIFIED'
    assert digest(previous['checkpoint']) == previous['checkpoint_sha256']
    for p, h in previous['code_sha256'].items():
        assert digest(p) == h
    cfg = previous['config']
    checkpoint = torch.load(previous['checkpoint'], map_location='cpu', weights_only=False)
    model = DirectorTransition(checkpoint['theta_mean'], checkpoint['theta_std'], cfg['width']).to(device)
    model.load_state_dict(checkpoint['ema'], strict=True); model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(f'tests/results/director_sampling_convergence/{seed}')
    raw = Path(f'tests/data/director_sampling_convergence/{seed}')
    if root.exists() or raw.exists():
        raise FileExistsError('preserve previous run')
    root.mkdir(parents=True); raw.mkdir(parents=True)
    report = dict(status='RUNNING', rows=[], source_sha256=digest(source), seed=seed,
                  code_sha256=digest(__file__), checkpoint_sha256=previous['checkpoint_sha256'],
                  steps=[64, 32, 128, 256], scope='Paired latent numerical refinement only. No model certification.')
    assert len(previous['rows']) == 12
    for row in previous['rows']:
        assert digest(row['reference_path']) == row['reference_sha256']
        assert digest(row['path']) == row['sha256']
        with np.load(row['reference_path']) as d:
            original = d['spins'][0, 0]
        with np.load(row['path']) as d:
            old = d['spins']
        state = None
        for steps in report['steps']:
            torch.cuda.synchronize(); started = time.monotonic()
            paths, rng_state = rollout(model, original, row['state']['theta'], cfg,
                checkpoint['latent_scale'], steps, row['latent_seed'], device)
            torch.cuda.synchronize(); elapsed = time.monotonic()-started
            if steps == 64:
                np.testing.assert_allclose(paths, old, atol=2e-6, rtol=0)
                state = rng_state
            np.testing.assert_array_equal(rng_state, state)
            assert paths.shape == old.shape and np.isfinite(paths).all()
            assert np.max(np.abs(np.linalg.norm(paths, axis=-1)-1)) < 1e-5
            path = raw/f"anchor{row['anchor']}_steps{steps}.npz"
            np.savez_compressed(path, spins=paths, reduced_time=np.arange(cfg['rollout_steps']+1)*cfg['lag'])
            report['rows'].append(dict(anchor=row['anchor'], split=row['split'], state=row['state'], steps=steps,
                path=str(path), sha256=digest(path), seconds=elapsed, latent_seed=row['latent_seed'],
                reference_path=row['reference_path'], reference_sha256=row['reference_sha256'],
                replay64_max_abs_error=float(np.max(np.abs(paths-old))) if steps==64 else None))
            save(root/'summary.json', report)
    report['status'] = 'COMPLETE_SAMPLING_REFINEMENT_NOT_CERTIFIED'
    save(root/'summary.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--seed', type=int, choices=[11,22,33], required=True)
    p.add_argument('--device', required=True); args = p.parse_args()
    run(args.seed, args.device)
