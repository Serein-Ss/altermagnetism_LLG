"""Frozen local-transition models on fresh initial states and longer rollouts."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.director_transition import DirectorTransition
from tests.experiments.director_midpoint_sampling import rollout


def rollout_config(training, reference, sampling):
    cfg = dict(training)
    steps = round(reference['duration']/cfg['lag'])
    if steps < 1 or not np.isclose(steps*cfg['lag'], reference['duration'], atol=1e-12, rtol=0):
        raise ValueError('physical duration must align with learned transition lag')
    if not np.isclose(reference['save_dt'], cfg['lag'], atol=1e-12, rtol=0):
        raise ValueError('reference output interval differs from learned lag')
    cfg.update(rollout_steps=steps, generated_paths=reference['paths_per_anchor'],
               generation_batch=sampling['generation_batch'])
    return cfg


def run(seed, device):
    if device != 'cuda' or not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Slurm CUDA allocation required')
    source = Path('tests/results/independent_candidate_reference/manifest.json')
    manifest = json.loads(source.read_text())
    sampling_path = Path('tests/conf/independent_candidate_validation.json')
    assert digest(sampling_path)==manifest['config_sha256']
    sampling = json.loads(sampling_path.read_text())['sampling']
    audit=json.loads(Path('tests/results/independent_candidate_reference/isolation_audit.json').read_text())
    assert audit['manifest_sha256']==digest(source) and audit['status']=='NO_OVERLAP_IN_AUDITED_SOURCES'
    frozen = next(r for r in manifest['frozen_models'] if r['seed'] == seed)
    assert digest(frozen['checkpoint']) == frozen['sha256']
    assert digest(frozen['report']) == frozen['report_sha256']
    previous = json.loads(Path(frozen['report']).read_text())
    for p, h in previous['code_sha256'].items():
        assert digest(p) == h
    bank = Path('tests/data/independent_candidate_reference/anchors.npz')
    assert digest(bank) == manifest['bank_sha256']
    with np.load(bank) as z:
        states = z['unit_initials']
    checkpoint = torch.load(frozen['checkpoint'], map_location='cpu', weights_only=False)
    cfg = rollout_config(checkpoint['config'], manifest['config'], sampling)
    model = DirectorTransition(checkpoint['theta_mean'], checkpoint['theta_std'], cfg['width']).to(device)
    model.load_state_dict(checkpoint['ema'], strict=True)
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(f'tests/results/independent_candidate_sampling/{seed}')
    raw = Path(f'tests/data/independent_candidate_sampling/{seed}')
    if root.exists() or raw.exists():
        raise FileExistsError('preserve prior sampling')
    root.mkdir(parents=True); raw.mkdir(parents=True)
    report = dict(status='RUNNING', rows=[], seed=seed, job_id=os.environ['SLURM_JOB_ID'],
        config=cfg, sampling=sampling, sampling_sha256=digest(sampling_path),
        manifest_sha256=digest(source), checkpoint_sha256=frozen['sha256'],
        code_sha256={p: digest(p) for p in [__file__, 'tests/experiments/director_midpoint_sampling.py',
            'tests/experiments/projected_midpoint_sampler.py']})
    save(root/'summary.json', report)
    for ai, anchor in enumerate(manifest['anchors']):
        latent = sampling['latent_base']+seed*1000+ai
        state = None
        for steps in sampling['flow_steps']:
            torch.cuda.synchronize(); started = time.monotonic()
            paths, rng_state = rollout(model, states[anchor['initial_id']], anchor['theta'],
                cfg, checkpoint['latent_scale'], steps, latent, device)
            torch.cuda.synchronize(); elapsed = time.monotonic()-started
            assert paths.shape == (cfg['generated_paths'], cfg['rollout_steps']+1, manifest['config']['sites'], 3)
            assert np.isfinite(paths).all()
            assert np.max(abs(np.linalg.norm(paths, axis=-1)-1)) < 1e-5
            if state is not None:
                np.testing.assert_array_equal(rng_state, state)
            state = rng_state
            path = raw/f'anchor{ai}_steps{steps}.npz'
            np.savez_compressed(path, spins=paths, reduced_time=np.arange(cfg['rollout_steps']+1)*cfg['lag'], rng_state=rng_state)
            report['rows'].append(dict(anchor=ai, state=anchor, steps=steps, path=str(path),
                sha256=digest(path), latent_seed=latent, seconds=elapsed))
            save(root/'summary.json', report)
    report['status'] = 'COMPLETE_INDEPENDENT_CANDIDATE_SAMPLING_NOT_CERTIFIED'
    save(root/'summary.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, choices=[11,22,33], required=True)
    p.add_argument('--device', required=True)
    args = p.parse_args()
    run(args.seed, args.device)
