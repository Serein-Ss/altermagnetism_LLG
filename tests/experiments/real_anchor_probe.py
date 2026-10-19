"""Frozen-model restarts from real intermediate states, without future inputs."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from scripts.model.gate_f import sample
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.velocity_controls import CartesianPathNet


def generate(net, anchors, theta, cfg, case_id, start_index, device):
    anchors = np.asarray(anchors)
    if anchors.ndim != 3 or anchors.shape[-1] != 3 or not np.isfinite(anchors).all():
        raise ValueError('finite [parents, sites, 3] anchors required')
    if np.max(np.abs(np.linalg.norm(anchors, axis=-1)-1)) > 1e-5:
        raise ValueError('unit anchors required')
    batch = cfg['batch_size']
    if batch < 1 or len(anchors) % batch:
        raise ValueError('equal nonempty batches required')
    paths = []; seeds = []
    for index in range(len(anchors)//batch):
        x = torch.as_tensor(anchors[index*batch:(index+1)*batch, None, :, None],
                            dtype=torch.float32, device=device)
        c = x.new_tensor([theta, .1]).expand(len(x), -1)
        t = torch.linspace(0, 1, 201, device=device)[:cfg['frames']][None].expand(len(x), -1)
        seed = cfg['latent_base']+case_id*10000+start_index*100+index
        with torch.no_grad():
            pred = sample(net, x, c, t, None, None,
                          torch.Generator(device=device).manual_seed(seed), steps=cfg['steps'])
        if not torch.equal(pred[:, 0], x) or not torch.isfinite(pred).all():
            raise ValueError('invalid generated anchor or trajectory')
        if float((pred.norm(dim=-1)-1).abs().max()) > 1e-5:
            raise ValueError('unit sphere violation')
        paths.append(pred[:, :, 0, :, 0].cpu().numpy()); seeds.append(seed)
    return np.concatenate(paths), seeds


def run(seed):
    cfg_path = Path('tests/conf/real_anchor_probe.json')
    cfg = json.loads(cfg_path.read_text())
    if seed not in cfg['training_seeds']:
        raise ValueError('unregistered model')
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Slurm CUDA required')
    audit = json.loads(Path(cfg['audit']).read_text())
    if audit['status'] != 'COMPLETE_INPUT_INTEGRITY_ONLY':
        raise ValueError('complete input audit required')
    freeze = json.loads(Path(cfg['freeze']).read_text())
    checkpoint = Path(f'tests/output/stable_window_learning/window11/{seed}/step_40000.pt')
    if digest(checkpoint) != freeze['checkpoint_sha256'][str(seed)]:
        raise ValueError('checkpoint hash mismatch')
    state = torch.load(checkpoint, map_location='cuda', weights_only=False)
    if state['inference_weights'] != 'ema':
        raise ValueError('EMA required')
    net = CartesianPathNet().cuda().eval(); net.load_state_dict(state['model'])
    records = [r for r in audit['records'] if r['kind']=='reference' and r['case']['role']=='primary']
    if len(records)!=8 or {(r['case']['family'],r['case']['theta']) for r in records}!={(f,t) for f in range(4) for t in (.11,.13)}:
        raise ValueError('all eight primary groups required')
    out = Path(f'tests/results/real_anchor_probe/{seed}')
    raw = Path(f'tests/data/real_anchor_probe/{seed}')
    if out.exists() or raw.exists():
        raise FileExistsError('preserve previous probe')
    out.mkdir(parents=True); raw.mkdir(parents=True)
    result = dict(status='RUNNING', config=cfg, config_sha256=digest(cfg_path),
        audit_sha256=digest(cfg['audit']), checkpoint_sha256=digest(checkpoint),
        code_sha256={p:digest(p) for p in [__file__, 'scripts/model/gate_f.py', 'tests/experiments/velocity_controls.py']},
        job_id=os.getenv('SLURM_JOB_ID'), rows=[])
    save(out/'summary.json', result)
    tasks = [(r, i, start) for r in records for i, start in enumerate(cfg['start_frames'])]
    order = np.random.default_rng(cfg['schedule_seed']+seed).permutation(len(tasks))
    for task_index in order:
        r, i, start = tasks[task_index]; case = r['case']
        if digest(r['path']) != r['sha256']:
            raise ValueError('reference changed since audit')
        with np.load(r['path']) as d:
            anchors = d['spins'][:cfg['n_parents'], start].copy()
        pred, seeds = generate(net, anchors, case['theta'], cfg, case['case_id'], i, 'cuda')
        path = raw/f"family_{case['family']}_T{case['theta']}_start{start}.npz"
        np.savez_compressed(path, spins=pred, parent_index=np.arange(len(pred)),
                            reduced_time=np.arange(cfg['frames'])*80, batch_seeds=seeds)
        result['rows'].append(dict(family=case['family'], theta=case['theta'], start_frame=start,
            reference_path=r['path'], reference_sha256=r['sha256'], path=str(path), sha256=digest(path)))
        save(out/'summary.json', result)
    result['status']='COMPLETE_REAL_ANCHOR_PROBE_NOT_CERTIFIED'; save(out/'summary.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', choices=['cuda'], required=True)
    args = p.parse_args(); torch.set_num_threads(1); run(args.seed)
