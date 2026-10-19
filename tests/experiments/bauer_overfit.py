"""Construct isolated small-sample arms, then use the controlled trainer."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scripts.workflow.bauer_campaign import save
from tests.experiments.bauer_budget_comparison import run


def main(index, plan_path='tests/conf/bauer_overfit_20260913.json'):
    plan = json.loads(Path(plan_path).read_text())
    arms = list(plan['arms']); n = len(plan['seeds'])
    if not 0 <= index < len(arms)*n:
        raise ValueError('unknown experiment index')
    arm = arms[index//n]; seed = plan['seeds'][index % n]
    cfg = json.loads(Path(plan['base_experiment']).read_text())
    indices = plan['arms'][arm]
    with np.load(cfg['data']) as d:
        y = d['train'][indices]; c = d['condition'][indices]
    if not np.all(c == c[0]) or not np.all(y[:, 0] == y[0, 0]):
        raise ValueError('overfit arm must share exact initial and condition')
    endpoint = y[:, -1, :, 2].mean(-1)
    if arm == 'single_positive' and not endpoint[0] > .6:
        raise ValueError('positive exemplar changed')
    if arm == 'single_negative' and not endpoint[0] < -.6:
        raise ValueError('negative exemplar changed')
    experiment = plan.get('experiment', 'bauer_overfit')
    cfg.update(train_indices=indices, evaluate_training_subset=True,
        model=plan.get('model', 'gate_f'),
        temporal_dilations=plan.get('temporal_dilations'),
        global_context=plan.get('global_context',False),
        flow_weighting=plan.get('flow_weighting','uniform'),
        compare_legacy=False, evaluation_theta=[float(c[0, 0])],
        initials=1, reference_latents=len(indices), latents=plan['latents'],
        budgets=plan['budgets'], sampling_steps=plan['sampling_steps'], scope=plan['scope'],
        output=f'tests/output/{experiment}/{arm}', raw=f'tests/data/{experiment}/{arm}',
        analysis=f'tests/results/{experiment}/{arm}')
    config_directory = 'overfit_runs' if experiment == 'bauer_overfit' else f'{experiment}_runs'
    path = Path(f'tests/conf/{config_directory}/{arm}_{seed}.json')
    if path.exists():
        raise FileExistsError(path)
    save(path, cfg)
    run(path, seed, 'cuda')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--index', type=int, required=True)
    p.add_argument('--plan', default='tests/conf/bauer_overfit_20260913.json')
    args = p.parse_args(); torch.set_num_threads(1); main(args.index, args.plan)
