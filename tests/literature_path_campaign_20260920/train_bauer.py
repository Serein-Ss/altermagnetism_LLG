"""Paired old/group-symmetrized continuation; no test-set model selection."""
import argparse
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.bauer_closed_loop_20260919.model import Net
from tests.bauer_closed_loop_20260919.run import config, batch
from tests.bauer_transition_20260920.run import config as path_config
from tests.bauer_moment_training_20260920.train import validate
from tests.bauer_moment_training_20260920.data import TrainingBranchPairs
from tests.bauer_moment_training_20260920.objective import differentiable_sample, replica_mean_loss
from tests.bauer_two_stage_20260919.models import base_path
from tests.bauer_two_stage_20260919.common import allocation, dump
from tests.mdgen_path_refactor_20260918.data import PathWindows
from .models import ReversalNet

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--device', required=True, choices=['cuda'])
    p.add_argument('--index', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 3)))
    p.add_argument('--benchmark', action='store_true'); a = p.parse_args()
    allocation(a.device); torch.set_num_threads(1)
    if a.index not in range(6): raise ValueError('Expected index 0..5')
    kind = ['old', 'fixed'][a.index//3]; seed = [11,22,33][a.index%3]
    torch.manual_seed(seed); rng = np.random.default_rng(seed); mrng = np.random.default_rng(seed+100)
    trng = torch.Generator(device='cuda').manual_seed(seed+2000)
    oc = config(); pc = path_config(); cfg = dict(mean_score_scale=.01)
    roots = [ROOT/oc['data']/f'L{n}' for n in pc['sizes']]
    windows = [PathWindows([r], 'train', .1) for r in roots]
    pairs = [TrainingBranchPairs([r]) for r in roots]
    net = (Net if kind == 'old' else ReversalNet)('C', 64, 3).cuda()
    warm = HERE.parent/'bauer_moment_training_20260920/runs'/f'w0_{seed}'/'best.pt'
    net.load_state_dict(torch.load(warm, map_location='cuda', weights_only=False)['model'])
    out = HERE/('benchmark_bauer' if a.benchmark else 'bauer')/f'{kind}_{seed}'
    out.mkdir(parents=True, exist_ok=False)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-5)
    best = float('inf'); history = []; seconds = []; started = time.perf_counter()
    dump(out/'config.json', dict(kind=kind, seed=seed, updates=1500, moment_weight=10,
         warm=str(warm), comparison='Same batches, warm checkpoint, updates and mean weight; unequal compute',
         scientific_certification=False))
    for step in range(1, (8 if a.benchmark else 1500)+1):
        net.train(); torch.cuda.synchronize(); ts = time.perf_counter()
        y, times = batch(windows, rng, dict(oc, batch=16), 'cuda', 'C')
        z = base_path(y[:, 0], 2, trng, 'anchored'); tau = torch.rand(len(y), device='cuda', generator=trng)
        x, target, _ = path_interpolate(z, y, tau, trng)
        fm = (net(x, tau, y[:, 0], times, pc['temperature_K'])[:, 1]-target[:, 1]).square().sum(-1).mean()
        ds = pairs[int(mrng.integers(len(pairs)))]; n = 2; b = 4
        groups = [ds.sample(int(mrng.integers(len(ds))), mrng, b) for _ in range(n)]
        s = F.normalize(torch.tensor(np.stack([r['initial'] for r in groups]), device='cuda'), dim=-1)
        initial = s[:, None].expand(-1, 2*b, -1, -1).reshape(n*2*b, s.shape[1], 3)
        gen = differentiable_sample(net, initial, times, seed*100000+step, 16, pc['temperature_K'])
        gen = gen[:, 1].reshape(n, 2*b, s.shape[1], 3)
        ra, rb = [F.normalize(torch.tensor(np.stack([r[k] for r in groups]), device='cuda'), dim=-1)
                  for k in ['reference_a', 'reference_b']]
        moment = replica_mean_loss(gen[:, :b], gen[:, b:], ra, rb)
        loss = fm+10*moment; opt.zero_grad(set_to_none=True); loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
        if not torch.isfinite(loss) or not torch.isfinite(grad): raise RuntimeError('Nonfinite optimization')
        opt.step(); torch.cuda.synchronize(); seconds.append(time.perf_counter()-ts)
        if step % 100 == 0: print(step, float(fm), float(moment), flush=True)
        if not a.benchmark and step % 500 == 0:
            dev = validate(net, pc, cfg, False); dump(out/f'dev_{step}.json', dev)
            history.append(dict(step=step, fm=float(fm), moment=float(moment), dev_score=dev['score']))
            state = dict(model=net.state_dict(), kind=kind, seed=seed, step=step, dev_score=dev['score'])
            if dev['score'] < best: best = dev['score']; torch.save(state, out/'best.pt')
            torch.save(state, out/'last.pt'); dump(out/'history.json', history)
    if a.benchmark:
        ts = time.perf_counter(); validate(net, pc, cfg, True); torch.cuda.synchronize()
        dump(out/'report.json', dict(warm_step_seconds=float(np.median(seconds[2:])),
             one_parent_validation_seconds=time.perf_counter()-ts,
             estimated_1500_updates_plus_3_dev_seconds=float(np.median(seconds[2:]))*1500+(time.perf_counter()-ts)*36*3))
    else:
        dump(out/'complete.json', dict(best_dev_score=best, seconds=time.perf_counter()-started,
             scientific_certification=False, heldout_test_used=False))


if __name__ == '__main__': main()
