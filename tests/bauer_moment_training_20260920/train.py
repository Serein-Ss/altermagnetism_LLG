"""Paired flow-only/mean-constrained fine tuning, selected only on development."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
import yaml
from scripts.model.gate_f import path_interpolate
from tests.bauer_closed_loop_20260919.model import Net
from tests.bauer_closed_loop_20260919.run import config as old_config, batch, records, reference
from tests.mdgen_path_refactor_20260918.data import PathWindows
from tests.bauer_transition_20260920.run import config as path_config, development, rollout
from tests.bauer_two_stage_20260919.models import base_path
from tests.bauer_two_stage_20260919.common import allocation, dump
from tests.bauer_publication_20260920.audit import mean_bias_squared
from .data import TrainingBranchPairs
from .objective import differentiable_sample, replica_mean_loss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


@torch.no_grad()
def validate(net, pc, cfg, smoke):
    result = development(net, 'transition', pc, smoke)
    selected = records(old_config(), 'dev')
    if smoke: selected = selected[:1]
    values = []
    for j, row in enumerate(selected):
        truth, _, initial = reference(row, old_config())
        s = F.normalize(torch.tensor(initial, device='cuda', dtype=torch.float32), dim=-1)
        x = rollout(net, s[None].repeat(16, 1, 1), .1, 261000+j,
                    pc['dev_flow_steps'], 'transition', pc).cpu().numpy()
        values.append(mean_bias_squared(x[:, 1]-x[:, 0], truth[:, 1]-truth[:, 0]))
    result['single_step_mean'] = values
    result['path_score'] = result['score']
    result['score'] += float(np.mean([v['corrected_mse'] for v in values]))/cfg['mean_score_scale']
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--index', type=int, choices=range(6), required=True)
    p.add_argument('--device', choices=['cuda'], required=True); p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); allocation('cuda'); torch.set_num_threads(1)
    cfg = yaml.safe_load((HERE/'config.yaml').read_text()); pc = path_config(); oc = old_config()
    weight, seed = cfg['weights'][a.index//3], cfg['seeds'][a.index%3]
    if a.smoke:
        cfg.update(updates=2, validate_every=2, moment_parents_per_update=1)
        pc.update(batch=2, branches=2, dev_flow_steps=2)
    out = HERE/('train_smoke' if a.smoke else 'runs')/f'w{weight:g}_{seed}'
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed); rng = np.random.default_rng(seed); mrng = np.random.default_rng(seed+100)
    trng = torch.Generator(device='cuda').manual_seed(seed+2000)
    net = Net('C', pc['width'], pc['depth']).cuda()
    checkpoint = HERE.parent/'bauer_transition_20260920/runs'/f'transition_{seed}'/'best.pt'
    net.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=False)['model'])
    roots = [ROOT/oc['data']/f'L{n}' for n in pc['sizes']]
    windows = [PathWindows([root], 'train', .1) for root in roots]
    pairs = [TrainingBranchPairs([root]) for root in roots]
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['learning_rate'], weight_decay=1e-5)
    history = []; best = float('inf'); started = time.perf_counter()
    dump(out/'config.json', dict(cfg=cfg, path_config=pc, weight=weight, seed=seed, checkpoint=str(checkpoint)))
    for step in range(1, cfg['updates']+1):
        net.train(); y, times = batch(windows, rng, dict(oc, batch=pc['batch']), 'cuda', 'C')
        z = base_path(y[:, 0], 2, trng, 'anchored'); tau = torch.rand(len(y), device='cuda', generator=trng)
        x, target, _ = path_interpolate(z, y, tau, trng)
        fm = (net(x, tau, y[:, 0], times, pc['temperature_K'])[:, 1]-target[:, 1]).square().sum(-1).mean()
        moment = fm.new_zeros(())
        if weight:
            ds = pairs[int(mrng.integers(len(pairs)))]; n = cfg['moment_parents_per_update']; b = cfg['moment_branches_per_group']
            groups = [ds.sample(int(mrng.integers(len(ds))), mrng, b) for _ in range(n)]
            s = F.normalize(torch.tensor(np.stack([r['initial'] for r in groups]), device='cuda'), dim=-1)
            initial = s[:, None].expand(-1, 2*b, -1, -1).reshape(n*2*b, s.shape[1], 3)
            gen = differentiable_sample(net, initial, times, seed*100000+step, cfg['moment_steps'], pc['temperature_K'])
            gen = gen[:, 1].reshape(n, 2*b, s.shape[1], 3)
            ra, rb = [F.normalize(torch.tensor(np.stack([r[k] for r in groups]), device='cuda'), dim=-1) for k in ['reference_a', 'reference_b']]
            moment = replica_mean_loss(gen[:, :b], gen[:, b:], ra, rb)
        loss = fm+weight*moment; opt.zero_grad(set_to_none=True); loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
        if not torch.isfinite(loss) or not torch.isfinite(grad): raise RuntimeError('Nonfinite loss/gradient')
        opt.step()
        if step % 100 == 0: print(step, float(fm.detach()), float(moment.detach()), flush=True)
        if step % cfg['validate_every'] == 0:
            dev = validate(net, pc, cfg, a.smoke); dump(out/f'dev_{step}.json', dev)
            history.append(dict(step=step, score=dev['score'], fm=float(fm.detach()), moment=float(moment.detach())))
            state = dict(model=net.state_dict(), step=step, seed=seed, weight=weight, config=cfg, dev_score=dev['score'])
            if dev['score'] < best: best = dev['score']; torch.save(state, out/'best.pt')
            torch.save(dict(state, optimizer=opt.state_dict(), numpy_rng=rng.bit_generator.state,
                            moment_rng=mrng.bit_generator.state, latent_rng=trng.get_state()), out/'last.pt')
            dump(out/'history.json', history)
    dump(out/'complete.json', dict(updates=cfg['updates'], best_dev_score=best,
         seconds=time.perf_counter()-started, smoke=a.smoke, scientific_certification=False))


if __name__ == '__main__':
    main()
