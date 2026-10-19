"""Four-parent learnability diagnostic; not held-out generalization evidence."""
import argparse
import hashlib
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.bauer_closed_loop_20260919.model import Net
from tests.bauer_closed_loop_20260919.run import config, records
from tests.bauer_two_stage_20260919.models import base_path, sample_path
from tests.bauer_two_stage_20260919.common import allocation, dump
from tests.bauer_moment_training_20260920.objective import differentiable_sample, replica_mean_loss
from tests.bauer_publication_20260920.audit import mean_bias_squared

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(); p.add_argument('--index', type=int, choices=range(6), required=True)
    p.add_argument('--device', choices=['cuda'], required=True); p.add_argument('--benchmark', action='store_true')
    a = p.parse_args(); allocation(a.device); torch.set_num_threads(1)
    seed = [11, 22, 33][a.index % 3]; weight = [0., 10.][a.index//3]
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    cfg = config(); temp = cfg['temperature_K']
    families = ['positive', 'negative', 'tilted', 'single_wall']
    candidates = sorted([r for r in records(cfg, 'train') if r['L'] == 16], key=lambda r:r['parent_id'])
    selected = [next(r for r in candidates if r['family'] == f) for f in families]
    arrays = []
    for row in selected:
        with h5py.File(row['path']) as h:
            x = h['spins'][:, :2]
            assert h.attrs['complete'] and len(set(h['seeds'][:])) == 8
            assert np.allclose(h['time_ps'][:2], [0., .1])
            assert np.allclose(x[:, 0], x[0, 0], atol=2e-7)
            arrays.append(x)
    truth = F.normalize(torch.tensor(np.stack(arrays), device='cuda'), dim=-1)
    times = torch.tensor([0., .1], device='cuda')
    net = Net('C', 64, 3).cuda()
    ck = HERE.parent/'bauer_moment_training_20260920/runs'/f'w0_{seed}'/'best.pt'
    net.load_state_dict(torch.load(ck, map_location='cuda', weights_only=False)['model'])
    out = HERE/('bauer_benchmark' if a.benchmark else 'bauer')/f'w{weight:g}_{seed}'
    out.mkdir(parents=True, exist_ok=False)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    trng = torch.Generator(device='cuda').manual_seed(seed+900)
    mrng = np.random.default_rng(seed+100)
    history = []; timed = []; cut_count = 0; spin_count = 0

    @torch.no_grad()
    def assess(label):
        rows = []
        with h5py.File(out/f'{label}.h5', 'x') as h:
            for j, row in enumerate(selected):
                initial = truth[j, :1, 0].repeat(64, 1, 1)
                gen = sample_path(net, initial, times, 272000+j, 64, 'anchored', dict(anchored_sigma=.7, anchored_time_rho=.8))
                x = gen.cpu().numpy(); ref = truth[j].cpu().numpy()
                g = h.create_group(row['parent_id']); g['generated'] = x; g['reference'] = ref
                dg = x[:, 1]-x[:, 0]; dr = ref[:, 1]-ref[:, 0]
                rows.append(dict(parent_id=row['parent_id'], family=row['family'],
                                 mean_increment=mean_bias_squared(dg, dr),
                                 generated_variance=float(dg.var(0, ddof=1).mean()),
                                 reference_variance=float(dr.var(0, ddof=1).mean())))
        return rows

    before = None if a.benchmark else assess('before')
    gradient_probe = None
    for step in range(8 if a.benchmark else 1000):
        torch.cuda.synchronize(); started = time.perf_counter()
        pi = rng.integers(4, size=16); bi = rng.integers(8, size=16)
        y = truth[pi, bi]
        z = base_path(y[:, 0], 2, trng, 'anchored', .7, .8)
        tau = torch.rand(16, device='cuda', generator=trng)
        state, target, info = path_interpolate(z, y, tau, trng)
        cut_count += info['near_cut_count']; spin_count += info['free_spin_count']
        fm = (net(state, tau, y[:, 0], times, temp)[:, 1]-target[:, 1]).square().sum(-1).mean()
        moment = fm.new_zeros(())
        if weight:
            ids = mrng.integers(4, size=2)
            s = truth[ids, 0, 0]; initial = s[:, None].expand(-1, 8, -1, -1).reshape(16, 16, 3)
            gen = differentiable_sample(net, initial, times, seed*100000+step, 16, temp)[:, 1].reshape(2, 8, 16, 3)
            perm = np.stack([mrng.permutation(8) for _ in ids])
            refs = torch.stack([truth[i, b, 1] for i, b in zip(ids, perm)])
            moment = replica_mean_loss(gen[:, :4], gen[:, 4:], refs[:, :4], refs[:, 4:])
            if step == 0:
                params = list(net.parameters())
                gf = torch.autograd.grad(fm, params, retain_graph=True, allow_unused=True)
                gm = torch.autograd.grad(moment, params, retain_graph=True, allow_unused=True)
                pairs = [(f.flatten(), m.flatten()) for f, m in zip(gf, gm) if f is not None and m is not None]
                f, m = [torch.cat([v[k] for v in pairs]) for k in [0, 1]]
                gradient_probe = dict(fm_norm=float(f.norm()), unweighted_moment_norm=float(m.norm()),
                                      cosine=float(F.cosine_similarity(f[None], m[None])))
        loss = fm+weight*moment; opt.zero_grad(set_to_none=True); loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
        assert torch.isfinite(loss) and torch.isfinite(grad)
        opt.step(); torch.cuda.synchronize(); timed.append(time.perf_counter()-started)
        if step % 100 == 0: history.append(dict(step=step, fm=float(fm), moment=float(moment), grad=float(grad)))
    if a.benchmark:
        dump(out/'report.json', dict(step_seconds=timed, median_warm_seconds=float(np.median(timed[2:])))); return
    after = assess('after')
    torch.save(dict(model=net.state_dict(), step=1000, seed=seed, weight=weight), out/'last.pt')
    dump(out/'report.json', dict(before=before, after=after, history=history, gradient_probe=gradient_probe,
         step_seconds_median=float(np.median(timed)), near_cut_fraction=cut_count/spin_count,
         weight=weight, seed=seed, source_checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),
         selected_parents=[r['parent_id'] for r in selected], scientific_certification=False,
         interpretation='In-sample fixed-four-parent diagnostic. References reused in fitting; no independent accuracy claim. '
         'Weight 10 is a sensitivity intervention, not a selected production recipe. Equal updates, unequal compute.'))


if __name__ == '__main__': main()
