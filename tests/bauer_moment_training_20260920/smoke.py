"""GPU feasibility only: six joint-loss updates on training parents."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.bauer_closed_loop_20260919.model import Net
from tests.bauer_two_stage_20260919.models import base_path
from tests.bauer_two_stage_20260919.common import allocation
from .data import TrainingBranchPairs
from .objective import differentiable_sample, replica_mean_loss

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(); p.add_argument('--device', choices=['cuda'], required=True); p.parse_args()
    allocation('cuda'); torch.set_num_threads(1); torch.manual_seed(11)
    out = HERE/'smoke_results'; out.mkdir(exist_ok=False)
    net = Net('C', 64, 3).cuda()
    checkpoint = HERE.parent/'bauer_transition_20260920/runs/transition_11/best.pt'
    net.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=False)['model'])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-5)
    rng = np.random.default_rng(260920); generator = torch.Generator(device='cuda').manual_seed(260921)
    times = torch.tensor([0., .1], device='cuda'); temperature = 63.82484966960072
    rows = []
    for length in [16, 25, 32]:
        ds = TrainingBranchPairs([HERE.parent/'bauer_active_data_20260918/data'/f'L{length}'])
        for iteration in range(2):
            data = ds.sample(int(rng.integers(len(ds))), rng, branches=4)
            s = F.normalize(torch.tensor(data['initial'], device='cuda'), dim=-1)
            ra = F.normalize(torch.tensor(data['reference_a'], device='cuda'), dim=-1)[None]
            rb = F.normalize(torch.tensor(data['reference_b'], device='cuda'), dim=-1)[None]
            initial = s[None].repeat(8, 1, 1)
            y = torch.stack((initial, torch.cat((ra[0], rb[0]))), dim=1)
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()
            z = base_path(initial, 2, generator, 'anchored')
            tau = torch.rand(8, device='cuda', generator=generator)
            x, target, _ = path_interpolate(z, y, tau, generator)
            pred = net(x, tau, initial, times, temperature)
            fm = (pred[:, 1]-target[:, 1]).square().sum(-1).mean()
            generated = differentiable_sample(net, initial, times, 260930+len(rows), 16, temperature)
            moment = replica_mean_loss(generated[:4, 1][None], generated[4:, 1][None], ra, rb)
            loss = fm+moment
            opt.zero_grad(set_to_none=True); loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
            if not torch.isfinite(loss) or not torch.isfinite(grad):
                raise RuntimeError('Nonfinite joint loss/gradient')
            norm = float((generated.norm(dim=-1)-1).abs().max())
            if norm > 2e-6 or not torch.equal(generated[:, 0], initial):
                raise RuntimeError('Sphere or anchor constraint failed')
            opt.step(); torch.cuda.synchronize()
            row = dict(L=length, parent_id=data['parent_id'], fm=float(fm.detach()),
                       moment=float(moment.detach()), grad_norm=float(grad), max_norm_error=norm,
                       seconds=time.perf_counter()-started, peak_memory_bytes=torch.cuda.max_memory_allocated())
            rows.append(row); print(json.dumps(row), flush=True)
    (out/'complete.json').write_text(json.dumps(dict(feasibility_only=True, rows=rows,
         scientific_certification=False, integration_steps=16, mean_weight=1., updates=6), indent=2)+'\n')


if __name__ == '__main__':
    main()
