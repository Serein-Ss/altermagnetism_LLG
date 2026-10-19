"""Stream every generated frame; literature-condition demonstrations, not certification."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from tests.bauer_closed_loop_20260919.model import Net
from tests.bauer_two_stage_20260919.models import sample_path
from tests.bauer_two_stage_20260919.common import allocation, dump, latent_seed
from tests.path_smoke_20260917.model import PathNet, generate
from tests.literature_preflight_20260917.models import build
from .models import ReversalNet, SpaceTimeNet
from .train_miltat import physical_config

HERE = Path(__file__).resolve().parent


def selected(system, kind):
    candidates = list((HERE/system).glob(f'{kind}_*/best.pt'))
    if len(candidates) != 3: raise RuntimeError('All three training seeds must finish before model selection')
    return min(candidates, key=lambda p: torch.load(p, map_location='cpu', weights_only=False)['dev_score'])


def setup(system, kind, benchmark):
    if system == 'bauer':
        checkpoint = (HERE.parent/'bauer_moment_training_20260920/runs/w0_11/best.pt' if benchmark else selected('bauer', 'fixed'))
        net = ReversalNet('C', 64, 3).cuda()
        source = HERE.parent/'bauer_fig2_10ns_20260917/results/trajectory.h5'
        with h5py.File(source) as h: initial = h['spins'][:, 0]
        times = torch.tensor([0., .1], device='cuda')
        def sample(s, seed):
            return sample_path(net, s, times, seed, 64, 'anchored', dict(anchored_sigma=.7, anchored_time_rho=.8))
        total = 100000; per_block = 1
    else:
        checkpoint = (None if benchmark else selected('miltat', kind))
        net = (SpaceTimeNet() if kind == 'full' else PathNet()).cuda()
        source = HERE.parent/'miltat_1ns_20260919/results/thermal.h5'
        with h5py.File(source) as h: initial = h['spins'][0]
        cell, reduced, _ = build(physical_config(), 'miltat', shape=(200,116))
        times = torch.arange(11, device='cuda')*.1; frame = torch.eye(3, device='cuda')
        def sample(s, seed):
            return generate(net, s, cell, frame, reduced['theta'], times, seed=seed, steps=32)
        total = 1000; per_block = 10
    if checkpoint:
        net.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=False)['model'])
    net.eval()
    return net, F.normalize(torch.tensor(initial, device='cuda'), dim=-1), sample, total, per_block, source, checkpoint


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(); p.add_argument('--device', required=True, choices=['cuda'])
    p.add_argument('--system', required=True, choices=['bauer','miltat']); p.add_argument('--kind', default='full', choices=['local','full'])
    p.add_argument('--benchmark', action='store_true'); a = p.parse_args()
    allocation(a.device); torch.set_num_threads(1); torch.manual_seed(260920)
    _, initial, sample, total, per_block, source, checkpoint = setup(a.system, a.kind, a.benchmark)
    name = a.system if a.system=='bauer' else f'miltat_{a.kind}'
    out = HERE/('benchmark_long' if a.benchmark else 'literature')/name
    out.mkdir(parents=True, exist_ok=a.benchmark is False)
    if a.benchmark:
        seconds = []
        for k in range(5):
            torch.cuda.synchronize(); ts = time.perf_counter()
            x = sample(initial, latent_seed(260920, k)); initial = x[:, -1]
            torch.cuda.synchronize(); seconds.append(time.perf_counter()-ts)
        dump(out/'report.json', dict(block_seconds=seconds, total_blocks=total,
             estimated_generation_seconds=float(np.median(seconds[1:]))*total,
             output_uncompressed_bytes=int((total*per_block+1)*initial.numel()*4),
             note='Generation benchmark excludes file I/O; scheduling estimates add margin',
             peak_memory_bytes=torch.cuda.max_memory_allocated()))
        return
    path = out/'generated.h5'; resumed = path.exists(); started = time.perf_counter()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with h5py.File(path, 'r+' if resumed else 'x') as h:
        if resumed:
            if h.attrs['checkpoint_sha256'] != digest: raise ValueError('Resume checkpoint mismatch')
            if h.attrs['complete']: raise RuntimeError('Result is already complete; refusing overwrite')
            start = int(h.attrs['completed_blocks']); initial = torch.tensor(h['spins'][:, start*per_block], device=initial.device)
        else:
            start = 0
            h.attrs.update(complete=False, completed_blocks=0, checkpoint_sha256=digest, checkpoint=str(checkpoint),
                           reference=str(source), time_unit='ps', scientific_certification=False,
                           seed=260920, flow_steps=64 if a.system=='bauer' else 32,
                           boundary='open', temperature_K=63.82484966960072 if a.system=='bauer' else 12.5,
                           limitation='Autoregressive extrapolation; independently sampled noise; not a certified distribution. '
                           '2D reference uses 2.5 fs integration versus training 0.3125 fs and a finite fixed open window.')
            h.create_dataset('spins', shape=(len(initial), total*per_block+1, *initial.shape[1:]), dtype='f4',
                             chunks=(1,1,*initial.shape[1:]), compression='lzf')
            h.create_dataset('time_ps', data=np.arange(total*per_block+1)*.1)
            h['spins'][:,0] = initial.cpu().numpy()
            if a.system=='miltat':
                with h5py.File(source) as ref: h['positions_nm'] = ref['positions_nm'][:]
        norm_error = float(h.attrs.get('max_norm_error', 0.))
        for k in range(start, total):
            x = sample(initial, latent_seed(260920, k)); raw = x[:, 1:].cpu().numpy()
            error = float(abs(np.linalg.norm(raw, axis=-1)-1).max())
            if not np.isfinite(raw).all() or error > 3e-6: raise RuntimeError('Generated trajectory constraint failure')
            h['spins'][:, k*per_block+1:(k+1)*per_block+1] = raw
            initial = x[:, -1]; norm_error = max(norm_error, error)
            # Commit only after writing a complete block; resume repeats at most 20 blocks.
            if (k+1)%20==0 or k+1==total:
                h.flush(); h.attrs['completed_blocks'] = k+1; h.attrs['max_norm_error'] = norm_error; h.flush()
                dump(out/'progress.json', dict(completed_blocks=k+1, total_blocks=total))
        h.attrs['complete'] = True; h.flush()
    dump(out/'complete.json', dict(checkpoint=str(checkpoint), reference=str(source),
         seconds_this_invocation=time.perf_counter()-started, resumed=resumed, total_blocks=total,
         physical_duration_ps=total*per_block*.1, branches=len(initial), flow_steps=64 if a.system=='bauer' else 32,
         scientific_certification=False, max_norm_error=norm_error))


if __name__ == '__main__': main()
