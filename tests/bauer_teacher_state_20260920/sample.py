"""New independent futures from frozen real states; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from tests.bauer_active_data_20260918.run import simulate
from tests.bauer_closed_loop_20260919.model import Net
from tests.bauer_transition_20260920.run import config, rollout
from tests.bauer_two_stage_20260919.common import allocation, seed_stream

HERE = Path(__file__).resolve().parent


def checked_state(h, row):
    s = h[row['key']][:]
    if hashlib.sha256(s.tobytes()).hexdigest() != row['state_sha256']:
        raise ValueError('Frozen initial state hash mismatch')
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['reference', 'model'])
    p.add_argument('--seed', type=int, choices=[11, 22, 33], default=11)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.device != ('cpu' if a.stage == 'reference' else 'cuda'):
        raise ValueError('Reference CPU; model CUDA only')
    allocation(a.device); torch.set_num_threads(1)
    spec = json.loads((HERE/'data/specification.json').read_text())
    rows = spec['rows'][:1] if a.smoke else spec['rows']
    count = 4 if a.smoke else spec[a.stage+'_branches']
    name = 'reference' if a.stage == 'reference' else f'model_{a.seed}'
    out = HERE/('smoke' if a.smoke else 'results'); out.mkdir(exist_ok=True)
    cfg = config(); checkpoint = None
    if a.stage == 'model':
        checkpoint = HERE.parent/'bauer_transition_20260920/runs'/f'transition_{a.seed}'/'best.pt'
        ck = torch.load(checkpoint, map_location='cuda', weights_only=False)
        net = Net('C', cfg['width'], cfg['depth']).cuda()
        net.load_state_dict(ck['model']); net.eval()
    with h5py.File(HERE/'data/initial_states.h5', 'r') as initials, h5py.File(out/f'{name}.h5', 'x') as dest:
        dest.attrs['complete'] = False
        dest.attrs['specification_json'] = json.dumps(spec)
        if checkpoint:
            dest.attrs['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        for i, row in enumerate(rows):
            s = checked_state(initials, row); initial = np.repeat(s[None], count, axis=0)
            start = time.perf_counter()
            seeds = np.array([seed_stream(260920810+i, j) for j in range(count)], dtype='i8')
            if a.stage == 'reference':
                c = row['physics_config']
                x, _ = simulate(initial, seeds, c, spec['llg_dt_fs'], spec['future_ps'])
            else:
                initial = torch.tensor(initial, device='cuda', dtype=torch.float32)
                initial = torch.nn.functional.normalize(initial, dim=-1)
                x = rollout(net, initial, spec['future_ps'], 260920910+i,
                            spec['flow_steps'], 'transition', cfg).cpu().numpy()
            if x.shape != (count, 2, row['L'], 3) or not np.isfinite(x).all():
                raise ValueError('Invalid future shape or nonfinite values')
            if abs(np.linalg.norm(x, axis=-1)-1).max() > 2e-6 or not np.allclose(x[:, 0], s, atol=2e-7):
                raise ValueError('Initial or norm constraint failed')
            d = dest.create_dataset(row['key'], data=x, compression='lzf')
            d.attrs['seconds'] = time.perf_counter()-start
            if a.stage == 'reference':
                d.attrs['noise_seeds'] = seeds
            else:
                d.attrs['latent_seed'] = 260920910+i
            print(name, i+1, len(rows), flush=True)
        dest.attrs['complete'] = True


if __name__ == '__main__':
    main()
