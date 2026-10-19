"""Frozen train/dev anchor fit audit, not independent generalization certification."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from tests.bauer_closed_loop_20260919.model import Net
from tests.bauer_closed_loop_20260919.run import config as data_config
from tests.mdgen_path_refactor_20260918.data import PathWindows
from tests.bauer_transition_20260920.run import config, rollout
from tests.bauer_two_stage_20260919.common import allocation, dump
from tests.bauer_publication_20260920.audit import mean_bias_squared
from .evaluate import identity

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--device', choices=['cuda'], required=True)
    p.add_argument('--index', type=int, choices=range(6), required=True)
    a = p.parse_args(); allocation('cuda'); torch.set_num_threads(1)
    name = identity(a.index); cfg = config(); dc = data_config()
    run = HERE/'runs'/name
    assert not json.loads((run/'complete.json').read_text())['smoke']
    ckpath = run/'best.pt'; ck = torch.load(ckpath, map_location='cuda', weights_only=False)
    net = Net('C', cfg['width'], cfg['depth']).cuda(); net.load_state_dict(ck['model']); net.eval()
    out = HERE/'fit_audit'/name; out.mkdir(parents=True, exist_ok=False)
    rows = []
    with h5py.File(out/'generated.h5', 'x') as dest:
        dest.attrs['complete'] = False
        for split in ['train', 'dev']:
            ds = PathWindows([ROOT/dc['data']/f'L{n}' for n in cfg['sizes']], split, .1)
            group = dest.create_group(split)
            for j, (path, row, _, _) in enumerate(ds.records):
                with h5py.File(path, 'r') as ref:
                    truth = ref['spins'][:, :2].astype(float)
                    seeds = ref['seeds'][:]
                    assert len(set(seeds)) == len(seeds)
                    assert np.allclose(truth[:, 0], truth[0, 0], atol=2e-7)
                s = F.normalize(torch.tensor(truth[0, 0], device='cuda', dtype=torch.float32), dim=-1)
                latent = 261100000+(0 if split == 'train' else 10000)+j
                x = rollout(net, s[None].repeat(64, 1, 1), .1, latent, 64, 'transition', cfg).cpu().numpy().astype(float)
                d = group.create_dataset(row['parent_id'], data=x, compression='lzf'); d.attrs['seed'] = latent
                da, db = x[:, 1]-x[:, 0], truth[:, 1]-truth[:, 0]
                rows.append(dict(split=split, parent_id=row['parent_id'], family=row['family'], L=truth.shape[2],
                    model_branches=len(x), reference_branches=len(truth),
                    mean_increment=mean_bias_squared(da, db),
                    signed_mz=float(da[...,2].mean()-db[...,2].mean())))
                if j % 24 == 0: print(name, split, j, len(ds.records), flush=True)
        dest.attrs['complete'] = True
    dump(out/'metrics.json', dict(rows=rows, checkpoint_sha256=hashlib.sha256(ckpath.read_bytes()).hexdigest(),
        scientific_certification=False,
        note='Train references were used to fit the model and dev references for selection. '
             'Finite-sample correction here is descriptive, not an unbiased independent performance estimate. '
             'Compare parent families and sizes separately; train/dev state mixtures may differ.'))
    dump(out/'complete.json', dict(model=name, cases=len(rows)))


if __name__ == '__main__': main()
