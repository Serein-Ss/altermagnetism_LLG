"""Complete per-parent 2D diagnostic table, including full Hamiltonian energy."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from tests.bauer_two_stage_20260919.common import allocation, dump
from tests.literature_preflight_20260917.models import build
from tests.physical_audit_20260917.gallery import triangular_charge
from tests.miltat_library_20260920.geometry import CoreGeometry
from tests.path_smoke_20260917.model import PathNet, neighbors
from tests.cross_system_audit_20260920.miltat import DATA
from .models import SpaceTimeNet
from .train_miltat import physical_config, evaluate
from .long import selected

HERE = Path(__file__).resolve().parent


@torch.no_grad()
def features(raw, cell, geometry):
    s = raw.astype('f8'); b, t = s.shape[:2]; n = np.prod(s.shape[2:-1])
    flat = s.reshape(b,t,-1,3); d = np.diff(flat,axis=1)
    centered = d-d.mean(0)
    covariance = np.einsum('btnc,btnd->tncd',centered,centered)/(b-1)
    energy = []; local = []
    for branch in s:
        tensor = torch.tensor(branch, device='cuda')
        energy.append((cell.energy(tensor)/n).cpu().numpy())
        local.append((tensor*neighbors(tensor[None],cell)[0]).sum(-1).flatten(1).mean(1).cpu().numpy())
    q, qi = triangular_charge(s)
    # Reciprocal coordinates of rectangular cell; basis average, not radial physical S(q).
    z = s[...,2].mean(-1); z = z-z.mean((-2,-1),keepdims=True)
    spectrum = (abs(np.fft.rfft2(z,axes=(-2,-1)))**2/np.prod(z.shape[-2:])).mean(0)
    core = [[geometry.measure(frame) for frame in branch] for branch in s]
    return dict(mz=flat[...,2].mean(-1), energy=np.array(energy), Q=q, Q_interior=qi,
                overlap=(flat*flat[:,:1]).sum(-1).mean(-1), nearest_corr=np.array(local),
                increment_mean=d.mean(0), increment_cov=covariance,
                increment_power=(d*d).sum(-1).mean((0,2)),
                adjacent_increment_dot=(d[:,1:]*d[:,:-1]).sum(-1).mean((0,2)),
                spectrum=spectrum, core=core,
                norm_error=float(abs(np.linalg.norm(flat,axis=-1)-1).max()))


def metrics_file(path):
    rows = []; examples = []
    rmse = lambda a,b: float(np.sqrt(np.mean((a-b)**2)))
    with h5py.File(path) as h:
        if not h.attrs['complete']: raise RuntimeError('Incomplete generated file')
        for pid in h:
            group = h[pid]; r = group['reference'][:]; x = group['generated'][:]
            shape = x.shape[2:4]; cell, _, _ = build(physical_config(), 'miltat', shape=shape)
            geometry = CoreGeometry(group['positions_nm'][:])
            a,b = features(x,cell,geometry), features(r,cell,geometry)
            row = dict(parent_id=pid, shape=shape, branches_generated=len(x), branches_reference=len(r),
                       max_norm_error=a['norm_error'], anchor_error=float(abs(x[:,0]-r[0,0]).max()))
            for key in ['mz','energy','Q','Q_interior','overlap','nearest_corr']:
                row[key+'_mean_rmse'] = rmse(a[key].mean(0), b[key].mean(0))
                row[key+'_variance_rmse'] = rmse(a[key].var(0,ddof=1), b[key].var(0,ddof=1))
            for key in ['increment_mean','increment_cov','increment_power','adjacent_increment_dot','spectrum']:
                row[key+'_rmse'] = rmse(a[key],b[key])
            row['increment_power_ratio'] = float(a['increment_power'].mean()/b['increment_power'].mean())
            row['endpoint_mz_w1'] = float(wasserstein_distance(a['mz'][:,-1],b['mz'][:,-1]))
            # Scale fixed across models; retain signed finite-sample unbiased estimator.
            f,g = [np.concatenate([v[k][:,1:] for k in ['mz','energy','overlap','nearest_corr']],axis=1) for v in [a,b]]
            f,g = f/np.sqrt(f.shape[1]),g/np.sqrt(g.shape[1])
            row['path_feature_energy_distance_u'] = float(2*cdist(f,g).mean()-cdist(f,f).sum()/(len(f)*(len(f)-1))-cdist(g,g).sum()/(len(g)*(len(g)-1)))
            for label,v in [('generated',a),('reference',b)]:
                states = [c['status'] for branch in v['core'] for c in branch]
                row[label+'_core_status_counts'] = {state:states.count(state) for state in set(states)}
            rows.append(row)
            # Keep full scalar time-series and validity labels, not just a scalar score.
            examples.append(dict(parent_id=pid, time_ps=group['time_ps'][:].tolist(),
                **{label:{k:(v[k].tolist() if isinstance(v[k],np.ndarray) else v[k])
                          for k in ['mz','energy','Q','Q_interior','overlap','nearest_corr','increment_power','adjacent_increment_dot','core']}
                   for label,v in [('generated',a),('reference',b)]}))
    return dict(rows=rows, series=examples, source=str(path), scientific_certification=False,
         limitations=['Only 8 branches per parent; covariance and topology probabilities remain noisy.',
         'Fixed-open finite system; Q need not be integer. Core failures are retained.',
         'Spectrum uses rectangular-cell reciprocal coordinates; no radial isotropy claim.',
         'Energy includes exchange, anisotropy, DMI and full open dipoles, used only for evaluation.',
         'The energy-distance U estimator can be negative; this is finite-sample noise, not negative true distance.'])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--device', required=True, choices=['cuda'])
    p.add_argument('--existing', action='store_true'); a = p.parse_args()
    allocation(a.device); torch.set_num_threads(1)
    out = HERE/'evaluation'; out.mkdir(exist_ok=True)
    if a.existing:
        for i in range(3):
            source = HERE.parent/'cross_system_audit_20260920/miltat'/str(i)/'generated.h5'
            dump(out/f'existing_{i}.json', metrics_file(source))
    else:
        manifest = json.loads((DATA/'manifest.json').read_text())['parents']
        for kind in ['local','full']:
            ck = selected('miltat',kind); state = torch.load(ck,map_location='cuda',weights_only=False)
            net = (PathNet() if kind=='local' else SpaceTimeNet()).cuda(); net.load_state_dict(state['model'])
            for split in ['dev','test']:
                path = out/f'{kind}_{split}.h5'
                result = evaluate(net,[r for r in manifest if r['split']==split],path)
                result.update(checkpoint=str(ck), split=split)
                dump(out/f'{kind}_{split}_selection.json',result)
                dump(out/f'{kind}_{split}.json',metrics_file(path))
            # Fixed dev parent / latent seeds: numerical convergence, not another selection split.
            for steps in [16,64,128]:
                path = out/f'{kind}_steps{steps}.h5'
                evaluate(net,[r for r in manifest if r['split']=='dev'][:1],path,steps=steps)
                dump(out/f'{kind}_steps{steps}.json',metrics_file(path))


if __name__ == '__main__': main()
