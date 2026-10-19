"""Current EMA model risk by flow time and free-frame position."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from scripts.model.gate_f import reference_path, path_interpolate
from scripts.workflow.bauer_campaign import save, digest
from tests.experiments.velocity_controls import CartesianPathNet


def free_frame_risk(pred, target):
    if pred.shape != target.shape or pred.ndim != 6 or pred.shape[-1] != 3 or pred.shape[1] < 2:
        raise ValueError('matching [B,F,A,L,Y,3] tensors required')
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError('finite velocities required')
    reduce = lambda v: v[:, 1:].square().sum(-1).mean(dim=(2, 3, 4))
    return reduce(pred-target), reduce(target)


@torch.no_grad()
def run():
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Slurm CUDA required')
    cfg_path = Path('tests/conf/stable_flow_stage.json'); cfg = json.loads(cfg_path.read_text())
    if digest(cfg['data']) != cfg['data_sha256']:
        raise ValueError('training/development data changed')
    freeze = json.loads(Path(cfg['freeze']).read_text())
    out = Path('tests/results/stable_flow_stage')
    if out.exists(): raise FileExistsError('preserve prior audit')
    out.mkdir(parents=True)
    with np.load(cfg['data']) as d:
        arrays = {k: d[k] for k in ('train', 'condition', 'development', 'development_condition')}
    report = dict(status='RUNNING', config=cfg, config_sha256=digest(cfg_path), rows=[],
        code_sha256={p:digest(p) for p in [__file__, 'scripts/model/gate_f.py', 'tests/experiments/velocity_controls.py']})
    save(out/'summary.json', report)
    for seed in cfg['seeds']:
        cp = Path(f'tests/output/stable_window_learning/window11/{seed}/step_40000.pt')
        if digest(cp) != freeze['checkpoint_sha256'][str(seed)]: raise ValueError('checkpoint mismatch')
        state = torch.load(cp, map_location='cuda', weights_only=False)
        if state['inference_weights'] != 'ema': raise ValueError('EMA required')
        net = CartesianPathNet().cuda().eval(); net.load_state_dict(state['model'])
        for split_index, (split, cond_key) in enumerate((('train','condition'), ('development','development_condition'))):
            a = arrays[split]; n = len(a); w = cfg['windows_per_parent']; f = cfg['window_frames']
            rng = np.random.default_rng(cfg['random_seed']+split_index)
            starts = rng.integers(0, a.shape[1]-f+1, size=(n,w))
            parent = np.repeat(np.arange(n), w); flat_starts = starts.ravel()
            windows = a[parent[:,None], flat_starts[:,None]+np.arange(f)]
            errors = np.empty((n*w,len(cfg['tau']),f-1)); targets = np.empty_like(errors)
            for offset in range(0,n*w,cfg['batch_size']):
                stop = min(offset+cfg['batch_size'],n*w)
                y = torch.as_tensor(windows[offset:stop,:,None,:,None], dtype=torch.float32, device='cuda')
                c = torch.as_tensor(arrays[cond_key][parent[offset:stop]], dtype=y.dtype, device='cuda')
                t = torch.linspace(0,1,a.shape[1],device='cuda')[:f][None].expand(len(y),-1)
                local_seed = cfg['random_seed']+10000+split_index*100000+offset
                z = reference_path(y[:,0],f,torch.Generator(device='cuda').manual_seed(local_seed))
                for ti, tau in enumerate(cfg['tau']):
                    flow = y.new_full((len(y),),tau)
                    x, target, _ = path_interpolate(z,y,flow,torch.Generator(device='cuda').manual_seed(local_seed+1))
                    pred = net(x,flow,y[:,0],c,t,None,None)
                    error, target_size = free_frame_risk(pred,target)
                    errors[offset:stop,ti] = error.cpu().numpy(); targets[offset:stop,ti] = target_size.cpu().numpy()
            errors = errors.reshape(n,w,len(cfg['tau']),f-1)
            path = out/f'{split}_seed{seed}.npz'
            np.savez_compressed(path, squared_vector_error=errors,
                target_squared_norm=targets.reshape(errors.shape), window_starts=starts,
                parent_index=np.arange(n), condition=arrays[cond_key], tau=cfg['tau'])
            report['rows'].append(dict(seed=seed, split=split, checkpoint_sha256=digest(cp),
                path=str(path), sha256=digest(path),
                mean_risk_by_theta={str(theta): errors[np.isclose(arrays[cond_key][:,0],theta)].mean(axis=(0,1)).tolist()
                                    for theta in (.11,.13)}))
            save(out/'summary.json', report)
    report['status']='COMPLETE_SUPERVISED_RISK_NOT_CERTIFIED'; save(out/'summary.json',report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cuda'], required=True)
    parser.parse_args()
    torch.set_num_threads(1); run()
