"""Save every development path. No tiling, overwriting or physical GO claim."""
import argparse
import json
import os
from pathlib import Path
import time as clock
import h5py
import torch
from scripts.validation.gate_f_contract import load_contract, digest
from scripts.datasets.gate_f_paths import GateFPaths
from scripts.literature.gomonay_2024.model import DoubleLayer
from scripts.model.gate_f import CellGraph, build_model, sample


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--initial-id', required=True)
    p.add_argument('--latent-seeds', type=int, nargs='+', required=True)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    c = load_contract(args.contract)
    if args.device == 'cuda' and (not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available()):
        raise RuntimeError('CUDA requires Slurm allocation')
    if len(set(args.latent_seeds)) != len(args.latent_seeds) or not set(args.latent_seeds) <= set(c['latent_seeds']):
        raise ValueError('latent seeds outside contract or duplicated')
    if args.steps not in c['integration_steps']:
        raise ValueError('integration steps outside contract')
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if checkpoint['schema'] != 'gate_f_model_v1' or checkpoint['contract_sha256'] != c['contract_sha256']:
        raise ValueError('checkpoint/contract mismatch')
    dataset = GateFPaths(c, 'development')
    candidates = [i for i, row in enumerate(dataset.rows) if row['initial_id'] == args.initial_id]
    if not candidates:
        raise ValueError('initial ID not in development split')
    row = dataset[candidates[0]]
    initial = row['spins'][:1].to(device=args.device, dtype=torch.float32)
    time = row['time'][None].to(initial)
    condition = row['condition'][None].to(initial)
    axis = initial.new_tensor([[0., 0., 1.]])
    cell = DoubleLayer(c['model']['reduced'], initial.shape[2:4], wall=True,
                       orientation=c['model']['orientation'], periodic=c['model']['periodic'])
    graph = CellGraph(cell).to(args.device)
    net = build_model(checkpoint['model_kind'], **checkpoint['model_config']).to(args.device)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    args.output.mkdir(parents=True, exist_ok=False)
    records, started = [], clock.monotonic()
    for seed in args.latent_seeds:
        if clock.monotonic()-started > c['verified_budget']['walltime_seconds']:
            raise RuntimeError('STOP: sampling walltime exceeded')
        generator = torch.Generator(device=args.device).manual_seed(seed)
        prediction = sample(net, initial, condition, time, graph, axis, generator,
                            steps=args.steps, kind=checkpoint['model_kind'])
        if not torch.isfinite(prediction).all():
            raise RuntimeError('STOP: nonfinite generated path; no silent discard')
        energy = graph.energy(prediction.reshape(1, time.shape[1], -1, 3), axis)
        signs = prediction.new_tensor([1., -1.])[None, :, None, None, None]
        path = args.output/f'latent_{seed}.h5'
        temporary = path.with_suffix('.h5.partial')
        with h5py.File(temporary, 'x') as h:
            h.attrs.update(schema='gate_f_generated_v1', complete=False, initial_id=args.initial_id,
                           latent_seed=seed, training_seed=checkpoint['training_seed'],
                           model_kind=checkpoint['model_kind'], integration_steps=args.steps,
                           contract_sha256=c['contract_sha256'], checkpoint_sha256=digest(args.checkpoint),
                           source_path_sha256=dataset.rows[candidates[0]]['sha256'],
                           scope='development_not_final_test', condition_json=json.dumps(condition[0].tolist()),
                           maximum_norm_error=float((prediction.norm(dim=-1)-1).abs().max()))
            h.create_dataset('spins', data=prediction[0].cpu().numpy(), chunks=(1, *prediction.shape[2:]), compression='gzip')
            h.create_dataset('initial', data=initial[0].cpu().numpy())
            h.create_dataset('time', data=time[0].cpu().numpy())
            h.create_dataset('energy', data=energy[0].cpu().numpy())
            h.create_dataset('magnetization', data=prediction[0].mean((1, 2, 3)).cpu().numpy())
            h.create_dataset('neel', data=(prediction[0]*signs).mean((1, 2, 3)).cpu().numpy())
            h.attrs['complete'] = True
        temporary.rename(path)
        records.append(dict(path=str(path.resolve()), sha256=digest(path), latent_seed=seed))
        if sum(f.stat().st_size for f in args.output.iterdir()) > c['verified_budget']['max_output_bytes']:
            raise RuntimeError('STOP: sampling output budget exceeded')
    with (args.output/'manifest.json').open('x') as h:
        json.dump(dict(paths=records, status='generated_pending_distribution_analysis'), h, indent=2)


if __name__ == '__main__':
    main()
