"""Guarded Gate-F training. Requires real frozen P1/P2/data/resource evidence.

Full-space batch=1 with gradient accumulation. No legacy data conversion,
automatic production, silent cropping, or final-test model selection.
"""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from scripts.validation.gate_f_contract import load_contract
from scripts.datasets.gate_f_paths import GateFPaths
from scripts.literature.gomonay_2024.model import DoubleLayer
from scripts.model.gate_f import CellGraph, build_model, matching_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', required=True)
    p.add_argument('--model', choices=['rfm', 'deterministic', 'euclidean', 'autoregressive'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    c = load_contract(args.contract)
    if args.seed not in c['training_seeds'] or args.model not in c['models']:
        raise ValueError('model/seed outside contract')
    if args.device == 'cuda' and (not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available()):
        raise RuntimeError('CUDA requires a Slurm GPU allocation')
    dataset = GateFPaths(c, 'train')
    development = GateFPaths(c, 'development')
    # These are per-path training-only moments; zero-variance alpha uses a frozen floor.
    values = torch.tensor([[r['theta'], r['alpha']] for r in dataset.rows], dtype=torch.float64)
    mean, std = values.mean(0), values.std(0, unbiased=False).clamp_min(c['condition_scale_floor'])
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    order_generator = torch.Generator().manual_seed(args.seed)
    config = dict(width=c['training']['width'], blocks=c['training']['blocks'],
                  condition_mean=mean.tolist(), condition_std=std.tolist())
    net = build_model(args.model, **config).to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=c['training']['learning_rate'])
    args.output.mkdir(parents=True, exist_ok=False)
    started, updates = time.monotonic(), 0
    budget, graphs = c['verified_budget'], {}
    accumulation = c['training']['gradient_accumulation']
    if not isinstance(accumulation, int) or accumulation < 1:
        raise ValueError('positive accumulation required')

    def check_budget():
        if time.monotonic()-started > budget['walltime_seconds']:
            raise RuntimeError('STOP: walltime budget exceeded')
        if sum(f.stat().st_size for f in args.output.iterdir() if f.is_file()) > budget['max_output_bytes']:
            raise RuntimeError('STOP: output budget exceeded')

    def get_loss(row, rng):
        y = row['spins'][None].to(device=args.device, dtype=torch.float32)
        shape = tuple(y.shape[3:5])
        if shape not in graphs:
            cell = DoubleLayer(c['model']['reduced'], shape, wall=True,
                               orientation=c['model']['orientation'], periodic=c['model']['periodic'])
            graphs[shape] = CellGraph(cell).to(args.device)
        return matching_loss(net, y, row['condition'][None].to(y), row['time'][None].to(y),
                             graphs[shape], y.new_tensor([[0., 0., 1.]]), rng, kind=args.model)

    with (args.output/'history.jsonl').open('x') as log:
        for epoch in range(c['training']['epochs']):
            net.train()
            optimizer.zero_grad(set_to_none=True)
            order = torch.randperm(len(dataset), generator=order_generator).tolist()
            total, cut, free = 0., 0, 0
            for n, index in enumerate(order):
                check_budget()
                loss, info = get_loss(dataset[index], generator)
                if not torch.isfinite(loss):
                    raise RuntimeError('STOP: nonfinite loss')
                group_start = (n//accumulation)*accumulation
                (loss/min(accumulation, len(order)-group_start)).backward()
                total += float(loss.detach())
                cut += info.get('near_cut_count', 0)
                free += info.get('free_spin_count', 0)
                if (n+1) % accumulation == 0 or n+1 == len(order):
                    if updates >= budget['max_optimizer_steps']:
                        raise RuntimeError('STOP: optimizer budget exceeded')
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                        raise RuntimeError('STOP: nonfinite gradient')
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
            net.eval()
            dev_rng = torch.Generator(device=args.device).manual_seed(c['latent_seeds'][0])
            with torch.no_grad():
                dev = 0.
                for index in range(len(development)):
                    check_budget()
                    loss, _ = get_loss(development[index], dev_rng)
                    if not torch.isfinite(loss):
                        raise RuntimeError('STOP: nonfinite development loss')
                    dev += float(loss)
            log.write(json.dumps(dict(epoch=epoch, train_loss=total/len(dataset),
                       development_loss=dev/len(development), near_cut_count=cut, free_spin_count=free,
                       elapsed_seconds=time.monotonic()-started, optimizer_steps=updates))+'\n')
            log.flush()
            temporary = args.output/'last.pt.partial'
            torch.save(dict(schema='gate_f_model_v1', model_kind=args.model, model_config=config,
                       state_dict=net.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                       training_seed=args.seed, rng_state=generator.get_state(),
                       contract_sha256=c['contract_sha256'], production_enabled=False), temporary)
            temporary.replace(args.output/'last.pt')
            check_budget()
    print(json.dumps(dict(status='training_finished_not_gate_f_certified', output=str(args.output))))


if __name__ == '__main__':
    main()
