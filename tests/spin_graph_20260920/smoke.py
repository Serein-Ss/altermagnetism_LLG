"""Bounded CPU plumbing check using existing Bauer TRAIN samples only."""
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from scripts.core.cell_hamiltonian import CellHamiltonian
from scripts.model.gate_f import path_interpolate
from tests.bauer_closed_loop_20260919.run import config, records
from tests.path_smoke_20260917.model import generate
from .model import SpinGraphNet


def main():
    torch.set_num_threads(1)
    torch.manual_seed(41)
    started = time.perf_counter()
    out = Path(__file__).resolve().parent/'smoke'
    out.mkdir(exist_ok=False)
    cfg = config()
    row = records(cfg, 'train')[0]
    with h5py.File(row['path']) as h:
        truth = torch.tensor(h['spins'][:2, :11], dtype=torch.float32).unsqueeze(-2)
        physical = torch.tensor(h['time_ps'][:11], dtype=torch.float32)
    physical -= physical[0].clone()
    truth = F.normalize(truth, dim=-1)
    cell = CellHamiltonian((truth.shape[2],), 1, [(0, 0, (1,), 1)], periodic=(False,))
    net = SpinGraphNet((False,), 16, 2)
    frame = torch.eye(3)
    # Fixed numerical conditioning scale, NOT a fitted training-set standardizer
    # and NOT a redefinition of physical temperature or time.
    temperature = cfg['temperature_K']/100.
    rng = torch.Generator().manual_seed(42)
    source = F.normalize(torch.randn(truth.shape, generator=rng), dim=-1)
    source[:, 0] = truth[:, 0]
    tau = torch.tensor([.3, .7])
    state, target, _ = path_interpolate(source, truth, tau, rng)
    optimizer = torch.optim.Adam(net.parameters(), lr=.002)
    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        velocity = net(state, tau, truth[:, 0], cell, frame, temperature, physical)
        loss = (velocity[:, 1:]-target[:, 1:]).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite loss')
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    initial = truth[:1, 0].expand(4, *truth.shape[2:])
    before = time.perf_counter()
    generated = generate(net, initial, cell, frame, temperature, physical, seed=43, steps=16)
    sample_seconds = time.perf_counter()-before
    norm_error = float((generated.norm(dim=-1)-1).abs().max())
    anchor_error = float((generated[:, 0]-initial).abs().max())
    assert losses[-1] < losses[0] and norm_error < 1e-6 and anchor_error == 0
    report = dict(status='software_smoke_passed', scientific_certification=False,
        source_file=str(row['path']), parent_id=row['parent_id'], split='train',
        reference_branches=2, generated_branches=4, frames=len(physical),
        duration_ps=float(physical[-1]), output_dt_ps=float(physical[1]),
        temperature_K=cfg['temperature_K'], temperature_input_scale_K=100.,
        shape=list(cell.shape), periodic=list(cell.periodic), width=16, depth=2,
        updates=50, seed=41, flow_steps=16, losses=losses,
        max_norm_error=norm_error, anchor_error=anchor_error,
        batch_generation_seconds=sample_seconds, total_seconds=time.perf_counter()-started,
        limitation='Two training branches and one fixed flow interpolation batch; no held-out accuracy or speedup claim')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(out/'paths.npz', reference=truth.numpy(), generated=generated.numpy(), time_ps=physical.numpy())
    torch.save(dict(model=net.state_dict(), metadata=report), out/'smoke_only.pt')
    print(json.dumps({k:v for k,v in report.items() if k != 'losses'}, indent=2))


if __name__ == '__main__':
    main()
