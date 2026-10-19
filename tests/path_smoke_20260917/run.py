"""Bounded software test. Pilot data are NOT scientifically certified."""
import argparse
import json
import os
from pathlib import Path
import time as clock
import h5py
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tests.literature_preflight_20260917.models import build
from tests.path_smoke_20260917.model import PathNet, neighbors, generate
from scripts.model.gate_f import path_interpolate

BASE = Path(__file__).resolve().parent
SOURCE = BASE.parent / 'literature_preflight_20260917'


def contracts(net, config, name, device):
    cell, _, _ = build(config, name, shape={'bauer': [7], 'miltat': [5, 4], 'hirst': [4, 4, 4]}[name])
    cell.periodic = (True,) * len(cell.shape)
    dtype = next(net.parameters()).dtype
    x = torch.nn.functional.normalize(torch.randn((1, 3, *cell.shape, cell.basis, 3), device=device, dtype=dtype), dim=-1)
    frame = torch.eye(3, device=device, dtype=dtype)
    q, _ = torch.linalg.qr(torch.randn(3, 3, device=device, dtype=dtype))
    q[:, 0] *= torch.linalg.det(q)
    tau, time = x.new_tensor([.4]), x.new_tensor([0., .5, 1.])
    with torch.no_grad():
        v = net(x, tau, x[:, 0], cell, frame, .5, time)
        vr = net(x @ q, tau, x[:, 0] @ q, cell, frame @ q, .5, time)
        rotations = (vr - v @ q).abs().max().item()
        translations = []
        for axis in range(len(cell.shape)):
            rolled = x.roll(1, 2 + axis)
            vt = net(rolled, tau, rolled[:, 0], cell, frame, .5, time)
            translations.append((vt - v.roll(1, 2 + axis)).abs().max().item())
        a = generate(net, x[:, 0], cell, frame, .5, time, source=x, steps=4)
        b = generate(net, x[:, 0] @ q, cell, frame @ q, .5, time, source=x @ q, steps=4)
        generation_rotation = (b - a @ q).abs().max().item()
        # Independent explicit edge sum for open-boundary aggregation.
        cell.periodic = (False,) * len(cell.shape)
        expected, count = torch.zeros_like(x), torch.zeros_like(x[..., :1])
        for a0, b0, delta, _ in cell.templates:
            for index in np.ndindex(*cell.shape):
                other = tuple(i + d for i, d in zip(index, delta))
                if not all(0 <= j < n for j, n in zip(other, cell.shape)):
                    continue
                ia, ib = (slice(None), slice(None), *index, a0), (slice(None), slice(None), *other, b0)
                expected[ia] += x[ib]; expected[ib] += x[ia]
                count[ia] += 1; count[ib] += 1
        open_error = (neighbors(x, cell) - expected / count.clamp_min(1)).abs().max().item()
    result = dict(rotation_velocity=rotations, rotation_generated_path=generation_rotation,
                  periodic_translation=translations, open_boundary_aggregation=open_error,
                  tangent_error=(v * x).sum(-1).abs().max().item())
    result['passed'] = max(rotations, generation_rotation, *translations, open_error, result['tangent_error']) < 5e-5
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--system', choices=['bauer', 'miltat', 'hirst'], required=True)
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--updates', type=int, default=80)
    args = p.parse_args()
    out = BASE / 'results' / args.system
    out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((SOURCE / 'config.yaml').read_text())
    report = json.loads((SOURCE / 'results' / args.system / 'report.json').read_text())
    with h5py.File(SOURCE / 'results' / args.system / 'trajectory.h5') as f:
        data = f['spins'][:]
        times = f['time_ps'][:]
    integrity = dict(shape=list(data.shape), finite=bool(np.isfinite(data).all()),
                     norm_error=float(np.abs(np.linalg.norm(data, axis=-1) - 1).max()),
                     time_increasing=bool((np.diff(times) > 0).all()), time_ps=[float(times[0]), float(times[-1])],
                     certification_holds=report['certification_holds'], scientific_status='NOT_CERTIFIED')
    (out / 'data_audit.json').write_text(json.dumps(integrity, indent=2))
    assert integrity['finite'] and integrity['norm_error'] < 1e-10 and integrity['time_increasing']
    if args.audit_only:
        print(json.dumps(integrity)); return
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Training requires Slurm CUDA allocation')
    torch.set_num_threads(1)
    torch.manual_seed(20260917)
    device = 'cuda'
    cell, reduced, _ = build(config, args.system)
    # Full spatial domains; independent trajectory IDs reserved, no crop leakage.
    train_ids, dev_id = list(range(data.shape[1] - 1)), data.shape[1] - 1
    net = PathNet().to(device)
    frame = torch.eye(3, device=device)
    # One temperature per pilot: no claim of learned temperature dependence.
    temp = float(reduced['theta'])
    t = torch.tensor(times[:3] / times[2], dtype=torch.float32, device=device)
    rng = torch.Generator(device=device).manual_seed(819)
    opt = torch.optim.Adam(net.parameters(), lr=.002)
    history = []
    start = clock.perf_counter()
    def loss_for(y, generator):
        z = torch.nn.functional.normalize(torch.randn(y.shape, device=device, generator=generator), dim=-1)
        z[:, 0] = y[:, 0]
        tau = torch.rand(1, device=device, generator=generator)
        state, velocity, _ = path_interpolate(z, y, tau, generator)
        return (net(state, tau, y[:, 0], cell, frame, temp, t)[:, 1:] - velocity[:, 1:]).square().sum(-1).mean()
    dev = torch.tensor(data[:3, dev_id][None], dtype=torch.float32, device=device)
    with torch.no_grad():
        before = float(loss_for(dev, torch.Generator(device=device).manual_seed(42)))
    for step in range(args.updates):
        parent = train_ids[step % len(train_ids)]
        offset = step % (len(data) - 2)
        y = torch.tensor(data[offset:offset + 3, parent][None], dtype=torch.float32, device=device)
        opt.zero_grad(set_to_none=True)
        loss = loss_for(y, rng)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
        opt.step()
        history.append(float(loss))
        if step % 10 == 0: print(args.system, step, history[-1], flush=True)
    with torch.no_grad():
        after = float(loss_for(dev, torch.Generator(device=device).manual_seed(42)))
        generated = torch.cat([generate(net, dev[:, 0], cell, frame, temp, t, seed=s) for s in (71, 72)], 0)
    checks = contracts(net, config, args.system, device)
    gen = generated.cpu().numpy()
    np.savez_compressed(out / 'generated.npz', spins=gen, time_ps=times[:3], reference=data[:3, dev_id], latent_seeds=[71, 72])
    torch.save({'model': net.state_dict(), 'width': 24, 'system': args.system, 'temperature_reduced': temp,
                'train_ids': train_ids, 'dev_id': dev_id, 'normalized_time_ps_scale': float(times[2])}, out / 'checkpoint.pt')
    checks.update(unit_norm_error=float(np.abs(np.linalg.norm(gen, axis=-1) - 1).max()),
                  initial_error=float(np.abs(gen[:, 0] - dev[:, 0].cpu().numpy()).max()),
                  finite=bool(np.isfinite(gen).all()), latent_pair_rms=float(np.sqrt(np.mean((gen[0] - gen[1]) ** 2))))
    checks['passed'] = checks['passed'] and checks['unit_norm_error'] < 1e-6 and checks['initial_error'] == 0 and checks['finite']
    summary = dict(system=args.system, device=torch.cuda.get_device_name(), job=os.environ['SLURM_JOB_ID'],
                   updates=args.updates, train_ids=train_ids, dev_id=dev_id, heldout_fixed_loss_before=before,
                   heldout_fixed_loss_after=after, checks=checks, seconds=clock.perf_counter() - start,
                   scientific_status='SOFTWARE_SMOKE_ONLY_NOT_DISTRIBUTION_CERTIFIED', loss=history,
                   limitations=['single temperature', 'only two future frames', 'minimal local two-hop prototype, not final multiscale architecture',
                                'joint frame/spin SO(3) covariance, not fixed-crystal spin symmetry', 'pilot reference not literature-certified'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax[0, 0].plot(history); ax[0, 0].set(title='Training flow-matching loss', xlabel='Update', ylabel='Squared tangent error')
    for component, color in enumerate(['tab:blue', 'tab:orange', 'tab:green']):
        ref = data[:3, dev_id].reshape(3, -1, 3).mean(1)
        pred = gen.reshape(2, 3, -1, 3).mean(2)
        ax[0, 1].plot(times[:3], ref[:, component], color=color, label='Reference ' + 'xyz'[component])
        ax[0, 1].plot(times[:3], pred[0, :, component], '--', color=color, label='Generated ' + 'xyz'[component])
    ax[0, 1].set(title='Mean magnetization (one held-out path)', xlabel='Time (ps)', ylabel='Mean spin'); ax[0, 1].legend(fontsize=7)
    for panel, values, title in [(ax[1, 0], data[2, dev_id], 'Reference final sz'), (ax[1, 1], gen[0, 2], 'Generated final sz')]:
        z = values[..., 2]
        if args.system == 'bauer':
            panel.plot(z[:, 0]); panel.set(xlabel='Chain site', ylabel='sz', ylim=(-1, 1))
        else:
            z = z[..., 0] if args.system == 'miltat' else z[:, :, z.shape[2] // 2, 0]
            im = panel.imshow(z.T, origin='lower', vmin=-1, vmax=1, cmap='coolwarm'); fig.colorbar(im, ax=panel, label='sz')
            panel.set(xlabel='Cell x', ylabel='Cell y')
        panel.set_title(title + (' (basis 0; middle z slice)' if args.system == 'hirst' else ''))
    fig.suptitle(args.system + ': training/generation smoke test ONLY; no statistical certification')
    fig.savefig(out / 'overview.png', dpi=160); plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)
    if not checks['passed']: raise RuntimeError('Constraint contract failed; see summary.json')


if __name__ == '__main__':
    main()
