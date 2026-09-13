"""Bounded streaming inventory of completed Bauer references, never a certificate.

No basin threshold is invented. Zero crossings are descriptive only; neither
completed reversals nor censored lifetimes are inferred from them.
"""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import yaml


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def audit(path, output):
    path, output = Path(path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(source=str(path), scope='descriptive_reference_inventory_not_P1_P2',
                  production_enabled=False, training_enabled=False)
    if not path.exists():
        report.update(status='HOLD_missing_completed_reference', numerical_failure_is_censoring=False)
    else:
        with h5py.File(path, 'r') as h:
            if not h.attrs.get('complete', False):
                raise ValueError('incomplete reference cannot enter physical statistics')
            data = h['spins']; frames, chains, length, xyz = data.shape
            if xyz != 3: raise ValueError('expected [frame,chain,site,3]')
            time = h['time'][:]
            if len(time) != frames or not np.isfinite(time).all() or (np.diff(time) <= 0).any():
                raise ValueError('invalid physical time grid')
            magnetization = np.empty((frames, chains, 3))
            spatial_std = np.empty((frames, chains))
            energy = np.empty((frames, chains))
            initial = data[0]
            max_norm = 0.
            bins = np.linspace(-1, 1, 101)
            histogram = np.zeros((chains, 100), dtype=np.int64)
            for start in range(0, frames, 128):
                x = data[start:start+128]
                if not np.isfinite(x).all(): raise ValueError('nonfinite spins: STOP, not right censoring')
                end = start+len(x)
                max_norm = max(max_norm, float(np.abs(np.linalg.norm(x, axis=-1)-1).max()))
                magnetization[start:end] = x.mean(2)
                spatial_std[start:end] = x[..., 2].std(2)
                energy[start:end] = (-(x[:, :, :-1]*x[:, :, 1:]).sum((2, 3))-.1*(x[..., 2]**2).sum(2))/length
                # Retained late-time occupancy, not proof of stationary mixing.
                lo = max(0, frames//2-start)
                for chain in range(chains):
                    histogram[chain] += np.histogram(magnetization[start+lo:end, chain, 2], bins=bins)[0]
            mz = magnetization[..., 2]
            selection = np.unique(np.linspace(0, frames-1, min(1001, frames)).astype(int))
            with (output/'observables.npz').open('xb') as stream:
                np.savez_compressed(stream, time=time, magnetization=magnetization,
                    energy_per_spin=energy, spatial_std_z=spatial_std, initial=initial,
                    late_histogram=histogram, histogram_edges=bins,
                    spatial_preview_time=time[selection], spatial_preview_sz=data[selection][..., 2])
            report.update(status='HOLD_basin_and_literature_acceptance_pending', frames=frames,
                chains=chains, length=length, dt=float(h.attrs['dt']),
                method=str(h.attrs.get('method', 'see_frozen_configuration')),
                time_unit='hbar/E0', saved_interval_max=float(np.diff(time).max()),
                zero_crossing_counts=((mz[1:]*mz[:-1]) < 0).sum(0).tolist(),
                negative_endpoint=(mz[-1] < 0).tolist(), maximum_norm_error=max_norm,
                committed_switch=None, basin_threshold=None, dwell_time=None,
                limitation='Zero crossings/occupancy are not reversal, equilibrium, or lifetime certification',
                initial_sha256=hashlib.sha256(initial.tobytes()).hexdigest())
        report['source_sha256'] = digest(path)
    with (output/'report.json').open('x') as stream: json.dump(report, stream, indent=2)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    args = p.parse_args()
    m = json.loads(Path(args.manifest).read_text())
    reports = []
    for row in m['reference_runs']:
        if digest(row['config']) != row['config_sha256']:
            raise ValueError('reference config changed')
        c = yaml.safe_load(Path(row['config']).read_text())
        r = c['reduced']
        if (r['equation_convention'] != 'bauer_ll' or r['exchange'] != 1.
                or r['anisotropy'] != .1 or r['theta'] != .11 or r['alpha'] != .1):
            raise ValueError('audit energy/time convention only covers frozen original Bauer case')
        reports.append(audit(row['path'], Path(m['output'])/'reference_audit'/row['run_id']))
    with (Path(m['output'])/'reference_review.json').open('x') as stream:
        json.dump(dict(status='HOLD_scientific_review_required', reports=reports,
                       pending=m['pending'], production_enabled=False), stream, indent=2)


if __name__ == '__main__': main()
