"""Read-only full-file integrity and numerical-reference audit."""
import hashlib
import json
import collections
import h5py
import numpy as np
from .common import HERE, ROOT, cfg, dump, energy, comparison


def main():
    config = cfg()
    manifests = {k: [] for k in ('train', 'dev', 'diagnostic')}
    seeds, fingerprints, sources = set(), {}, {}
    numerical, dynamics = [], []
    max_norm = max_energy = 0.
    for size in config['sizes']:
        root = ROOT/config['data']/f'L{size}'
        assert (root/'complete.json').exists()
        frozen = json.loads((root/'frozen_config.json').read_text())
        assert frozen['physics']['boundary'] == 'open'
        for key in ('exchange_meV', 'anisotropy_over_J', 'damping', 'theta'):
            assert frozen['physics'][key] == config[key]
        assert frozen['physics']['external_field'] == 0
        for name, digest in json.loads((root/'sources.json').read_text()).items():
            path = ROOT/'tests'/name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
            sources[str(path.relative_to(ROOT))] = digest
        rows = json.loads((root/'manifest.json').read_text())
        parent_splits = {}
        for row in rows:
            path = root/row['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
            assert parent_splits.setdefault(row['parent_id'], row['split']) == row['split']
            with h5py.File(path) as h:
                assert h.attrs['complete'] and h.attrs['arm'] == row['arm']
                assert json.loads(h.attrs['config_json']) == frozen
                parent = json.loads(h.attrs['parent_json'])
                assert parent['split'] == row['split'] and parent['parent_id'] == row['parent_id']
                assert h['spins'].shape == (row['branches'], row['frames'], size, 3)
                assert np.allclose(np.diff(h['time_ps'][:]), config['save_ps'])
                assert h['time_ps'][0] == 0 and h['time_ps'][-1] == 100
                assert h.attrs['dt_fs'] == frozen['numerics']['dt_fs' if row['arm']=='base' else 'fine_dt_fs']
                assert abs(h.attrs['temperature_K']-config['temperature_K']) < 1e-5
                branch_seeds = set(h['seeds'][:].tolist())
                assert len(branch_seeds) == row['branches'] and not seeds.intersection(branch_seeds)
                seeds.update(branch_seeds)
                initial = h['initial'][:]
                digest = hashlib.sha256(initial.tobytes()).hexdigest()
                assert fingerprints.setdefault(digest, row['parent_id']) == row['parent_id']
                for i in range(row['branches']):
                    s = h['spins'][i].astype('f8')
                    assert np.isfinite(s).all()
                    assert np.max(abs(s[0]-initial)) < 1e-7
                    max_norm = max(max_norm, float(abs(np.linalg.norm(s, axis=-1)-1).max()))
                    max_energy = max(max_energy, float(abs(energy(s)-h['energy_per_spin_J'][i]).max()))
                item = dict(row, L=size, path=str(path.relative_to(ROOT)),
                    candidate_seed=parent['candidate_seed'], preparation_seed=parent['preparation_seed'])
                if row['arm'] == 'base':
                    manifests['diagnostic' if row['split']=='test' else row['split']].append(item)
                    if row['split'] == 'train':
                        s = h['spins'][:, :11].astype('f8')
                        dynamics.append(dict(parent_id=row['parent_id'], family=row['family'],
                            displacement_1ps=float(((s[:, -1]-s[:, 0])**2).sum(-1).mean())))
            if row['arm'] == 'fine':
                with h5py.File(path) as f, h5py.File(root/row['file'].replace('_fine', '_base')) as b:
                    numerical.append(dict(parent_id=row['parent_id'], L=size,
                        metrics=comparison(b['spins'][:, :11], f['spins'][:, :11], repeats=200)))
    assert max_norm < 1e-6 and max_energy < 1e-6
    for split, rows in manifests.items():
        dump(HERE/f'manifest_{split}.json', rows)
    dump(HERE/'numerical_reference.json', dict(comparisons=numerical,
        interpretation='Independent base/fine ensembles, dt=0.25/0.125 fs; CI not proof of equivalence. No pre-agreed physical tolerance.',
        sampling_unit='whole independent branch conditional on each fixed parent'))
    dump(HERE/'audit.json', dict(integrity_pass=True, sources=sources,
        counts={k: dict(parents=len(v), branches=sum(r['branches'] for r in v),
                       families=dict(collections.Counter(r['family'] for r in v))) for k,v in manifests.items()},
        max_norm_error=max_norm, max_stored_energy_error=max_energy,
        physics='H/J = -sum(open nearest neighbours) s_i dot s_j - 0.1 sum s_iz^2; no field, no dipole, no DMI',
        physical_time='tau=t J/hbar; hbar=0.6582119569 meV ps; T=theta J/kB',
        numerical_status='finite-ensemble dt comparison available; quantitative tolerance certification pending',
        final_holdout='none: historical test is diagnostic only',
        coverage='6 deliberately balanced preparation families, not natural occupancy; positive/negative names describe preparation, not guaranteed final sign',
        training_dynamics=dynamics))
    print(json.dumps(dict(counts={k:len(v) for k,v in manifests.items()}, max_norm=max_norm, max_energy=max_energy)))


if __name__ == '__main__':
    main()
