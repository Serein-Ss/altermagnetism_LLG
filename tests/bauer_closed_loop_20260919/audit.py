"""Full integrity audit; development-only numerical screen; training-only baseline fit."""
import hashlib
import json
import h5py
import numpy as np
from .run import HERE, ROOT, config, dump
from .metrics import compare, summary


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    cfg = config()
    parents, fingerprints, seeds = {}, set(), set()
    count = 0
    max_norm = 0.
    numerical, temperatures, diffusion = [], [], []
    for length in cfg['sizes']:
        root = ROOT/cfg['data']/f'L{length}'
        for row in json.loads((root/'manifest.json').read_text()):
            path = root/row['file']
            assert digest(path) == row['sha256']
            assert parents.setdefault(row['parent_id'], row['split']) == row['split']
            with h5py.File(path) as h:
                assert h.attrs['complete']
                physical = json.loads(h.attrs['config_json'])['physics']
                assert physical['boundary'] == 'open' and physical['external_field'] == 0.
                assert physical['exchange_meV'] == 50. and physical['anisotropy_over_J'] == .1
                assert physical['damping'] == .1 and physical['theta'] == .11
                meta = json.loads(h.attrs['parent_json'])
                assert meta['parent_id'] == row['parent_id'] and meta['split'] == row['split']
                assert h['spins'].shape == (row['branches'], 1001, length, 3)
                assert np.allclose(h['time_ps'][:], np.arange(1001)*.1)
                assert h.attrs['arm'] == row['arm']
                newseeds = h['seeds'][:].tolist()
                assert len(set(newseeds)) == len(newseeds) and not seeds.intersection(newseeds)
                seeds.update(newseeds)
                if row['arm'] == 'base':
                    fingerprint = hashlib.sha256(h['initial'][:].tobytes()).hexdigest()
                    assert fingerprint not in fingerprints
                    fingerprints.add(fingerprint)
                for b in range(row['branches']):
                    s = h['spins'][b].astype('f8')
                    assert np.isfinite(s).all()
                    max_norm = max(max_norm, float(abs(np.linalg.norm(s, axis=-1)-1).max()))
                    assert np.max(abs(s[0]-h['initial'][:])) < 1e-6
                if row['split'] == 'train' and row['arm'] == 'base':
                    temperatures.append(float(h.attrs['temperature_K']))
                    s = h['spins'][:, :11].astype('f8')
                    angle = np.arccos(np.clip((s[:, 1:]*s[:, :-1]).sum(-1), -1, 1))
                    diffusion.append(float((angle*angle).mean()/2))
                if row['split'] == 'dev' and row['arm'] == 'base':
                    fine = path.with_name(path.name.replace('_base.h5', '_fine.h5'))
                    with h5py.File(fine) as f:
                        a, b = h['spins'][:, :11].astype('f8'), f['spins'][:, :11].astype('f8')
                        assert np.max(abs(f['initial'][:]-h['initial'][:])) < 1e-12
                    numerical.append(dict(parent_id=row['parent_id'], L=length, family=row['family'], **compare(a, b)))
                count += 1
    assert max_norm < 1e-6
    report = dict(integrity_pass=True, files=count, parents=len(parents), future_seeds=len(seeds),
                  max_stored_norm_error=max_norm, train_temperature_mean=float(np.mean(temperatures)),
                  train_temperature_scale=max(float(np.std(temperatures)), 1.),
                  random_baseline_sigma=float(np.sqrt(np.mean(diffusion))),
                  numerical_parents=numerical, numerical_statistics=summary(numerical, cfg['margins']),
                  reference_status='DEVELOPMENT_STEP_SCREEN_NOT_DISTRIBUTION_CERTIFIED',
                  test_use='integrity only, no selection or thresholds',
                  baseline_fit='isotropic spherical random walk; sigma estimated from train 0.1 ps increments only')
    dump(HERE/'audit.json', report)
    print(json.dumps({k: report[k] for k in ['integrity_pass', 'files', 'parents', 'max_stored_norm_error']}), flush=True)


if __name__ == '__main__':
    main()
