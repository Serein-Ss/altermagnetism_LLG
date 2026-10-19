"""Descriptive no-learning controls for short conditional rollouts."""
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest, save


def random_walk(initial, n, frames, scale, seed):
    rng = np.random.default_rng(seed)
    x = np.broadcast_to(initial, (n, *initial.shape)).copy()
    paths = [x.copy()]
    for _ in range(frames-1):
        x = x+scale*rng.normal(size=x.shape)
        x = x/np.linalg.norm(x, axis=-1, keepdims=True)
        paths.append(x.copy())
    return np.stack(paths, axis=1)


def profiles(paths):
    p = np.asarray(paths, dtype=np.float64)
    if p.ndim != 4 or p.shape[0] < 2 or not np.isfinite(p).all():
        raise ValueError('invalid paths')
    return dict(mean_increment=(p-p[:, :1]).mean(0),
                site_variance=p.var(0, ddof=1).sum(-1),
                nearest_bond=(p[:, :, :-1]*p[:, :, 1:]).sum(-1).mean(0),
                temporal_dot=(p*p[:, :1]).sum(-1).mean(0))


def compare(p, reference):
    return {k: np.sqrt(np.mean((v-reference[k])**2, axis=tuple(range(1, v.ndim)))).tolist()
            for k, v in p.items()}


def run():
    root = Path('tests/results/local_transition_baselines')
    raw = Path('tests/data/local_transition_baselines')
    if root.exists() or raw.exists():
        raise FileExistsError('preserve previous baseline analysis')
    root.mkdir(); raw.mkdir()
    report = dict(status='RUNNING', rows=[], code_sha256=digest(__file__),
        scope='Descriptive anchor-conditional curves, no confidence intervals or pass decision. '
        'Persistence has zero stochastic diversity and cannot be a correct finite-temperature path model. '
        'Random walk uses pooled training-only latent scale; no Hamiltonian, no learned conditional drift. '
        'Baselines diagnose added value, not replacements. Reference and generated noise paths are unpaired. '
        'Train rows reuse training references. Development is not an independent final test.')
    for seed in [11, 22, 33]:
        source = Path(f'tests/results/local_transition_training/{seed}/summary.json')
        training = json.loads(source.read_text())
        assert training['status'] == 'COMPLETE_TRAINING_AND_GENERATION_NOT_CERTIFIED'
        assert len(training['rows']) == 12 and len({r['anchor'] for r in training['rows']}) == 12
        for row in training['rows']:
            assert digest(row['path']) == row['sha256']
            assert digest(row['reference_path']) == row['reference_sha256']
            with np.load(row['path']) as d:
                model, times = d['spins'].astype(float), d['reduced_time']
            with np.load(row['reference_path']) as d:
                reference, rt = d['spins'][:, ::4], d['reduced_time'][::4]
            assert model.shape == reference.shape == (256, 9, 25, 3)
            np.testing.assert_allclose(times, rt, atol=1e-12, rtol=0)
            np.testing.assert_allclose(model[:, 0], reference[:, 0], atol=2e-7, rtol=0)
            noise_seed = 970000000+seed*1000+row['anchor']
            walk = random_walk(reference[0, 0], 256, 9, training['latent_scale'], noise_seed)
            persistence = np.broadcast_to(reference[:, :1], reference.shape)
            rp = profiles(reference)
            arrays = {'time': times}
            errors = {}
            for name, paths in [('reference', reference), ('model', model), ('random_walk', walk), ('persistence', persistence)]:
                pp = profiles(paths)
                arrays.update({name+'_'+k: v for k, v in pp.items()})
                errors[name] = compare(pp, rp)
            path = root/f"seed{seed}_anchor{row['anchor']}.npz"
            np.savez_compressed(path, **arrays)
            walk_path = raw/f"seed{seed}_anchor{row['anchor']}.npz"
            np.savez_compressed(walk_path, spins=walk, reduced_time=times)
            report['rows'].append(dict(seed=seed, anchor=row['anchor'], split=row['split'], state=row['state'],
                source_sha256=digest(source), profile_path=str(path), profile_sha256=digest(path),
                walk_path=str(walk_path), walk_sha256=digest(walk_path), walk_seed=noise_seed, errors=errors))
            save(root/'summary.json', report)
    report['status'] = 'COMPLETE_DESCRIPTIVE_BASELINES_NOT_CERTIFIED'
    save(root/'summary.json', report)


if __name__ == '__main__':
    run()
