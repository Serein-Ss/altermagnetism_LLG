"""Freeze real intermediate states for paired model/LLG one-step diagnostics."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent


def frame_indices(times, requested):
    times = np.asarray(times)
    indices = [int(np.argmin(abs(times-t))) for t in requested]
    if not np.allclose(times[indices], requested, atol=1e-8, rtol=0):
        raise ValueError('Requested physical times absent from reference')
    return indices


def main():
    out = HERE/'data'
    out.mkdir(exist_ok=True)
    rows = []
    with h5py.File(out/'initial_states.h5', 'x') as dest:
        dest.attrs['complete'] = False
        for length in [16, 32, 100]:
            for parent in range(6):
                source = HERE.parent/'bauer_two_stage_20260919/data'/f'L{length}'/f'test_future_{parent}.h5'
                with h5py.File(source, 'r') as src:
                    assert src.attrs['complete']
                    requested = [0., 1., 5., 9.]
                    indices = frame_indices(src['time_ps'][:], requested)
                    for time_ps, index in zip(requested, indices):
                        state = src['spins'][0, index].astype(np.float64)
                        assert state.shape == (length, 3)
                        assert np.isfinite(state).all()
                        assert abs(np.linalg.norm(state, axis=-1)-1).max() < 1e-6
                        # Identical normalization for model and LLG input.
                        state /= np.linalg.norm(state, axis=-1, keepdims=True)
                        key = f'L{length}_parent{parent}_frame{index}'
                        dest.create_dataset(key, data=state)
                        rows.append(dict(key=key, L=length, parent=parent,
                                         time_ps=time_ps, source=str(source), branch=0,
                                         reference_seed=int(src['seeds'][0]),
                                         physics_config=json.loads(src.attrs['config_json']),
                                         state_sha256=hashlib.sha256(state.tobytes()).hexdigest()))
        dest.attrs['manifest_json'] = json.dumps(rows)
        dest.attrs['complete'] = True
    specification = dict(conditions=len(rows), future_ps=0.1, llg_dt_fs=0.25,
                         model_seeds=[11, 22, 33], model_branches=64, reference_branches=128,
                         flow_steps=64, rows=rows, scientific_certification=False,
                         interpretation='Exploratory diagnostic on previously inspected parents. '
                         'Times within a parent are correlated; not 72 independent parent replicates. '
                         'Future noises must be newly sampled; no outcome selection.')
    (out/'specification.json').write_text(json.dumps(specification, indent=2)+'\n')
    print('Prepared', len(rows), 'real-state conditions; no future simulation or training performed')


if __name__ == '__main__':
    main()
