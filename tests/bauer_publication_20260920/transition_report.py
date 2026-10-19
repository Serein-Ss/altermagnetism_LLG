"""Postprocess all frozen temporal-factorization models without selecting on tests."""
import json
from pathlib import Path
import h5py
import numpy as np
from tests.bauer_publication_20260920.temporal_floor import paired_comparisons
from tests.bauer_transition_20260920.run import config, identity

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent/'bauer_transition_20260920'
REFERENCE = HERE.parent/'bauer_two_stage_20260919/data'


def collect_model(name, cfg):
    folder = EXPERIMENT/'evaluation'/name
    completion = json.loads((folder/'complete.json').read_text())
    if completion['model'] != name:
        raise ValueError('Model completion identity mismatch')
    rows = []
    for length in cfg['evaluation_sizes']:
        with h5py.File(folder/f'L{length}.h5') as generated:
            if not generated.attrs['complete'] or generated.attrs['model'] != name:
                raise ValueError('Incomplete or incorrect model archive')
            for parent in range(6):
                with h5py.File(REFERENCE/f'L{length}'/f'test_future_{parent}.h5') as reference:
                    if not reference.attrs['complete']:
                        raise ValueError('Incomplete reference')
                    for horizon in [1., cfg['evaluation_duration_ps']]:
                        end = round(horizon/cfg['save_ps'])+1
                        t = generated['time_ps'][:end]
                        if len(t) != end or not np.allclose(t, reference['time_ps'][:end]) or not np.isclose(t[-1], horizon):
                            raise ValueError('Time grid mismatch')
                        metrics = paired_comparisons(generated[f'parent_{parent}'][:, :end],
                            reference['spins'][:, :end], reference['seeds'][:], cfg['branches'])
                        rows.append(dict(model=name, L=length, parent=parent, horizon_ps=horizon, **metrics))
    return rows


def main():
    cfg = config(); rows = []
    for index in range(6):
        variant, seed = identity(index, cfg)
        rows.extend(collect_model(f'{variant}_{seed}', cfg))
    result = dict(scientific_certification=False, selected_model=None,
                  all_six_models_required=True, rows=rows,
                  interpretation='Equal branch-count descriptive comparisons; no acceptance threshold or final holdout claim.')
    out = HERE/'results'/'transition_temporal_comparison.json'
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Completed', len(rows), 'parent/horizon comparisons across all six models')


if __name__ == '__main__':
    main()
