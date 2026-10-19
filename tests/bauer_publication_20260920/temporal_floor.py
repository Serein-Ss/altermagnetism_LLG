"""Equal-branch descriptive temporal comparisons; no pass threshold or model selection."""
import json
from pathlib import Path
import h5py
import numpy as np
from tests.bauer_transition_20260920.run import temporal_errors

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'bauer_two_stage_20260919'


def paired_comparisons(generated, reference, seeds, branches=16):
    g, r = np.asarray(generated, dtype=float), np.asarray(reference, dtype=float)
    if len(g) != branches or len(r) != 2*branches:
        raise ValueError('Require n generated and 2n independent reference branches')
    if len(seeds) != len(r) or len(set(map(int, seeds))) != len(r):
        raise ValueError('Reference branch seeds must be distinct')
    if g.shape[1:] != r.shape[1:] or not np.isfinite(g).all() or not np.isfinite(r).all():
        raise ValueError('Incompatible or nonfinite paths')
    if not np.allclose(r[:, 0], r[0, 0], atol=2e-7, rtol=0) or not np.allclose(g[:, 0], r[0, 0], atol=2e-7, rtol=0):
        raise ValueError('All paths must share the same initial parent')
    a, b = r[:branches], r[branches:]
    return dict(branches_per_group=branches,
                model_vs_reference_a=temporal_errors(g, a),
                model_vs_reference_b=temporal_errors(g, b),
                reference_b_vs_a=temporal_errors(b, a),
                reference_a_vs_b=temporal_errors(a, b))


def main():
    rows = []
    for length in [16, 25, 32, 50, 100]:
        with h5py.File(SOURCE/'generated'/f'L{length}.h5') as model:
            if not model.attrs['complete']:
                raise ValueError('Incomplete generated archive')
            for parent in range(6):
                with h5py.File(SOURCE/'data'/f'L{length}'/f'test_future_{parent}.h5') as ref:
                    if not ref.attrs['complete']:
                        raise ValueError('Incomplete reference archive')
                    seeds = ref['seeds'][:]
                    for horizon in [1., 10., 100.]:
                        end = round(horizon/.1)+1
                        if len(ref['time_ps']) < end or len(model['time_ps']) < end:
                            raise ValueError('Requested horizon unavailable')
                        if not np.allclose(ref['time_ps'][:end], model['time_ps'][:end]):
                            raise ValueError('Mismatched observation times')
                        values = paired_comparisons(model[f'RG_{parent}'][:, :end], ref['spins'][:, :end], seeds)
                        rows.append(dict(L=length, parent=parent, horizon_ps=horizon, **values))
    result = dict(source=str(SOURCE), rows=rows, scientific_certification=False,
        notes=['Frozen old-model diagnosis, not current training results.',
               'Both directions retained because the relative error denominator uses its reference group.',
               'Two comparisons share generated paths; they are not independent replicates.',
               'A single disjoint reference split is descriptive, not an uncertainty interval or equivalence threshold.'])
    (HERE/'results'/'temporal_sampling_floor.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Completed descriptive comparisons:', len(rows))


if __name__ == '__main__':
    main()
