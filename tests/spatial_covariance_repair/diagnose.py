import json
from pathlib import Path
import h5py
import numpy as np
from tests.chain_reversal_repair.run import verify
from tests.bauer_p2_20260923.common import dump
from .metrics import compare

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def main():
    verify();rows=[];full=[]
    for index in range(6):
        source=ROOT/f'tests/chain_reversal_repair/train/{index:02d}'
        task=json.loads((source/'complete.json').read_text())['task']
        with h5py.File(source/'intermediate.h5') as h:
            assert h.attrs['complete']
            for name,g in h.items():
                meta=json.loads(g.attrs['metadata'])
                with h5py.File(ROOT/f'tests/intermediate_restart_repair/reference/{meta["parent_index"]:02d}/paths.h5') as ref:
                    a=g['spins'][:];r=ref[f'{name}/fine/spins'][:];base=ref[f'{name}/base/spins'][:]
                rows.append(dict(task=task,metadata=meta,metrics=compare(a,r),reference_numerical=compare(base,r)))
        with h5py.File(source/'paths.h5') as h:
            assert h.attrs['complete']
            for name,g in h['dev'].items():
                parent=json.loads(g.attrs['parent']);idx=int(name.removeprefix('parent'))
                with h5py.File(ROOT/f'tests/reference_resolution/reference/{idx:02d}/paths.h5') as ref:
                    r=ref['fine/spins'][:];base=ref['base/spins'][:]
                a=g['midpoint128'][:]
                for k in (1,3,5,10):
                    full.append(dict(task=task,parent=parent,time_ps=k*.1,
                        metrics=compare(a[:,[0,k]],r[:,[0,k]]),
                        reference_numerical=compare(base[:,[0,k]],r[:,[0,k]])))
    assets=HERE/'diagnosis';assets.mkdir(exist_ok=True)
    dump(assets/'report.json',dict(conditional=rows,full=full,scope='Frozen checkpoint/no new samples, repeated DEV exploratory, related states not independent parents',scientific_certification=False))
    summary=[]
    for arm in ('control','reversal'):
        for lag in (0,1,2,4,8):
            a=[q for row in rows if row['task']['repair']==arm for q in row['metrics']['lags'] if q['lag']==lag]
            b=[q for row in rows if row['task']['repair']==arm for q in row['reference_numerical']['lags'] if q['lag']==lag]
            summary.append(dict(arm=arm,lag=lag,states_including_correlated=len(a),
                covariance_rmse=float(np.mean([x['covariance_rmse'] for x in a])),
                covariance_squared_bias_crossfold=float(np.mean([x['covariance_squared_bias_crossfold'] for x in a])),
                reference_squared_bias_crossfold=float(np.mean([x['covariance_squared_bias_crossfold'] for x in b])),
                signed_trace_difference=float(np.mean([x['generated_trace']-x['reference_trace'] for x in a]))))
    dump(assets/'summary.json',summary);print(json.dumps(summary))


if __name__=='__main__':main()
