import numpy as np
import pytest
from tests.experiments.flow_film_statistics import moment_errors, validate_rows


def test_prefix_excludes_future_pair_endpoint_and_unpaired_raw_paths():
    r=dict(magnetization_mean=np.zeros((201,3)),temporal_dot_lag8=np.zeros(193),
           magnetization_per_path=np.zeros((16,201,3)))
    g={k:v.copy() for k,v in r.items()}; g['magnetization_per_path']=np.ones((128,201,3))
    g['temporal_dot_lag8'][3:]=1
    out=moment_errors(g,r)
    assert out['frame10/temporal_dot_lag8']==0
    assert out['frame200/temporal_dot_lag8']>0
    assert not any('per_path' in k for k in out)


def test_complete_group_gate():
    r=dict(status='COMPLETE_DEVELOPMENT_GENERATION_NOT_CERTIFIED',arm='control',seed=11,
           rows=[dict(family=f,theta=t) for f in (0,1) for t in (.11,.13)])
    validate_rows(r,'control',11)
    r['rows'][-1]=r['rows'][0]
    with pytest.raises(ValueError): validate_rows(r,'control',11)
