import numpy as np
import pytest
from tests.experiments.independent_comparison_statistics import prefix_discrepancies,compare_ensembles


def test_prefix_excludes_future_endpoints():
    ref=dict(magnetization_mean=np.zeros((1,21,3)),temporal_dot_lag8=np.zeros((1,13)))
    g={k:v.copy() for k,v in ref.items()}
    g['temporal_dot_lag8'][:,3:]=100 # start3 has endpoint11, outside prefix10.
    result=prefix_discrepancies(g,ref,ref,ends=(2,10,20))
    assert 'frame2/temporal_dot_lag8/excess_rmse' not in result
    assert result['frame10/temporal_dot_lag8/excess_rmse'][0]==0
    assert result['frame20/temporal_dot_lag8/excess_rmse'][0]>0
    with pytest.raises(ValueError):prefix_discrepancies(g,ref,ref,ends=(21,))


def test_bootstrap_replay_and_batch_partition_invariance():
    rng=np.random.default_rng(19)
    groups=[rng.normal(size=(4,21,5,3)) for _ in range(3)]
    for a in groups:a/=np.linalg.norm(a,axis=-1,keepdims=True);a[:,0]=[0,0,1]
    kwargs=dict(reference_seed=100,generated_seed=102,replicates=7,ends=(10,20),temporal_lags=(1,8),spatial_lags=(1,4))
    summary,a=compare_ensembles(*groups,batch_size=3,**kwargs)
    _,b=compare_ensembles(*groups,batch_size=7,**kwargs)
    for key in a:np.testing.assert_allclose(a[key],b[key],atol=1e-14)
    assert all(len(row['descriptive_percentile_interval95'])==2 for row in summary.values())
    # Reference draws must remain identical when only the generated paths change.
    altered=groups[0].copy();altered[:,1:]*=-1
    _,c=compare_ensembles(altered,*groups[1:],batch_size=3,**kwargs)
    for key in a:
        if key.endswith('/reference_reference_rmse'):np.testing.assert_array_equal(a[key],c[key])
