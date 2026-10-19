import numpy as np
import pytest

from tests.experiments.independent_candidate_analysis import compare
from tests.experiments.verify_fresh_model_audit import profiles
from tests.experiments.verify_independent_candidate import verify_metrics


def test_bootstrap_points_match_independent_profiles_and_replay():
    rng = np.random.default_rng(80)
    a,b = [rng.normal(size=(n,4,5,3)) for n in [8,10]]
    a /= np.linalg.norm(a,axis=-1,keepdims=True)
    b /= np.linalg.norm(b,axis=-1,keepdims=True)
    arrays,metrics = compare(a,b,np.random.default_rng(4),20)
    replay,other = compare(a,b,np.random.default_rng(4),20)
    am,av,ab,at = profiles(a); bm,bv,bb,bt = profiles(b)
    expected = dict(mean_rmse=np.sqrt(((am-bm)**2).mean((1,2))),
        variance_difference=av-bv, bond_rmse=np.sqrt(((ab-bb)**2).mean(1)),
        temporal_connected_difference=at-bt)
    assert set(metrics)==set(expected) and metrics==other
    for key,value in expected.items():
        np.testing.assert_allclose(metrics[key]['estimate'],value,atol=1e-12)
        np.testing.assert_array_equal(arrays[key],replay[key])
        np.testing.assert_allclose(metrics[key]['ci95'],np.quantile(arrays[key],[.025,.975],axis=0))
    verify_metrics(a,b,metrics,arrays)
    metrics['mean_rmse']['estimate'][0] += 1
    with pytest.raises(AssertionError):
        verify_metrics(a,b,metrics,arrays)
