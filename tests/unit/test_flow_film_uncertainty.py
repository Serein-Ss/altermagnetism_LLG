import numpy as np
import pytest
from tests.experiments.flow_film_uncertainty import batched_errors, contrasts
from tests.experiments.flow_film_statistics import moment_errors
from tests.experiments.trajectory_bootstrap import weighted_observables


def test_batched_prefix_and_draw_axis():
    r=dict(magnetization_mean=np.zeros((2,201,3)),temporal_dot_lag8=np.zeros((2,193)))
    g={k:v.copy() for k,v in r.items()}; g['temporal_dot_lag8'][0,3:]=1
    g['magnetization_mean'][1]=2
    v=batched_errors(g,r)
    assert v['frame10/temporal_dot_lag8'][0]==0
    np.testing.assert_array_equal(v['frame200/magnetization_mean'],[0,2])


def test_paired_contrasts_cancel_identical_arms():
    values={'metric':np.arange(20.)}
    d=contrasts({a:values for a in ('control','frozen','flow_film')})
    assert all(np.array_equal(v,np.zeros(20)) for v in d.values())
    with pytest.raises(ValueError):
        contrasts(dict(control=values,frozen={'metric':np.zeros(2)},flow_film=values))


def test_whole_path_weights_equal_explicit_resampling():
    rng=np.random.default_rng(31)
    s=rng.normal(size=(3,201,25,3)); s/=np.linalg.norm(s,axis=-1,keepdims=True); s[:,0]=s[0,0]
    from tests.experiments.path_observables import path_observables
    w=np.array([[2/3,0,1/3],[0,1,0]])
    g=weighted_observables(s,w); r=weighted_observables(s,np.full((2,3),1/3))
    got=batched_errors(g,r)
    ref=path_observables(s)
    for i,ix in enumerate(([0,0,2],[1,1,1])):
        expected=moment_errors(path_observables(s[ix]),ref)
        for k in expected: np.testing.assert_allclose(got[k][i],expected[k],atol=1e-14)
