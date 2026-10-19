import numpy as np
import pytest
from tests.experiments.fine_transition_pilot import select_anchors
from tests.experiments.independent_reference import generate


def test_select_distinct_train_families_and_explicit_normalization():
    s=np.zeros((4,3,2,3)); s[...,2]=1+1e-8; s[2:,...,2]*=-1
    c=np.array([[.11,.1]]*4)
    cfg=dict(theta=[.11],families_per_theta=2,start_frames=[0,2])
    original,states,rows=select_anchors(s,c,cfg)
    assert [r['parent'] for r in rows]==[0,0,2,2]
    assert np.max(np.abs(original-states))>0
    np.testing.assert_allclose(np.linalg.norm(states,axis=-1),1,atol=1e-15)
    with pytest.raises(ValueError): select_anchors(s[:2],c[:2],cfg)


@pytest.mark.parametrize('method,dt',[('mt',.01),('mt',.005),('mt',.0025),('heun',.0025)])
def test_fine_generator_grid_and_reproducibility(method,dt):
    s=np.zeros((2,25,3)); s[...,2]=1
    args=dict(dt=dt,duration=.08,save_dt=dt,anisotropy=.1,damping=.1,theta=.13,method=method)
    a,_=generate(s,np.array([929900001,929900002]),**args)
    b,_=generate(s,np.array([929900001,929900002]),**args)
    assert a.shape==(2,round(.08/dt)+1,25,3)
    np.testing.assert_array_equal(a,b)
    assert not np.array_equal(a[0,1:],a[1,1:])
