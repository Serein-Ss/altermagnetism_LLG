import numpy as np
from tests.experiments.local_state_coverage import descriptors


def test_rotation_and_director_sign_invariance():
    rng=np.random.default_rng(83)
    x=rng.normal(size=(3,25,3)); x/=np.linalg.norm(x,axis=-1,keepdims=True)
    q,_=np.linalg.qr(rng.normal(size=(3,3)))
    q[:,0]*=np.linalg.det(q)
    axis=np.array([0.,0.,1.])
    d=descriptors(x,axis)
    assert d.shape==(3,25,10)
    np.testing.assert_allclose(d,descriptors(x@q.T,axis@q.T),atol=1e-14)
    np.testing.assert_array_equal(d,descriptors(x,-axis))


def test_open_boundary_no_wrap():
    x=np.zeros((1,5,3)); x[:,:,2]=1
    a=descriptors(x)
    x[:,-1]=[1,0,0]
    b=descriptors(x)
    np.testing.assert_array_equal(a[:,:3],b[:,:3])
    assert a[0,0,0]==0 and a[0,-1,1]==0
