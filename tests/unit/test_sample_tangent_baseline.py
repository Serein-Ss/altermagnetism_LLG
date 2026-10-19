import numpy as np
from tests.experiments.sample_tangent_baseline import basis,exponential,logarithm,fit,sample


def test_fit_recovers_known_synthetic_mean_and_zero_noise():
    rng=np.random.default_rng(70)
    x=rng.normal(size=(32,25,3));x/=np.linalg.norm(x,axis=-1,keepdims=True)
    axis=np.array([0.,0.,1.]); coefficients=np.array([.001,.002,-.003,.004])
    tangent=np.einsum('nlkc,k->nlc',basis(x,axis),coefficients)
    target=exponential(x,tangent)
    np.testing.assert_allclose(logarithm(x,target),tangent,atol=1e-14)
    recovered,sigma,diagnostics=fit(x,target,axis)
    np.testing.assert_allclose(recovered,coefficients,atol=1e-14)
    assert sigma<1e-14 and diagnostics['rank']==4
    np.testing.assert_allclose(exponential(x,np.zeros_like(x)),x,atol=1e-14)


def test_symmetries_unit_constraint_and_random_diversity():
    rng=np.random.default_rng(71)
    x=rng.normal(size=(4,25,3));x/=np.linalg.norm(x,axis=-1,keepdims=True)
    axis=np.array([0.,0.,1.]); q,_=np.linalg.qr(rng.normal(size=(3,3)))
    if np.linalg.det(q)<0:q[:,0]*=-1
    np.testing.assert_allclose(basis(x@q,axis@q),basis(x,axis)@q,atol=1e-14)
    np.testing.assert_allclose(basis(x[:,::-1],axis),basis(x,axis)[:,::-1],atol=1e-14)
    np.testing.assert_allclose(basis(x,-axis),basis(x,axis),atol=1e-14)
    p=sample(x,axis,np.zeros(4),.01,rng)
    np.testing.assert_allclose(np.linalg.norm(p,axis=-1),1,atol=1e-14)
    assert np.max(abs(p-x))>0


def test_path_rollout_replay_and_initial_condition():
    from tests.experiments.sample_tangent_path_comparison import rollout
    x=np.zeros((25,3));x[:,2]=1
    parameter=dict(coefficients=[.001,-.01,.0002,-.002],sigma=.015)
    p=rollout(x,parameter,np.array([0.,0.,1.]),8,10,12)
    np.testing.assert_array_equal(p,rollout(x,parameter,np.array([0.,0.,1.]),8,10,12))
    assert p.shape==(8,11,25,3)
    np.testing.assert_array_equal(p[:,0],np.repeat(x[None],8,axis=0))
    np.testing.assert_allclose(np.linalg.norm(p,axis=-1),1,atol=1e-12)
    assert np.var(p[:,-1,0,0])>0
