import numpy as np
import torch
import pytest
from tests.experiments.learned_tangent_source import LearnedTangentSource
from tests.experiments.sample_tangent_baseline import basis,project,exponential
from tests.experiments.director_transition import DirectorTransition
from tests.experiments.reflection_augmentation import augmented_train_step
import copy


def source():
    return LearnedTangentSource(dict(lag=.01,rows=[dict(theta=.11,coefficients=[.001,-.01,.0002,-.002],sigma=.015)]))


def test_matches_numpy_and_zero_flow_keeps_source():
    rng=np.random.default_rng(30)
    x=rng.normal(size=(4,25,3));x/=np.linalg.norm(x,axis=-1,keepdims=True)
    noise=rng.normal(size=x.shape);axis=np.array([0.,0.,1.])
    mean=np.einsum('nlkc,k->nlc',basis(x,axis),np.array([.001,-.01,.0002,-.002]))
    expected=exponential(x,mean+.015*project(x,noise))
    a=torch.tensor(x); t=torch.full((4,),.11,dtype=a.dtype); axes=torch.tensor(axis).expand(4,-1)
    prior=source();actual=prior.from_noise(a,t,axes,torch.tensor(noise))
    np.testing.assert_allclose(actual.numpy(),expected,atol=1e-12)
    class Zero:
        def __call__(self,x,*args):return torch.zeros_like(x)
    g1=torch.Generator().manual_seed(8);g2=torch.Generator().manual_seed(8)
    expected=prior.draw(a,t,axes,g1)
    p=prior.sample(Zero(),a,t,torch.full_like(t,.01),axes,g2,99.,32)
    torch.testing.assert_close(p,expected,atol=1e-12,rtol=0)
    assert torch.equal(g1.get_state(),g2.get_state())
    with pytest.raises(ValueError):prior.draw(a,t+.01,axes,g1)


def test_training_preserves_control_random_draw_counts():
    torch.manual_seed(30)
    a=torch.nn.functional.normalize(torch.randn(4,25,3),dim=-1)
    b=torch.nn.functional.normalize(a+.01*torch.randn_like(a),dim=-1)
    model=DirectorTransition(.12,.01,8);control=copy.deepcopy(model)
    t=torch.full((4,),.11);lag=torch.full((4,),.01);axis=torch.tensor([0.,0.,1.]).expand(4,-1)
    generators=[torch.Generator().manual_seed(8) for _ in range(2)]
    aug=[torch.Generator().manual_seed(9) for _ in range(2)]
    result=source().train_step(model,torch.optim.Adam(model.parameters()),a,b,t,lag,axis,generators[0],.016,aug[0])
    augmented_train_step(control,torch.optim.Adam(control.parameters()),a,b,t,lag,axis,generators[1],.016,aug[1])
    assert np.isfinite(result[:2]).all()
    assert torch.equal(generators[0].get_state(),generators[1].get_state())
    assert torch.equal(aug[0].get_state(),aug[1].get_state())


def test_analysis_verifier_labels_match():
    from tests.experiments.learned_source_analysis import compare
    from tests.experiments.verify_learned_source_analysis import verify_paired_draws
    rng=np.random.default_rng(98)
    data=[rng.normal(size=(8,3,5,3)) for _ in range(3)]
    arrays,metrics=compare(data,rng,20)
    assert set(metrics)=={'control','learned_source','paired_change'}
    for key in metrics['paired_change']:verify_paired_draws(arrays,key)
    arrays['paired_change_mean_rmse']+=1
    with pytest.raises(AssertionError):verify_paired_draws(arrays,'mean_rmse')
