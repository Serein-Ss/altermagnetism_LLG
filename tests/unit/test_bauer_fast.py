import numpy as np
import pytest
import torch
from scripts.core.bauer_fast import mt_step, heun_step, ensemble
from scripts.literature.bauer_2011.integrator import BauerWeakRK
from scripts.literature.bauer_2011.model import OpenChain
from scripts.core.reduced_llg import ReducedLLG


@pytest.mark.parametrize('project',[False,True])
@pytest.mark.parametrize('theta',[0.,.11])
def test_same_draws_match_independent_torch_implementation(project,theta):
    rng=np.random.default_rng(73)
    s=rng.normal(size=(13,3));s/=np.linalg.norm(s,axis=-1,keepdims=True)
    model=OpenChain(dict(equation_convention='bauer_ll',exchange=1.,anisotropy=.1))
    ref=BauerWeakRK(model,alpha=.1,theta=theta,project=project)
    a=s.copy();b=torch.from_numpy(s[None].copy())
    for _ in range(20):
        xi=rng.choice([-np.sqrt(3),0.,np.sqrt(3)],size=s.shape,p=[1/6,2/3,1/6])
        zeta=rng.choice([-1.,1.],size=s.shape)
        if theta==0: xi*=0;zeta*=0
        a,err=mt_step(a,xi,zeta,.01,.1,.1,theta,project)
        b,errors=ref.step(b,.01,torch.from_numpy(np.stack((xi,zeta),-1)[None]))
        np.testing.assert_allclose(a,b[0].numpy(),atol=2e-13,rtol=2e-13)
        assert abs(err-errors[0].item())<2e-13


def test_heun_same_wiener_matches_reference():
    rng=np.random.default_rng(7);s=rng.normal(size=(7,3));s/=np.linalg.norm(s,axis=-1,keepdims=True)
    dw=rng.normal(size=s.shape)*.1
    model=OpenChain(dict(equation_convention='bauer_ll',exchange=1.,anisotropy=.1))
    ref=ReducedLLG(model,alpha=.1,theta=.11,equation_convention='bauer_ll')
    a,_=heun_step(s,dw,.01,.1,.1,.11)
    b,_=ref.step(torch.from_numpy(s[None]),.01,torch.from_numpy(dw[None]),method='heun')
    np.testing.assert_allclose(a,b[0].numpy(),atol=2e-13)


def test_path_identity_independent_of_batch_order():
    initial=np.zeros((3,5,3));initial[...,2]=1
    seeds=np.array([17,81,19])
    a,e=ensemble(initial,seeds,.01,20,5,.1,.1,.11)
    b,f=ensemble(initial[::-1].copy(),seeds[::-1].copy(),.01,20,5,.1,.1,.11)
    np.testing.assert_array_equal(a,b[::-1])
    np.testing.assert_allclose(np.linalg.norm(a,axis=-1),1,atol=1e-14)
    assert not np.array_equal(a[0],a[1])
