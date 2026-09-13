import itertools
import math
import numpy as np
import torch
from scripts.literature.bauer_2011.integrator import BauerWeakRK


class ZeroField:
    def field(self,s): return torch.zeros_like(s)


class ConstantField:
    def field(self,s): return torch.zeros_like(s)+s.new_tensor((0.,0.,1.))


def test_weak_rk_zero_noise_reduces_to_fourth_order_rk():
    errors=[]
    solver=BauerWeakRK(ConstantField(),alpha=0.,theta=0.,project=False)
    for dt in (.1,.05):
        s=torch.tensor([[[1.,0.,0.]]],dtype=torch.float64)
        for _ in range(round(1/dt)): s,_=solver.step(s,dt,torch.zeros_like(s))
        exact=s.new_tensor((math.cos(1),math.sin(1),0.))
        errors.append(float((s-exact).norm()))
    assert 15<errors[0]/errors[1]<17


def test_weak_rk_exact_enumeration_matches_rotational_diffusion_moments():
    xi_values=(-math.sqrt(3),0.,math.sqrt(3)); xi_prob=(1/6,2/3,1/6)
    draws=[]; weights=[]
    for ids in itertools.product(range(3),repeat=3):
        for zeta in itertools.product((-1.,1.),repeat=3):
            draws.append(np.stack(([xi_values[i] for i in ids],zeta),-1))
            weights.append(np.prod([xi_prob[i] for i in ids])/8)
    draws=torch.tensor(np.array(draws),dtype=torch.float64)[:,None]
    weights=torch.tensor(weights,dtype=torch.float64)
    s=torch.zeros((len(weights),1,3),dtype=torch.float64); s[...,2]=1.
    solver=BauerWeakRK(ZeroField(),alpha=1.,theta=.5,project=False)
    errors=[]
    for dt in (.1,.05):
        after,_=solver.step(s,dt,draws)
        mean=float((weights*after[:,0,2]).sum())
        second=float((weights*after[:,0,2]**2).sum())
        errors.append([abs(mean-math.exp(-dt)),abs(second-(1+2*math.exp(-3*dt))/3)])
    ratios=np.array(errors[0])/np.array(errors[1])
    assert np.all((ratios>6)&(ratios<10))


def test_weak_draws_and_projection_are_explicit():
    solver=BauerWeakRK(ZeroField(),alpha=.1,theta=.11,project=True)
    s=torch.zeros((128,4,3),dtype=torch.float64); s[...,2]=1
    draws=solver.noise_increment(s,.01,torch.Generator().manual_seed(17))
    assert draws.shape==(*s.shape,2)
    assert set(draws[...,1].unique().tolist())=={-1.,1.}
    after,errors=solver.step(s,.01,draws)
    assert (after.norm(dim=-1)-1).abs().max()<1e-14
