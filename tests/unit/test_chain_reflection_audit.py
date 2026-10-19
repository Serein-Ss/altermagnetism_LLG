import numpy as np
import torch
from scripts.core.bauer_fast import drift,heun_step,mt_step
from tests.experiments.chain_reflection_audit import reflected_field


def test_reference_chain_reversal_with_reversed_noise():
    rng=np.random.default_rng(67)
    s=rng.normal(size=(25,3)); s/=np.linalg.norm(s,axis=-1,keepdims=True)
    xi=rng.normal(size=s.shape); zeta=rng.normal(size=s.shape)
    rev=lambda a: np.ascontiguousarray(a[::-1])
    np.testing.assert_allclose(drift(s,.1,.1),rev(drift(rev(s),.1,.1)),atol=1e-14)
    a,_=heun_step(s,xi*.01,.00125,.1,.1,.11)
    b,_=heun_step(rev(s),rev(xi)*.01,.00125,.1,.1,.11)
    np.testing.assert_allclose(a,rev(b),atol=1e-14)
    a,_=mt_step(s,xi,zeta,.00125,.1,.1,.11)
    b,_=mt_step(rev(s),rev(xi),rev(zeta),.00125,.1,.1,.11)
    np.testing.assert_allclose(a,rev(b),atol=1e-14)


def test_reflection_maps_back_site_order_without_spin_inversion():
    class Identity:
        def __call__(self,x,*args): return x
    x=torch.randn(3,25,3)
    torch.testing.assert_close(reflected_field(Identity(),x,None,x,None,None,None),x)
