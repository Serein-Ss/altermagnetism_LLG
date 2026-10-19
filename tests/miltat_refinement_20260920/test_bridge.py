import torch
from tests.miltat_refinement_20260920.run import bridge


def test_sum_and_independent_halves():
    rng=torch.Generator().manual_seed(91);dt=.3
    dw=torch.randn(100000,dtype=torch.float64,generator=rng)*dt**.5
    eta=torch.randn(100000,dtype=torch.float64,generator=rng)
    a,b=bridge(dw,eta,dt)
    torch.testing.assert_close(a+b,dw,atol=1e-15,rtol=1e-15)
    assert abs(float(a.var())-dt/2)<.003
    assert abs(float(b.var())-dt/2)<.003
    assert abs(float(((a-a.mean())*(b-b.mean())).mean()))<.003


def test_bridge_does_not_change_parent_generator():
    g=torch.Generator().manual_seed(12);reference=torch.Generator().manual_seed(12)
    extra=torch.Generator().manual_seed(13)
    for _ in range(4):
        dw=torch.randn(2,3,generator=g)
        bridge(dw,torch.randn(2,3,generator=extra),.2)
        assert torch.equal(dw,torch.randn(2,3,generator=reference))
