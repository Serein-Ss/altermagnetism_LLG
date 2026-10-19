import pytest
import torch
from tests.experiments.velocity_controls import CartesianPathNet
from tests.experiments.reversal_symmetric_model import ReversalSymmetricPathNet


@pytest.mark.parametrize('length',[16,25,32])
def test_reversal_tangency_anchor_and_gradients(length):
    torch.manual_seed(61)
    net=ReversalSymmetricPathNet(CartesianPathNet(width=4,blocks=2)).double()
    x=torch.randn(2,11,1,length,1,3,dtype=torch.float64)
    x=x/x.norm(dim=-1,keepdim=True)
    tau=torch.tensor([.2,.8],dtype=x.dtype)
    c=torch.tensor([[.11,.1],[.13,.1]],dtype=x.dtype)
    t=torch.linspace(0,.05,11,dtype=x.dtype)[None].expand(2,-1)
    v=net(x,tau,x[:,0],c,t,None,None)
    flipped=net(x.flip(-3),tau,x[:,0].flip(-3),c,t,None,None)
    torch.testing.assert_close(flipped,v.flip(-3),rtol=0,atol=1e-14)
    assert (v*x).sum(-1).abs().max()<1e-14
    assert torch.count_nonzero(v[:,0])==0
    v.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert sum(p.numel() for p in net.parameters())==sum(p.numel() for p in net.backbone.parameters())
    with pytest.raises(ValueError): net(x,tau,x[:,0],c,t,object(),None)
