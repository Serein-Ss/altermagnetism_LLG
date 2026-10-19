import torch
from tests.experiments.local_pair_features import open_pair_features, LocalPairPathNet
from tests.experiments.velocity_controls import CartesianPathNet


def test_open_boundary_and_rotation_invariant_features():
    s=torch.tensor([[[[1.,1.,-1.]],[[0.,0.,0.]],[[0.,0.,0.]]]],dtype=torch.float64)
    f=open_pair_features(s)
    torch.testing.assert_close(f,torch.tensor([[[[0.,1.,-1.]],[[1.,-1.,0.]],[[0.,1.,1.]],[[1.,1.,0.]]]],dtype=s.dtype))
    q,_=torch.linalg.qr(torch.randn(3,3,dtype=s.dtype))
    rotated=torch.einsum('ij,bjfl->bifl',q,s)
    torch.testing.assert_close(open_pair_features(rotated),f)


def test_zero_branch_exact_warm_start_and_nonzero_gradient():
    torch.manual_seed(71); base=CartesianPathNet(width=4,blocks=2)
    candidate=LocalPairPathNet(width=4,blocks=2)
    missing,unexpected=candidate.load_state_dict(base.state_dict(),strict=False)
    assert missing==['input.pair_features.weight'] and not unexpected
    x=torch.randn(2,11,1,25,1,3); x=x/x.norm(dim=-1,keepdim=True)
    tau=torch.tensor([.2,.8]); c=torch.tensor([[.11,.1],[.13,.1]])
    t=torch.linspace(0,.05,11)[None].expand(2,-1)
    args=(x,tau,x[:,0],c,t,None,None)
    expected=base(*args); actual=candidate(*args)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert torch.count_nonzero(actual[:,0])==0
    assert (actual*x).sum(-1).abs().max()<1e-6
    actual.square().sum().backward()
    grad=candidate.input.pair_features.weight.grad
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad)>0
