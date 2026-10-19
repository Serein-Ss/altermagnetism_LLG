import torch
from scripts.model.gate_f import matching_loss,reference_path,path_interpolate
from tests.experiments.velocity_controls import CartesianPathNet
from tests.experiments.weighted_flow_loss import weighted_flow_loss


def test_uniform_matches_original_and_weighted_formula():
    torch.manual_seed(51)
    y=torch.randn(4,9,1,5,1,3);y/=y.norm(dim=-1,keepdim=True)
    net=CartesianPathNet(width=8,blocks=1)
    c=torch.tensor([[.13,.1]]).expand(4,-1);t=torch.linspace(0,1,9)[None].expand(4,-1)
    gen=lambda:torch.Generator().manual_seed(33)
    a=gen();b=gen()
    original,_=matching_loss(net,y,c,t,None,None,a)
    uniform,_=weighted_flow_loss(net,y,c,t,None,None,b,power=0)
    assert torch.allclose(original,uniform,atol=1e-7)
    assert torch.equal(a.get_state(),b.get_state())
    g=gen();tau=torch.rand(4,generator=g);z=reference_path(y[:,0],9,g)
    x,target,_=path_interpolate(z,y,tau,g)
    mse=(net(x,tau,y[:,0],c,t,None,None)-target)[:,1:].square().sum(-1).flatten(1).mean(1)
    weighted,_=weighted_flow_loss(net,y,c,t,None,None,gen())
    assert torch.allclose(weighted,(3*(1-tau).square()*mse).mean())
    weighted.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
