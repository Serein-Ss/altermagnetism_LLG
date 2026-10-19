import torch
from tests.experiments.velocity_controls import CartesianPathNet


def test_zero_context_matches_baseline_and_receives_gradients():
    torch.manual_seed(11);base=CartesianPathNet(width=8)
    torch.manual_seed(11);net=CartesianPathNet(width=8,global_context=True)
    x=torch.randn(2,21,1,5,1,3);x/=x.norm(dim=-1,keepdim=True)
    args=(torch.tensor([.3,.6]),x[:,0],torch.tensor([[.11,.1],[.13,.1]]),torch.linspace(0,1,21)[None].expand(2,-1),None,None)
    out=net(x,*args)
    assert torch.equal(out,base(x,*args))
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert all(layer.weight.grad.abs().max()>0 for layer in net.context)


def test_context_is_global_within_path_not_across_batch():
    torch.manual_seed(42);net=CartesianPathNet(width=8,global_context=True)
    with torch.no_grad():
        for layer in net.context:layer.weight.normal_(0,.1)
    x=torch.randn(2,101,1,5,1,3);x/=x.norm(dim=-1,keepdim=True);x.requires_grad_()
    out=net(x,torch.tensor([.3,.6]),x[:,0].detach(),torch.tensor([[.11,.1],[.13,.1]]),
            torch.linspace(0,1,101)[None].expand(2,-1),None,None)
    grad=torch.autograd.grad(out[0,50,0,2,0,0],x)[0]
    assert grad[0,10].abs().max()>1e-8
    assert grad[1].abs().max()==0
    assert (out*x).sum(-1).abs().max()<1e-5
    assert torch.equal(out[:,0],torch.zeros_like(out[:,0]))
