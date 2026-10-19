import torch
from tests.experiments.velocity_controls import CartesianPathNet, VelocityTable
from scripts.model.sphere import tangent_project


def test_table_can_represent_target_exactly():
    x=torch.randn(2,5,1,4,1,3);x/=x.norm(dim=-1,keepdim=True)
    target=tangent_project(x,torch.randn_like(x));target[:,0]=0
    net=VelocityTable(x.shape)
    with torch.no_grad():net.value.copy_(target)
    assert torch.allclose(net(x),target,atol=1e-6)


def test_cartesian_forward_gradient_and_initial():
    x=torch.randn(2,5,1,4,1,3);x/=x.norm(dim=-1,keepdim=True)
    net=CartesianPathNet(width=8,blocks=1)
    out=net(x,torch.tensor([.3,.7]),x[:,0],torch.tensor([[.11,.1],[.13,.1]]),
        torch.linspace(0,1,5)[None].expand(2,-1),None,None)
    assert out.shape==x.shape and torch.equal(out[:,0],torch.zeros_like(x[:,0]))
    assert (out*x).sum(-1).abs().max()<1e-5
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_cartesian_random_source_loss_and_sampler():
    from scripts.model.gate_f import matching_loss, sample
    gen=torch.Generator().manual_seed(432)
    y=torch.randn(2,7,1,5,1,3,generator=gen);y/=y.norm(dim=-1,keepdim=True)
    net=CartesianPathNet(width=8,blocks=1)
    c=torch.tensor([[.11,.1],[.13,.1]])
    t=torch.linspace(0,1,7)[None].expand(2,-1)
    loss,_=matching_loss(net,y,c,t,None,None,gen)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    pred=sample(net,y[:,0],c,t,None,None,gen,steps=16)
    assert torch.isfinite(pred).all()
    assert torch.equal(pred[:,0],y[:,0])
    assert torch.allclose(pred.norm(dim=-1),torch.ones_like(pred[...,0]),atol=1e-6)
