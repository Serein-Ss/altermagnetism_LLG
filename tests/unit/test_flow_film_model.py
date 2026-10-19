import torch
from tests.experiments.velocity_controls import CartesianPathNet
from tests.experiments.flow_film_model import FlowFiLMPathNet


def test_zero_film_exact_control_replay_and_trainable_tangent_velocity():
    torch.manual_seed(42)
    control=CartesianPathNet(width=8,blocks=2)
    candidate=FlowFiLMPathNet(width=8,blocks=2)
    missing,unexpected=candidate.load_state_dict(control.state_dict(),strict=False)
    assert missing and all(k.startswith('flow_film.') for k in missing) and not unexpected
    x=torch.randn(3,11,1,7,1,3); x=x/x.norm(dim=-1,keepdim=True)
    tau=torch.tensor([0.,.95,.99]); initial=x[:,0]
    c=torch.tensor([[.11,.1],[.13,.1],[.11,.1]])
    t=torch.linspace(0,.05,11)[None].expand(3,-1)
    args=(x,tau,initial,c,t,None,None)
    pred=candidate(*args)
    torch.testing.assert_close(pred,control(*args),rtol=0,atol=0)
    assert torch.count_nonzero(pred[:,0])==0
    torch.testing.assert_close((pred*x).sum(-1),torch.zeros_like(pred[...,0]),rtol=0,atol=2e-6)
    pred.square().mean().backward()
    assert all(layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
               and layer.weight.grad.abs().sum()>0 for layer in candidate.flow_film)
    with torch.no_grad():
        candidate.flow_film[0].weight.add_(.01)
    assert not torch.equal(candidate(*args),control(*args))
