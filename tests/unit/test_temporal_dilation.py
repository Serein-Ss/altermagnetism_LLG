import torch
import pytest
from tests.experiments.velocity_controls import CartesianPathNet


def test_equal_initial_parameters_and_expanded_influence():
    torch.manual_seed(13)
    local=CartesianPathNet(width=8)
    torch.manual_seed(13)
    wide=CartesianPathNet(width=8,temporal_dilations=[1,4,16,64])
    assert all(torch.equal(v,wide.state_dict()[k]) for k,v in local.state_dict().items())
    x=torch.randn(1,201,1,5,1,3);x/=x.norm(dim=-1,keepdim=True);x.requires_grad_()
    args=(torch.tensor([.4]),x[:,0].detach(),torch.tensor([[.13,.1]]),torch.linspace(0,1,201)[None],None,None)
    gradients=[]
    for net in (local,wide):
        prediction=net(x,*args)
        assert prediction.shape==x.shape and torch.isfinite(prediction).all()
        assert (prediction*x).sum(-1).abs().max()<1e-5
        assert torch.equal(prediction[:,0],torch.zeros_like(prediction[:,0]))
        gradients.append(torch.autograd.grad(prediction[0,100,0,2,0,0],x)[0])
    assert gradients[0][:,36].abs().max()==0
    assert gradients[1][:,36].abs().max()>1e-8


def test_explicit_local_dilation_preserves_default_output():
    torch.manual_seed(5);a=CartesianPathNet(width=8)
    torch.manual_seed(5);b=CartesianPathNet(width=8,temporal_dilations=[1,1,1,1])
    x=torch.randn(1,9,1,5,1,3);x/=x.norm(dim=-1,keepdim=True)
    args=(x,torch.tensor([.3]),x[:,0],torch.tensor([[.11,.1]]),torch.linspace(0,1,9)[None],None,None)
    assert torch.equal(a(*args),b(*args))
    with pytest.raises(ValueError):CartesianPathNet(temporal_dilations=[1,2])
