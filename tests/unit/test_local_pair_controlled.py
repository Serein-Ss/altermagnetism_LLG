import pytest
import torch
from tests.experiments.velocity_controls import CartesianPathNet
from tests.experiments.local_pair_controlled import warm_start
from scripts.model.gate_f import matching_loss


def test_warm_start_loss_rng_pairing_and_reject_missing_backbone():
    torch.manual_seed(52)
    weights=CartesianPathNet(width=8,blocks=2).state_dict()
    nets=[warm_start(a,weights,width=8,blocks=2) for a in ('control','local_pair')]
    y=torch.randn(2,11,1,5,1,3); y=y/y.norm(dim=-1,keepdim=True)
    c=torch.tensor([[.11,.1],[.13,.1]]); t=torch.linspace(0,.05,11)[None].expand(2,-1)
    gens=[torch.Generator().manual_seed(12) for _ in nets]
    losses=[matching_loss(net,y,c,t,None,None,g)[0] for net,g in zip(nets,gens)]
    torch.testing.assert_close(losses[0],losses[1],rtol=0,atol=0)
    assert torch.equal(gens[0].get_state(),gens[1].get_state())
    for net,loss in zip(nets,losses):
        opt=torch.optim.Adam(net.parameters(),lr=.0001)
        assert len(opt.state)==0
        loss.backward(); opt.step()
        assert all(torch.isfinite(p).all() for p in net.parameters())
    bad=dict(weights); bad.pop('input.weight')
    with pytest.raises(ValueError): warm_start('local_pair',bad,width=8,blocks=2)
