import copy
import torch
from tests.experiments.boundary_weighted_loss import loss,site_weights
from tests.experiments.local_equivariant_transition import loss as original_loss
from tests.experiments.director_transition import DirectorTransition


def test_weights_normalized_symmetric_and_balanced():
    w=site_weights(25,dtype=torch.float64,device='cpu')
    torch.testing.assert_close(w.mean(),torch.tensor(1.,dtype=w.dtype))
    torch.testing.assert_close(w[:3].sum()+w[-3:].sum(),w[3:-3].sum())
    torch.testing.assert_close(w,w.flip(0))
    assert (w>0).all()


def test_uniform_weights_match_original_loss_gradients_and_rng():
    torch.manual_seed(68)
    a=DirectorTransition(.12,.01,16); b=copy.deepcopy(a)
    initial=torch.nn.functional.normalize(torch.randn(4,25,3),dim=-1)
    target=torch.nn.functional.normalize(initial+.02*torch.randn_like(initial),dim=-1)
    theta=torch.full((4,),.11); lag=torch.full((4,),.01); axis=torch.tensor([0.,0.,1.]).expand(4,-1)
    ga=torch.Generator().manual_seed(551); gb=torch.Generator().manual_seed(551)
    va,_=original_loss(a,initial,target,theta,lag,axis,ga,.016)
    vb,_=loss(b,initial,target,theta,lag,axis,gb,.016,weights=torch.ones(25))
    va.backward(); vb.backward()
    torch.testing.assert_close(va,vb,atol=0,rtol=0)
    torch.testing.assert_close(ga.get_state(),gb.get_state(),atol=0,rtol=0)
    for pa,pb in zip(a.parameters(),b.parameters()): torch.testing.assert_close(pa.grad,pb.grad,atol=0,rtol=0)
