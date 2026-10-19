import math
import torch
from tests.experiments.projected_midpoint_sampler import midpoint_step, sample
from tests.experiments.local_equivariant_transition import sample as euler_sample
from tests.experiments.director_transition import DirectorTransition


def test_second_order_for_time_dependent_rotation():
    initial = torch.tensor([[[1.,0.,0.]]], dtype=torch.float64)
    axis = torch.tensor([0.,0.,1.], dtype=torch.float64)
    def field(x, t):
        return (1+t)*torch.linalg.cross(axis.expand_as(x), x)
    exact = torch.tensor([[[math.cos(1.5), math.sin(1.5),0.]]], dtype=torch.float64)
    errors = []
    for steps in [16,32,64]:
        x = initial.clone()
        for k in range(steps):
            x = midpoint_step(field, x, k/steps, 1/steps)
        torch.testing.assert_close(x.norm(dim=-1), torch.ones_like(x[...,0]), atol=1e-14, rtol=0)
        errors.append(float((x-exact).norm()))
    assert 3.5 < errors[0]/errors[1] < 4.5
    assert 3.5 < errors[1]/errors[2] < 4.5


def test_same_latent_consumption_and_rotation():
    torch.manual_seed(1)
    model = DirectorTransition(.12,.01,8).double()
    initial = torch.nn.functional.normalize(torch.randn(2,5,3,dtype=torch.float64),dim=-1)
    theta = initial.new_full((2,),.11); lag = theta*0+.01
    axis = initial.new_tensor([0.,0.,1.]).expand(2,-1)
    g1 = torch.Generator().manual_seed(2); g2 = torch.Generator().manual_seed(2)
    x = sample(model,initial,theta,lag,axis,g1,.02,4)
    euler_sample(model,initial,theta,lag,axis,g2,.02,8)
    assert torch.equal(g1.get_state(),g2.get_state())
    torch.testing.assert_close(x.norm(dim=-1),torch.ones_like(x[...,0]),atol=1e-14,rtol=0)
    q,_ = torch.linalg.qr(torch.randn(3,3,dtype=torch.float64)); q[:,0] *= torch.linalg.det(q)
    def f(x,t):
        return model(x,theta*0+t,initial,theta,lag,axis)
    def fr(x,t):
        return model(x,theta*0+t,initial@q,theta,lag,axis@q)
    torch.testing.assert_close(midpoint_step(fr,initial@q,.2,.01),midpoint_step(f,initial,.2,.01)@q,atol=1e-12,rtol=1e-12)
