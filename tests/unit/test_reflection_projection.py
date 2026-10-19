import torch
from tests.experiments.director_transition import DirectorTransition
from tests.experiments.reflection_projection import ReflectionProjection
from tests.experiments.projected_midpoint_sampler import midpoint_step


def test_projection_symmetries_and_tangency():
    torch.manual_seed(79)
    base=DirectorTransition(.12,.01,16).double(); model=ReflectionProjection(base)
    x=torch.nn.functional.normalize(torch.randn(3,25,3,dtype=torch.float64),dim=-1)
    initial=torch.nn.functional.normalize(torch.randn_like(x),dim=-1)
    tau=x.new_full((3,),.4); theta=x.new_full((3,),.11); lag=x.new_full((3,),.01)
    axis=x.new_tensor([0.,0.,1.]).expand(3,-1)
    args=(tau,initial,theta,lag,axis)
    v=model(x,*args)
    torch.testing.assert_close(v,model(x.flip(1),tau,initial.flip(1),theta,lag,axis).flip(1),atol=1e-12,rtol=0)
    torch.testing.assert_close(v,model(x,tau,initial,theta,lag,-axis),atol=1e-12,rtol=0)
    torch.testing.assert_close((v*x).sum(-1),torch.zeros_like(tau[:,None].expand(-1,25)),atol=1e-12,rtol=0)
    q,_=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype)); q[:,0]*=torch.linalg.det(q)
    torch.testing.assert_close(v@q.T,model(x@q.T,tau,initial@q.T,theta,lag,axis@q.T),atol=1e-11,rtol=0)
    assert list(model.parameters())==list(base.parameters())


def test_projected_integrator_reflects_explicit_state():
    torch.manual_seed(9)
    model=ReflectionProjection(DirectorTransition(.12,.01,16)).double()
    initial=torch.nn.functional.normalize(torch.randn(2,25,3,dtype=torch.float64),dim=-1)
    x=torch.nn.functional.normalize(initial+.01*torch.randn_like(initial),dim=-1)
    theta=x.new_full((2,),.11); lag=x.new_full((2,),.01); axis=x.new_tensor([0.,0.,1.]).expand(2,-1)
    field=lambda y,t:model(y,theta.new_full((2,),t),initial,theta,lag,axis)
    reverse=lambda y,t:model(y,theta.new_full((2,),t),initial.flip(1),theta,lag,axis)
    a=midpoint_step(field,x,.2,.01); b=midpoint_step(reverse,x.flip(1),.2,.01).flip(1)
    torch.testing.assert_close(a,b,atol=1e-12,rtol=0)
    torch.testing.assert_close(torch.linalg.norm(a,dim=-1),torch.ones(2,25,dtype=a.dtype),atol=1e-12,rtol=0)
