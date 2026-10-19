import math
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.cross_dimension_20260924.oracle import conditional as uniform_conditional
from .oracle import log_source,conditional


def test_density_normalization_and_radial_quadrature():
    torch.set_num_threads(1)
    c=torch.linspace(-1,1,20001,dtype=torch.float64)
    u=torch.stack(((1-c*c).sqrt(),torch.zeros_like(c),c),-1);a=u.new_tensor([0,0,1])
    assert abs(float(2*math.pi*torch.trapezoid(log_source(u,a).exp(),c))-1)<1e-7
    r=torch.linspace(0,12,20001,dtype=c.dtype)
    for b in (-2.,-1.,0.,1.,2.):
        cc=b/2;v=u.new_tensor([(1-cc*cc)**.5,0,cc])
        radial=torch.trapezoid(r*r*torch.exp(-.5*r*r+b*r-2),r)/(2*math.pi)**1.5
        assert torch.allclose(log_source(v,a).exp(),radial,atol=1e-9,rtol=1e-8)


def fixture():
    torch.manual_seed(9291)
    targets=F.normalize(torch.randn(3,4,2,3,dtype=torch.float64),dim=-1);targets[:,0]=targets[:1,0]
    initial=targets[:1,0].expand(6,-1,-1)
    z=F.normalize(torch.randn(6,4,2,3,dtype=torch.float64)+2*initial[:,None],dim=-1);z[:,0]=initial
    return targets,initial,z


def test_tau_zero_uniform_posterior_and_uniform_source_limit():
    y,a,z=fixture();t=z.new_zeros(len(z))
    v,w,bad=conditional(z,t,y,a)
    assert not bad.any() and torch.allclose(w,torch.full_like(w,1/3),atol=1e-10)
    t=t+.2;x,_,_=path_interpolate(z,y[torch.arange(len(z))%3],t,torch.Generator().manual_seed(1))
    v,w,bad=conditional(x,t,y,a,0.)
    vo,wo,bo=uniform_conditional(x,t,y[None].expand(len(x),-1,-1,-1,-1))
    assert torch.equal(bad,bo) and torch.allclose(w,wo,atol=1e-7)
    assert torch.allclose(v,vo,atol=1e-6)


def test_single_target_velocity_rotation_and_tangency():
    y,a,z=fixture();y=y[:1];t=z.new_full((len(z),),.5)
    x,truth,_=path_interpolate(z,y.expand(len(z),-1,-1,-1),t,torch.Generator().manual_seed(1))
    v,w,bad=conditional(x,t,y,a)
    assert not bad.any() and torch.allclose(v,truth,atol=1e-10)
    assert (v*x).sum(-1).abs().max()<1e-10
    q,_=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype));q[:,0]*=torch.linalg.det(q)
    vv,ww,bb=conditional(x@q,t,y@q,a@q)
    assert torch.allclose(vv,v@q,atol=1e-10) and torch.equal(bb,bad)
