import math
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from scripts.model.sphere import geodesic_interpolate, sphere_exp
from tests.literature_path_campaign_20260920.train_miltat import physical_config
from tests.literature_preflight_20260917.models import build
from .model import GeometryGraph, source, sample
from .oracle import conditional, targets


@pytest.mark.parametrize('arm',['shifted','uniform','global'])
def test_geometry_rotation_active_tangent_and_checkpoint(arm):
    torch.set_num_threads(1);torch.manual_seed(924)
    cell,_,_=build(physical_config(),'miltat',shape=(5,3))
    net=GeometryGraph(arm,width=16).double()
    x=F.normalize(torch.randn(2,4,5,3,2,3,dtype=torch.float64),dim=-1)
    initial=x[:,0].clone();times=torch.arange(4,dtype=x.dtype)*.1;tau=x.new_tensor([.3,.7])
    q,_=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype));q[:,0]*=torch.linalg.det(q)
    eye=torch.eye(3,dtype=x.dtype)
    a=net(x,tau,initial,cell,eye,0.,times)
    b=net(x@q,tau,initial@q,cell,q,0.,times)
    assert torch.allclose(a@q,b,atol=1e-8,rtol=1e-7)
    assert (a*x).sum(-1).abs().max()<1e-10 and a[:,0].abs().max()==0
    correction=net.correction(x,tau,initial,cell,eye,times)
    assert correction[:,1:].square().mean() > 1e-6
    correction.square().mean().backward()
    assert net.geometry_message[-1].weight.grad.norm()>1e-6
    (i,j),d=net.geometry(cell,x,eye)
    assert torch.equal(i[::2],j[1::2])
    assert torch.allclose(d[::2],-d[1::2])
    assert len(i)==sum(2*(5-abs(dx))*(3-abs(dy)) for _,_,(dx,dy),_ in cell.templates)
    copied=GeometryGraph(arm,width=16).double();copied.load_state_dict(net.state_dict())
    cell.periodic=(True,False)
    with pytest.raises(ValueError):net(x,tau,initial,cell,eye,0.,times)
    with pytest.raises(ValueError):GeometryGraph('uniform' if arm!='uniform' else 'shifted',width=16).load_state_dict(net.state_dict())


def test_energy_gradient_and_independent_pair_sum():
    cell,_,_=build(physical_config(),'miltat',shape=(3,2))
    torch.manual_seed(15)
    x=F.normalize(torch.randn(1,3,2,2,3,dtype=torch.float64),dim=-1).requires_grad_()
    e=cell.energy(x);g=torch.autograd.grad(e.sum(),x)[0]
    assert torch.allclose(-g,cell.field(x),atol=1e-10)
    net=GeometryGraph('shifted',16);(i,j),d=net.geometry(cell,x,torch.eye(3,dtype=x.dtype))
    i,j,d=i[::2],j[::2],d[::2];s=x.reshape(-1,3)
    direct=-(s[i]*s[j]).sum()-cell.reduced['K']*s[:,2].square().sum()
    dv=cell.reduced['D']*torch.linalg.cross(torch.tensor([0.,0.,1.],dtype=x.dtype).expand_as(d),d)
    direct-= (torch.linalg.cross(s[i],s[j])*dv).sum()
    positions=cell.positions(x).reshape(-1,3)
    for a in range(len(s)):
        for b in range(a):
            r=positions[a]-positions[b];length=r.norm();u=r/length
            direct+=cell.reduced['dipole']*((s[a]*s[b]).sum()-3*(s[a]*u).sum()*(s[b]*u).sum())/length**3
    assert torch.allclose(e[0],direct,atol=1e-10)


def test_sampler_anchor_norm_and_reproducibility():
    cell,_,_=build(physical_config(),'miltat',shape=(3,2))
    net=GeometryGraph('global',16).eval()
    a=F.normalize(torch.randn(1,3,2,2,3),dim=-1);times=torch.arange(3)*.1
    x=sample(net,a,cell,times,4,torch.Generator().manual_seed(3))
    y=sample(net,a,cell,times,4,torch.Generator().manual_seed(3))
    assert torch.equal(x,y) and torch.equal(x[:,0],a)
    assert (x.norm(dim=-1)-1).abs().max()<2e-6


def test_exact_oracle_single_mode_matches_geodesic_velocity():
    torch.manual_seed(25)
    initial=F.normalize(torch.randn(32,4,3,dtype=torch.float64),dim=-1)
    y=targets(initial,1);z=source(initial,11,torch.Generator().manual_seed(6),0.)
    tau=torch.linspace(.02,.94,len(z),dtype=z.dtype)
    x,v=geodesic_interpolate(z,y[:,0],tau)
    expected,weights,bad=conditional(x,tau,y)
    assert not bad.any() and torch.all(weights==1)
    assert torch.allclose(expected[:,1:],v[:,1:],atol=2e-5,rtol=2e-5)


def test_oracle_prior_symmetry_density_normalization_and_support():
    initial=F.normalize(torch.randn(20,4,3),dim=-1);ys=targets(initial)
    x=source(initial,11,torch.Generator().manual_seed(1),0.)
    _,w,bad=conditional(x,torch.zeros(20),ys)
    assert torch.allclose(w,torch.full_like(w,.5),atol=1e-6) and not bad.any()
    _,w2,_=conditional(x,torch.zeros(20),ys.flip(1))
    assert torch.allclose(w,w2.flip(1))
    for tau in (.1,.5,.95):
        a=1-tau;r=np.linspace(0,a*math.pi,10001)
        # Density integrates against 2*pi*sin(r) dr; constants cancel to 1/(2a).
        assert abs(np.trapz(np.sin(r/a)/(2*a),r)-1)<1e-7
    _,_,bad=conditional(x,torch.full((20,),.95),ys)
    assert bad.all()
