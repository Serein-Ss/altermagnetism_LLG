import torch
import pytest
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.bauer_p2_20260923.common import generate
from tests.cross_dimension_20260924.model import source
from .run import integrate,wilson,tasks


def test_midpoint_order_constraints_and_euler_compatibility():
    torch.set_num_threads(1);torch.manual_seed(22)
    x=F.normalize(torch.randn(3,11,7,3,dtype=torch.float64),dim=-1)
    axis=x.new_tensor([0.,0.,1.]).expand_as(x)
    def field(v,t):
        result=torch.linalg.cross(axis,v);result[:,0]=0;return result
    angle=x.new_tensor(1.);c=angle.cos();s=angle.sin()
    target=torch.stack((c*x[...,0]-s*x[...,1],s*x[...,0]+c*x[...,1],x[...,2]),-1);target[:,0]=x[:,0]
    err=[]
    for n in (16,32,64):
        y=integrate(field,x,0.,1.,n,'midpoint');err.append(float((y-target).square().mean().sqrt()))
        assert torch.equal(y[:,0],x[:,0]);assert (y.norm(dim=-1)-1).abs().max()<1e-12
    assert err[0]/err[1]>3.5 and err[1]/err[2]>3.5
    class Net:
        def velocity(self,v,t,initial,times):return field(v,t)
    a=generate(Net(),x[:,0],torch.arange(11)*.1,None,32,source=x)
    b=integrate(field,x,0.,1.,32,'euler');assert torch.equal(a,b)
    with pytest.raises(ValueError):integrate(field,x,1.,0.,32,'euler')


def test_restart_anchor_and_probability_interval():
    g=torch.Generator().manual_seed(2)
    initial=F.normalize(torch.randn(4,16,3,generator=g),dim=-1)
    z=source(initial,11,g,0.);y=source(initial,11,g,0.)
    for t in (0.,.03,.5,.95):
        x,v,_=path_interpolate(z,y,torch.full((4,),t),g)
        assert torch.equal(x[:,0],initial);assert v[:,0].abs().max()==0
        assert (x.norm(dim=-1)-1).abs().max()<2e-5
    lo,hi=wilson(128,256);assert lo<.5<hi and hi-lo<.13
    assert len(tasks())==9
