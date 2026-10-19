import numpy as np
import torch
from torch.nn import functional as F
from tests.bauer_p2_20260923.common import generate
from tests.cross_dimension_20260924.model import source
from .objectives import draw_tau,statistics,joint_loss,differentiable_sample


def test_time_density_and_importance_correction():
    torch.set_num_threads(1)
    for stop in (.95,1.):
        tau,w=draw_tau(200000,stop,'importance',torch.Generator().manual_seed(31),'cpu')
        assert abs(float(w.mean())-1)<.006
        assert abs(float((w*tau).mean())-stop/2)<.004
        assert abs(float((w*tau.square()).mean())-stop**2/3)<.004
        assert abs(float((tau<.1).float().mean())-(.5+.05/stop))<.004
        t2,w2=draw_tau(200000,stop,'early',torch.Generator().manual_seed(31),'cpu')
        assert torch.equal(tau,t2) and torch.equal(w2,torch.ones_like(w2))


def test_statistics_rotation_permutation_and_reference_formula():
    torch.manual_seed(32);x=F.normalize(torch.randn(16,11,7,3,dtype=torch.float64),dim=-1)
    x[:,0]=x[:1,0];v,c=statistics(x)
    a=x.numpy();d=np.diff(a,axis=1)
    assert np.allclose(v.numpy(),d.var(0,ddof=1).sum(-1).mean(-1))
    assert np.allclose(c.numpy(),np.einsum('btnc,bunc->tu',a,a)/(len(a)*7))
    q,_=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype));q[:,0]*=torch.linalg.det(q)
    vr,cr=statistics(x@q)
    assert torch.allclose(v,vr) and torch.allclose(c,cr)
    loss,_=joint_loss(x,x.flip(0));assert float(loss)<1e-20


def test_differentiable_sampler_gradient_and_frozen_sampler_agreement():
    torch.manual_seed(33)
    class Net(torch.nn.Module):
        def __init__(self):super().__init__();self.a=torch.nn.Parameter(torch.tensor(.2,dtype=torch.float64))
        def velocity(self,x,t,initial,times):
            v=self.a*torch.linalg.cross(x,x.new_tensor([0.,0.,1.]).expand_as(x))
            return torch.cat((torch.zeros_like(v[:,:1]),v[:,1:]),1)
    net=Net();initial=F.normalize(torch.randn(1,7,3,dtype=torch.float64),dim=-1).expand(8,-1,-1)
    z=source(initial,11,torch.Generator().manual_seed(34),2.)
    times=torch.arange(11,dtype=z.dtype)*.1
    x=differentiable_sample(net,initial,times,z,8)
    old=generate(net,initial,times,None,8,source=z)
    assert torch.equal(x,old) and torch.equal(x[:,0],initial)
    assert (x.norm(dim=-1)-1).abs().max()<1e-12
    reference=source(initial,11,torch.Generator().manual_seed(35),2.)
    loss,_=joint_loss(x,reference);loss.backward()
    assert torch.isfinite(net.a.grad) and net.a.grad.abs()>1e-8


def test_task_matrix():
    from .run import tasks
    rows=tasks();assert len(rows)==21
    for seed in (11,22,33):
        assert len([r for r in rows if r['seed']==seed and r['domain']=='bauer'])==4
        assert len([r for r in rows if r['seed']==seed and r['domain']=='toy'])==3
