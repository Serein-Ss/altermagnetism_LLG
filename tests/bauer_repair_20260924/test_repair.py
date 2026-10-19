import copy
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from scripts.model.sphere import sphere_exp
from tests.bauer_p2_20260923.common import cfg, Model, generate
from tests.bauer_comprehensive_20260923.components import VectorTemporalModel
from .model import RepairModel, correlated_noise, source_from_draws, draw_source, relational_pool


def example(n=7):
    torch.set_num_threads(1);torch.manual_seed(924)
    config=dict(cfg(),width=16,depth=1)
    initial=F.normalize(torch.randn(3,n,3,dtype=torch.float64),dim=-1)
    x=F.normalize(torch.randn(3,11,n,3,dtype=torch.float64),dim=-1);x[:,0]=initial
    return config,x,initial,torch.ones(3,dtype=torch.float64)*.4,torch.arange(11,dtype=torch.float64)*.1


def activate(net):
    torch.nn.init.normal_(net.residual_gate.weight,std=.1)
    torch.nn.init.normal_(net.residual_gate.bias,std=.1)


def test_reproduce_old_inactive_control_and_fix():
    config,x,initial,tau,times=example()
    old=VectorTemporalModel(config,mixing=False).double()
    torch.nn.init.normal_(old.temporal_gate.weight)
    state={k:v for k,v in old.state_dict().items() if not k.startswith('temporal_')}
    base=Model(config).double();state['_extra_state']=base.get_extra_state();base.load_state_dict(state)
    assert (old.velocity(x,tau,initial,times)-base.velocity(x,tau,initial,times)).abs().max()<1e-12
    fixed=RepairModel(config,'local_fixed').double();activate(fixed)
    assert fixed.residual(x,tau,initial,times).square().mean()>1e-5


@pytest.mark.parametrize('arm',['local_fixed','temporal_fixed','relational_global'])
def test_gate_and_internal_gradients_become_active(arm):
    config,x,initial,tau,times=example()
    net=RepairModel(config,arm).double();optimizer=torch.optim.Adam(net.parameters(),lr=.01)
    target=F.normalize(torch.randn_like(x),dim=-1)
    assert net.residual(x,tau,initial,times).abs().max()==0
    loss=(net.residual(x,tau,initial,times)-target).square().mean();loss.backward()
    assert net.residual_gate.weight.grad.norm()>1e-6
    optimizer.step();optimizer.zero_grad()
    correction=net.residual(x,tau,initial,times)
    assert correction.norm()>1e-5
    (correction-target).square().mean().backward()
    for layer in (net.residual_input[0],net.residual_qk,net.residual_gate):
        assert torch.isfinite(layer.weight.grad).all() and layer.weight.grad.norm()>1e-8


@pytest.mark.parametrize('arm',['local_fixed','temporal_fixed','relational_global','relational_shared'])
def test_joint_rotation_sampling_and_checkpoint(arm):
    config,x,initial,tau,times=example()
    net=RepairModel(config,arm).double();activate(net)
    q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64));q[:,0]*=torch.linalg.det(q)
    a=net.velocity(x,tau,initial,times);b=net.velocity(x@q,tau,initial@q,times,frame=q)
    torch.testing.assert_close(b,a@q,atol=1e-10,rtol=1e-10)
    assert (a*x).sum(-1).abs().max()<1e-10 and a[:,0].abs().max()==0
    class Rotated:
        def velocity(self,x,tau,initial,times):return net.velocity(x,tau,initial,times,frame=q)
    g=torch.Generator().manual_seed(3)
    p=generate(net,initial,times,g,4,source=x)
    r=generate(Rotated(),initial@q,times,g,4,source=x@q)
    torch.testing.assert_close(r,p@q,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(p[:,0],initial,atol=0,rtol=0)
    assert (p.norm(dim=-1)-1).abs().max()<1e-12
    other=RepairModel(config,arm).double();other.load_state_dict(copy.deepcopy(net.state_dict()))
    torch.testing.assert_close(other.velocity(x,tau,initial,times),a,atol=0,rtol=0)
    wrong=copy.deepcopy(net.state_dict());wrong['_extra_state']['rho']=.2
    with pytest.raises(ValueError,match='contract'):other.load_state_dict(wrong)
    with pytest.raises(ValueError,match='unsupported'):net.velocity(x,tau,initial,times,temperature=100.)


def test_information_paths_nonzero_and_parameter_matching():
    config,x,initial,tau,times=example(9);counts=[];effects={}
    for arm in ['local_fixed','temporal_fixed','relational_global']:
        net=RepairModel(config,arm).double();counts.append(sum(p.numel() for p in net.parameters()))
        # Constant weights isolate VALUE communication from scalar score changes.
        with torch.no_grad():
            net.residual_qk.weight.zero_();net.residual_qk.bias.zero_()
            net.residual_gate.weight.zero_();net.residual_gate.bias.zero_();net.residual_gate.bias[4]=1.
        base=net.residual(x,tau,initial,times)
        changed=x.clone();changed[:,9,0]=sphere_exp(changed[:,9,0],torch.tensor([.03,-.02,.04],dtype=x.dtype).expand(3,3))
        delta=net.residual(changed,tau,initial,times)-base
        effects[arm]=(float(delta[:,1,0].norm()),float(delta[:,1,8].norm()))
    assert len(set(counts))==1
    assert max(effects['local_fixed'])<1e-12
    assert effects['temporal_fixed'][0]>1e-5 and effects['temporal_fixed'][1]<1e-12
    assert effects['relational_global'][1]>1e-5


def test_pool_cancellation_counts_and_odd_size():
    _,x,initial,_,_=example(7);anchor=initial[:,None].expand_as(x)
    torch.testing.assert_close(relational_pool(x,anchor),torch.stack([torch.linalg.cross(anchor[:,:,j],x[:,:,j]) for j in range(7)],2).sum(2,keepdim=True)/7)
    s=torch.tensor([[[[1.,0,0],[-1.,0,0]]]])
    y=torch.tensor([[[[0.,1,0],[0.,1,0]]]])
    assert relational_pool(y,s).abs().max()==0  # no spurious unit coarse vector


def test_shared_source_marginals_rotation_and_rng():
    torch.set_num_threads(1);torch.manual_seed(5)
    n=40000
    initial=F.normalize(torch.tensor([[1.,2.,3.],[-2.,1.,1.]],dtype=torch.float64),dim=-1)[None].expand(n,-1,-1)
    eps=torch.randn(n,2,2,3,dtype=torch.float64)
    g=torch.randn(n,1,1,3,dtype=torch.float64);h=torch.randn(n,1,1,1,dtype=torch.float64)
    a=correlated_noise(eps,g,h,initial,0.);b=correlated_noise(eps,g,h,initial,.7)
    for value in [a,b]:
        torch.testing.assert_close(torch.cov(value[:,1,0].T),torch.eye(3,dtype=value.dtype),atol=.025,rtol=0)
        assert value[:,1,0].mean(0).abs().max()<.02
    assert (b[:,0,0]*b[:,1,0]).sum(-1).mean()>.9
    assert abs(float((a[:,0,0]*a[:,1,0]).sum(-1).mean()))<.05
    z0=source_from_draws(eps,g,h,initial,'temporal_fixed')
    z1=source_from_draws(eps,g,h,initial,'temporal_shared')
    torch.testing.assert_close(z0[:,1,0].mean(0),z1[:,1,0].mean(0),atol=.015,rtol=0)
    q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64));q[:,0]*=torch.linalg.det(q)
    zrot=source_from_draws(eps[:3]@q,g[:3]@q,h[:3],initial[:3]@q,'temporal_shared')
    torch.testing.assert_close(zrot,z1[:3]@q,atol=1e-12,rtol=1e-12)
    gen=torch.Generator().manual_seed(8);state=gen.get_state()
    z,f=draw_source((3,11,2,3),initial[:3],gen,'temporal_fixed')
    other=torch.Generator();other.set_state(state)
    zz,ff=draw_source((3,11,2,3),initial[:3],other,'temporal_shared')
    torch.testing.assert_close(f,ff,atol=0,rtol=0)
    assert not torch.equal(z,zz)
    torch.testing.assert_close(z[:,0],initial[:3],atol=0,rtol=0)
    with pytest.raises(ValueError):correlated_noise(eps,g,h,initial,1.)
