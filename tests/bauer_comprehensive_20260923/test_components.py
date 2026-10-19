import copy
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.bauer_p2_20260923.common import cfg, ROOT, Model, generate
from tests.bauer_closed_loop_20260919.run import batch
from tests.mdgen_path_refactor_20260918.data import PathWindows
from .components import (CachedWindows, make_model, source_from_noise, synthetic_parents,
                         synthetic_paths, diagnostic_arrays)


@pytest.mark.parametrize('arm',['local_extra','temporal','global_temporal'])
def test_joint_rotation_tangent_anchor_checkpoint_and_sampling(arm):
    torch.set_num_threads(1);torch.manual_seed(77)
    config=dict(cfg(),width=16,depth=1)
    net=make_model(config,arm).double()
    # Nonzero residual verifies equivariance of the new path, not just old model.
    torch.nn.init.normal_(net.temporal_gate.weight,std=.05)
    initial=F.normalize(torch.randn(2,7,3,dtype=torch.float64),dim=-1)
    noise=torch.randn(2,11,7,3,dtype=torch.float64)
    x=source_from_noise(noise,initial,'source');times=torch.arange(11,dtype=torch.float64)*.1
    q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64));q[:,0]*=torch.linalg.det(q)
    tau=torch.ones(2,dtype=torch.float64)*.4
    a=net.velocity(x,tau,initial,times)
    b=net.velocity(x@q,tau,initial@q,times,frame=q)
    torch.testing.assert_close(b,a@q,atol=1e-10,rtol=1e-10)
    assert (x*a).sum(-1).abs().max()<1e-10 and a[:,0].abs().max()==0
    torch.testing.assert_close(source_from_noise(noise@q,initial@q,'source'),x@q,atol=1e-12,rtol=1e-12)
    class Rotated:
        def velocity(self,x,tau,initial,times):
            return net.velocity(x,tau,initial,times,frame=q)
    g=torch.Generator().manual_seed(1)
    p=generate(net,initial,times,g,4,source=x)
    r=generate(Rotated(),initial@q,times,g,4,source=x@q)
    torch.testing.assert_close(r,p@q,atol=1e-10,rtol=1e-10)
    assert (p.norm(dim=-1)-1).abs().max()<1e-12
    torch.testing.assert_close(p[:,0],initial,atol=0,rtol=0)
    other=make_model(config,arm).double();other.load_state_dict(copy.deepcopy(net.state_dict()))
    torch.testing.assert_close(other.velocity(x,tau,initial,times),a,atol=0,rtol=0)
    state=copy.deepcopy(net.state_dict());state['_extra_state']['periodic']=(True,)
    with pytest.raises(ValueError,match='contract'):
        other.load_state_dict(state)
    with pytest.raises(ValueError,match='unsupported'):
        net.velocity(x,tau,initial,times,temperature=100.)


def test_zero_residual_common_initialization_parameter_matching():
    config=dict(cfg(),width=16,depth=1);torch.manual_seed(11);base=Model(config)
    x=F.normalize(torch.randn(2,11,5,3),dim=-1);times=torch.arange(11)*.1;tau=torch.ones(2)*.5
    counts=[]
    for arm in ('local_extra','temporal','global_temporal'):
        torch.manual_seed(11);net=make_model(config,arm)
        counts.append(sum(p.numel() for p in net.parameters()))
        torch.testing.assert_close(net.velocity(x,tau,x[:,0],times),base.velocity(x,tau,x[:,0],times),atol=0,rtol=0)
    assert len(set(counts))==1


@pytest.mark.parametrize('size',[16,25,32])
def test_cache_exact_windows_and_rng(size):
    torch.set_num_threads(1);config=cfg();roots=[ROOT/config['data']/f'L{size}']
    lazy=PathWindows(roots,'train',1.,stride_frames=10)
    cached=CachedWindows(roots,'train',1.,stride_frames=10)
    assert lazy.records==cached.records and lazy.ends==cached.ends
    indices=[0,len(lazy)-1,-1]+np.random.default_rng(5).integers(len(lazy),size=20).tolist()
    for i in indices:
        a,b=lazy[i],cached[i]
        for key in a:
            if isinstance(a[key],np.ndarray):np.testing.assert_array_equal(a[key],b[key])
            else:assert a[key]==b[key]
    for _ in range(2):
        a,t=batch([lazy],np.random.default_rng(6),config,'cpu','B')
        b,u=batch([cached],np.random.default_rng(6),config,'cpu','B')
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        torch.testing.assert_close(t,u,atol=0,rtol=0)


def test_synthetic_support_independence_and_modes():
    torch.set_num_threads(1)
    a,b=synthetic_parents('train'),synthetic_parents('dev')
    assert not any(torch.equal(x,y) for x in a for y in b)
    initial=a[:1].expand(256,-1,-1)
    for kind in ('deterministic','gaussian','mixture'):
        s=synthetic_paths(initial,kind,torch.Generator().manual_seed(4))
        torch.testing.assert_close(s[:,0],initial,atol=0,rtol=0)
        assert (s.norm(dim=-1)-1).abs().max()<1e-6
        if kind=='deterministic':
            torch.testing.assert_close(s,s[:1].expand_as(s),atol=0,rtol=0)
        else:assert float(s[:,-1].var(0).sum())>.01
        if kind=='mixture':
            score=torch.linalg.cross(s[:,0],s[:,-1])[...,2].mean(-1)
            assert .35<float((score>0).float().mean())<.65


def test_flow_target_finite_difference_and_metric_covariance():
    torch.set_num_threads(1);torch.manual_seed(14)
    y=F.normalize(torch.randn(4,11,5,3,dtype=torch.float64),dim=-1)
    z=source_from_noise(torch.randn_like(y),y[:,0],'source')
    tau=torch.full((4,),.4,dtype=torch.float64);eps=1e-6
    x,v,_=path_interpolate(z,y,tau,torch.Generator().manual_seed(10))
    plus=path_interpolate(z,y,tau+eps,torch.Generator().manual_seed(10))[0]
    minus=path_interpolate(z,y,tau-eps,torch.Generator().manual_seed(10))[0]
    torch.testing.assert_close((plus-minus)/(2*eps),v,atol=1e-8,rtol=1e-8)
    d=np.diff(y.numpy(),axis=1);m=diagnostic_arrays(y.numpy())
    np.testing.assert_allclose(m['increment_variance'],d.var(0,ddof=1).sum(-1).mean(-1))
    np.testing.assert_allclose(np.diag(m['two_time_spin_correlation']),1,atol=1e-12)
