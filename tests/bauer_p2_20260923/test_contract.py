import io
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from scripts.core.cell_hamiltonian import CellHamiltonian
from scripts.model.gate_f import path_interpolate
from .common import cfg, energy, Model, generate, compose, comparison, statistics


def setup():
    torch.set_num_threads(1)
    torch.manual_seed(37)
    config = dict(cfg(), width=16, depth=1)
    return Model(config).double(), F.normalize(torch.randn(3, 5, 3, dtype=torch.float64), dim=-1)


def test_energy_bonds_gradient_and_units():
    s = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
    cell = CellHamiltonian((5,), 1, [(0,0,(1,),1)], periodic=(False,), anisotropy=(0,0,.1))
    torch.testing.assert_close(energy(s)*5, cell.energy(s.unsqueeze(-2)))
    grad = torch.autograd.grad(energy(s).sum()*5, s)[0]
    torch.testing.assert_close(-grad, cell.field(s.detach().unsqueeze(-2)).squeeze(-2))
    plus = s.detach().clone(); minus = plus.clone()
    plus[0,2,1] += 1e-6; minus[0,2,1] -= 1e-6
    assert abs(float((energy(plus)[0]-energy(minus)[0])*5/2e-6-grad[0,2,1])) < 1e-8
    uniform = torch.zeros(1,5,3); uniform[...,2]=1
    assert abs(energy(uniform).item()-(-4/5-.1)) < 1e-7
    assert abs(cfg()['theta']*cfg()['exchange_meV']/.08617333262-cfg()['temperature_K']) < 1e-5


@pytest.mark.parametrize('angle', [0., 1e-9, np.pi-1e-9, np.pi])
def test_spherical_degeneracy(angle):
    z = torch.tensor([[[[0.,0.,1.]],[[0.,0.,1.]]]], dtype=torch.float64)
    y = z.clone(); y[:,1,:,0]=np.sin(angle); y[:,1,:,2]=np.cos(angle)
    x,v,d = path_interpolate(z,y,torch.tensor([.5]),torch.Generator().manual_seed(2))
    assert torch.isfinite(x).all() and torch.isfinite(v).all()
    assert (x.norm(dim=-1)-1).abs().max() < 1e-8
    assert (x*v).sum(-1).abs().max() < 1e-8


def test_condition_rng_checkpoint_and_composition():
    net, initial = setup()
    times = torch.arange(11, dtype=torch.float64)*.1
    with pytest.raises(ValueError, match='unsupported'):
        net.velocity(initial[:,None].expand(-1,11,-1,-1), torch.zeros(3), initial, times, temperature=100.)
    g = torch.Generator().manual_seed(18)
    a,state = compose(net,initial,1.,g,2)
    assert state['resumable']
    resumed = torch.Generator(); resumed.set_state(state['rng'])
    b,_ = compose(net,state['state'],1.,resumed,2,elapsed_ps=1.)
    whole,_ = compose(net,initial,2.,torch.Generator().manual_seed(18),2)
    torch.testing.assert_close(torch.cat((a,b[:,1:]),1),whole,rtol=0,atol=0)
    prefix,state = compose(net,initial,.3,torch.Generator().manual_seed(18),2)
    torch.testing.assert_close(prefix,whole[:,:4],rtol=0,atol=0)
    assert not state['resumable']
    with pytest.raises(ValueError,match='grid'):
        compose(net,initial,.35,g,2)
    buf=io.BytesIO(); torch.save(net.state_dict(),buf); buf.seek(0)
    other=Model(net.config).double(); other.load_state_dict(torch.load(buf,weights_only=True))
    c=generate(other,initial,times,torch.Generator().manual_seed(18),2)
    torch.testing.assert_close(c,a,atol=0,rtol=0)


def test_metric_replication_and_unbiased_covariance():
    rng=np.random.default_rng(8)
    s=rng.normal(size=(16,11,8,3)); s/=np.linalg.norm(s,axis=-1,keepdims=True)
    result=comparison(s,s,repeats=5)
    assert all(x['rmse']==0 for x in result.values())
    values=statistics(s)
    d=np.diff(s,axis=1)
    np.testing.assert_allclose(values['increment_cov'][2,3],np.cov(d[:,2,3].T,ddof=1))


def test_wrapper_joint_covariance_and_no_clean_future_condition():
    net,initial=setup()
    x=F.normalize(torch.randn(3,11,5,3,dtype=torch.float64),dim=-1); x[:,0]=initial
    q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64)); q[:,0]*=torch.linalg.det(q)
    times=torch.arange(11,dtype=torch.float64)*.1; tau=torch.ones(3,dtype=torch.float64)*.4
    a=net.velocity(x,tau,initial,times)
    b=net.velocity(x@q,tau,initial@q,times,frame=torch.eye(3,dtype=torch.float64)@q)
    torch.testing.assert_close(b,a@q,atol=1e-10,rtol=1e-10)
    assert (a*x).sum(-1).abs().max()<1e-10
    assert a[:,0].abs().max()==0
