import pytest
import torch
from torch.nn import functional as F
from scripts.core.cell_hamiltonian import CellHamiltonian
from scripts.model.sphere import sphere_exp
from tests.literature_preflight_20260917.models import build
from tests.literature_path_campaign_20260920.train_miltat import physical_config
from .model import SpinPathTransformer,geometry,pool_mean


def activate(net):
    with torch.no_grad():
        net.output.weight.normal_(std=.03)
        for b in net.blocks:
            b.modulation[-1].bias.fill_(.2)


@pytest.mark.parametrize('arm',['local','latent','hierarchical','dense'])
@pytest.mark.parametrize('ndim',[1,2,3])
def test_joint_rotation_tangent_nontrivial_and_gradients(arm,ndim):
    torch.set_num_threads(1);torch.manual_seed(52)
    shape=(5,)*ndim
    cell=CellHamiltonian(shape,1,[(0,0,tuple(int(i==j) for i in range(ndim)),1.) for j in range(ndim)],periodic=(False,)*ndim)
    net=SpinPathTransformer('software',arm,width=16,depth=2,periodic=cell.periodic).double();activate(net)
    x=F.normalize(torch.randn(2,3,*shape,1,3,dtype=torch.float64),dim=-1)
    initial=x[:,0];tau=x.new_tensor([.2,.8]);time=torch.arange(3,dtype=x.dtype)*.1;eye=torch.eye(3,dtype=x.dtype)
    q,_=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype));q[:,0]*=torch.linalg.det(q)
    v=net(x,tau,initial,cell,eye,0.,time);vr=net(x@q,tau,initial@q,cell,q,0.,time)
    assert v[:,1:].square().mean()>1e-6
    assert torch.allclose(v@q,vr,atol=1e-8,rtol=1e-7)
    assert (x*v).sum(-1).abs().max()<1e-9 and v[:,0].abs().max()==0
    s=sphere_exp(x,v*.03);assert (s.norm(dim=-1)-1).abs().max()<1e-10
    v.square().mean().backward()
    assert net.blocks[0].edge[-1].weight.grad.norm()>1e-8
    if arm in ('latent','hierarchical'):
        assert net.blocks[0].read.in_proj_weight.grad.norm()>1e-8
        assert net.blocks[0].write.in_proj_weight.grad.norm()>1e-8


def test_oriented_triangular_and_periodic_image_geometry():
    cell,_,_=build(physical_config(),'miltat',shape=(5,3))
    g=geometry(cell,'cpu',torch.float64)
    assert torch.allclose(g['direction'].norm(dim=-1),torch.ones(len(g['direction']),dtype=torch.float64))
    assert torch.allclose(g['direction'][::2],-g['direction'][1::2])
    assert int(g['count'].sum())==30 and sorted(g['count'].tolist())==[6.,24.]
    chain=CellHamiltonian((5,),1,[(0,0,(1,),1.)],periodic=(True,))
    g=geometry(chain,'cpu',torch.float64)
    assert len(g['edges'][0])==10
    assert torch.equal(g['direction'][-2],torch.tensor([1.,0.,0.],dtype=torch.float64))
    assert torch.equal(g['degree'],torch.full((5,),2.,dtype=torch.float64))


def test_pool_cancellation_and_odd_counts():
    cell=CellHamiltonian((5,),1,[(0,0,(1,),1.)],periodic=(False,))
    g=geometry(cell,'cpu',torch.float64)
    x=torch.tensor([[1.,0.,0.],[-1.,0.,0.],[1.,0.,0.],[-1.,0.,0.],[.2,0.,0.]],dtype=torch.float64)
    pooled=pool_mean(x,g['group'],g['count'])
    assert torch.equal(pooled[0],torch.zeros(3,dtype=x.dtype))
    assert torch.allclose(pooled[1],x[-1]) and g['count'].tolist()==[4.,1.]


def test_checkpoint_boundary_temperature_and_active_training():
    torch.manual_seed(8);torch.set_num_threads(1)
    net=SpinPathTransformer('bauer','hierarchical',width=16,depth=2)
    same=SpinPathTransformer('bauer','hierarchical',width=16,depth=2);same.load_state_dict(net.state_dict())
    with pytest.raises(ValueError):SpinPathTransformer('bauer','latent',width=16,depth=2).load_state_dict(net.state_dict())
    with pytest.raises(ValueError):SpinPathTransformer('bauer','hierarchical',width=16,depth=2,periodic=(True,)).load_state_dict(net.state_dict())
    x=F.normalize(torch.randn(4,11,7,3),dim=-1);t=torch.arange(11)*.1;tau=torch.rand(4)
    with pytest.raises(ValueError):net.velocity(x,tau,x[:,0],t,temperature=10.)
    opt=torch.optim.Adam(net.parameters(),lr=.01)
    target=torch.linalg.cross(x,torch.tensor([0.,0.,1.]).expand_as(x))
    for _ in range(8):
        pred=net.velocity(x,tau,x[:,0],t);loss=(pred[:,1:]-target[:,1:]).square().mean()
        opt.zero_grad();loss.backward();opt.step()
    assert net.blocks[0].read.in_proj_weight.grad.norm()>0
    assert net.velocity(x,tau,x[:,0],t)[:,0].abs().max()==0


def test_remote_information_path_exists():
    torch.manual_seed(27);torch.set_num_threads(1)
    x=F.normalize(torch.randn(1,3,25,3),dim=-1);initial=x[:,0].clone()
    x2=x.clone();x2[:,1:,-1]*=-1
    t=torch.arange(3)*.1;tau=torch.tensor([.5])
    for arm in ('local','latent','hierarchical'):
        net=SpinPathTransformer('software',arm,width=16,depth=2);activate(net)
        a=net.velocity(x,tau,initial,t);b=net.velocity(x2,tau,initial,t)
        difference=(a[:,:,0]-b[:,:,0]).abs().max()
        assert difference<1e-7 if arm=='local' else difference>1e-8
