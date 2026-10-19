import copy
import pytest
import torch
from torch.nn import functional as F
from tests.spin_path_transformer.model import SpinPathTransformer
from tests.spin_path_transformer.test_model import activate
from tests.transport_audit.run import integrate
from .model import RepairedNet,AnchorBlock,RelativeAttention


def inputs():
    torch.set_num_threads(1);torch.manual_seed(933)
    x=F.normalize(torch.randn(2,11,7,3,dtype=torch.float64),dim=-1)
    return x,x[:,0].clone(),torch.tensor([.2,.8],dtype=x.dtype),torch.arange(11,dtype=x.dtype)*.1


@pytest.mark.parametrize('repair',['control','anchor','relative_time'])
def test_zero_init_equivalence_checkpoint_and_contract(repair):
    x,a,tau,t=inputs();base=SpinPathTransformer('bauer','hierarchical',width=16,depth=2).double();activate(base)
    expected=base.velocity(x,tau,a,t)
    net=RepairedNet(copy.deepcopy(base),repair)
    torch.testing.assert_close(net.velocity(x,tau,a,t),expected,atol=1e-10,rtol=1e-9)
    same=RepairedNet(copy.deepcopy(base),repair);same.load_state_dict(net.state_dict())
    torch.testing.assert_close(same.velocity(x,tau,a,t),net.velocity(x,tau,a,t))
    wrong=RepairedNet(copy.deepcopy(base),'anchor' if repair=='control' else 'control')
    with pytest.raises(ValueError):wrong.load_state_dict(net.state_dict())
    with pytest.raises(ValueError):net.velocity(x,tau,a,t*2)
    with pytest.raises(ValueError):net.velocity(x,tau,a,t,temperature=10.)


@pytest.mark.parametrize('repair',['anchor','relative_time'])
def test_active_adapter_rotation_gradient_and_sampling(repair):
    x,a,tau,t=inputs();base=SpinPathTransformer('bauer','hierarchical',width=16,depth=2).double();activate(base)
    net=RepairedNet(base,repair)
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module,AnchorBlock):module.film[-1].weight.normal_(std=.02)
            if isinstance(module,RelativeAttention):module.bias[-1].weight.normal_(std=.02)
    q,_=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype));q[:,0]*=torch.linalg.det(q)
    v=net.velocity(x,tau,a,t);vr=net.velocity(x@q,tau,a@q,t,frame=q)
    torch.testing.assert_close(v@q,vr,atol=1e-9,rtol=1e-8)
    assert (v*x).sum(-1).abs().max()<1e-10 and v[:,0].abs().max()==0
    v.square().mean().backward()
    grads=[p.grad.norm().item() for n,p in net.named_parameters() if n.endswith(('film.2.weight','bias.2.weight'))]
    assert grads and all(g>1e-12 for g in grads)
    y=integrate(lambda z,u:net.velocity(z,u,a,t),x,0.,1.,8,'euler')
    yr=integrate(lambda z,u:net.velocity(z,u,a@q,t,frame=q),x@q,0.,1.,8,'euler')
    torch.testing.assert_close(y@q,yr,atol=1e-8,rtol=1e-7)
    assert (y.norm(dim=-1)-1).abs().max()<1e-10
    torch.testing.assert_close(y[:,0],a,atol=0,rtol=0)


def test_relative_bias_uses_physical_lag():
    inputs();attention=torch.nn.MultiheadAttention(16,4,batch_first=True).double()
    layer=RelativeAttention(attention).double()
    with torch.no_grad():layer.bias[-1].weight.normal_(std=.2)
    q=torch.randn(2,4,16,dtype=torch.float64)
    layer.time=torch.arange(4,dtype=q.dtype)*.1;a=layer(q,q,q)[0]
    layer.time=layer.time+.5;b=layer(q,q,q)[0]
    torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-9)
    layer.time=torch.arange(4,dtype=q.dtype)*.2;c=layer(q,q,q)[0]
    assert (a-c).abs().max()>1e-6
