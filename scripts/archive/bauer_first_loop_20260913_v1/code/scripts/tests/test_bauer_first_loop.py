import copy
import numpy as np
import torch
from scripts.workflow.bauer_first_loop import ar_loss,features
from scripts.workflow.bauer_campaign import graph
from scripts.model.gate_f import build_model,matching_loss,reference_path


def test_vectorized_ar_matches_independent_frame_loop_loss_and_gradients():
    torch.manual_seed(17);dtype=torch.float64
    initial=torch.randn(2,1,5,1,3,dtype=dtype);initial/=initial.norm(dim=-1,keepdim=True)
    gen=torch.Generator().manual_seed(13);y=reference_path(initial,4,gen)
    c=torch.tensor([[.11,.1],[.13,.1]],dtype=dtype);t=torch.linspace(0,1,4,dtype=dtype).expand(2,-1)
    g=graph(5,dtype=dtype);axis=torch.tensor([[0.,0.,1.]],dtype=dtype).expand(2,-1)
    a=build_model('autoregressive',width=8,blocks=1).to(dtype);b=copy.deepcopy(a)
    la,_=matching_loss(a,y,c,t,g,axis,gen,kind='autoregressive');lb,_=ar_loss(b,y,c,t,g,axis)
    la.backward();lb.backward()
    torch.testing.assert_close(la,lb,atol=1e-12,rtol=1e-12)
    for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa.grad,pb.grad,atol=1e-12,rtol=1e-12)


def test_primary_features_exactly_energy_and_magnetization():
    p=np.zeros((2,201,25,3));p[...,2]=1
    f,obs=features(p)
    assert f.shape==(2,44)
    np.testing.assert_allclose(f.reshape(2,11,4),np.broadcast_to([-1.06,0,0,1],(2,11,4)))
