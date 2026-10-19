import torch
from torch.nn import functional as F
from tests.flow_training_repair.objectives import joint_loss
from .objectives import cross_square,loss_parts


def test_plugin_control_and_component_additivity():
    torch.manual_seed(39)
    x=F.normalize(torch.randn(8,11,5,3),dim=-1);x[:,0]=x[:1,0]
    r=F.normalize(torch.randn_like(x),dim=-1);r[:,0]=x[:1,0]
    assert torch.allclose(loss_parts(x,r,'joint')[0],joint_loss(x,r)[0])
    assert torch.allclose(loss_parts(x,r,'variance')[0]+loss_parts(x,r,'correlation')[0],loss_parts(x,r,'joint')[0])


def test_cross_estimator_expectation_and_gradient():
    torch.set_num_threads(1);g=torch.Generator().manual_seed(92801)
    eps=torch.randn(100000,8,generator=g,dtype=torch.float64)
    sigma=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
    x=eps*sigma;a=x[:,:4].var(1,unbiased=True);b=x[:,4:].var(1,unbiased=True)
    cross=cross_square(a,b,torch.ones_like(a));gc=torch.autograd.grad(cross,sigma,retain_graph=True)[0]
    plugin=(x.var(1,unbiased=True)-1).square().mean();gp=torch.autograd.grad(plugin,sigma)[0]
    assert abs(float(cross))<.01 and abs(float(gc))<.05
    # Gaussian variance estimator: gradient at sigma=1 is8/(n-1).
    assert abs(float(gp)-8/7)<.05


def test_cross_negative_values_and_covariance():
    assert cross_square(torch.tensor([2.]),torch.tensor([0.]),torch.tensor([1.]))==-1
    torch.manual_seed(40)
    x=F.normalize(torch.randn(8,11,5,3,dtype=torch.float64),dim=-1);x[:,0]=x[:1,0]
    r=F.normalize(torch.randn_like(x),dim=-1);r[:,0]=x[:1,0]
    x.requires_grad_();loss,_=loss_parts(x,r,'cross');grad=torch.autograd.grad(loss,x)[0]
    assert torch.isfinite(grad).all() and grad.abs().max()>0
    q,_=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype));q[:,0]*=torch.linalg.det(q)
    assert torch.allclose(loss,loss_parts(x@q,r@q,'cross')[0])
    assert torch.allclose(loss,loss_parts(torch.cat((x[4:],x[:4])),r,'cross')[0])
