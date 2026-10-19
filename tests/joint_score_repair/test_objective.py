import pytest
import torch
from torch.nn import functional as F
from .objective import joint_loss,evaluate_distance


def paths(end):
    a=torch.zeros_like(end);a[...,2]=1
    return torch.stack((a,end),1)


def test_joint_dependence_not_just_local_marginals():
    up=torch.tensor([0.,0.,1.],dtype=torch.float64);down=-up
    r=paths(torch.stack((torch.stack((up,up)),torch.stack((down,down)))))
    wrong=paths(torch.stack((torch.stack((up,down)),torch.stack((down,up)))))
    exact=torch.cat((r,r));different=torch.cat((wrong,wrong))
    assert torch.equal(r[:,1].mean(0),wrong[:,1].mean(0))
    assert joint_loss(exact,r)[0]==pytest.approx(0.,abs=1e-12)
    assert joint_loss(different,r)[0]>.1


def test_rotation_gradient_and_coincident_samples():
    torch.manual_seed(6);x=paths(F.normalize(torch.randn(8,4,3,dtype=torch.float64),dim=-1)).requires_grad_()
    r=paths(F.normalize(torch.randn(12,4,3,dtype=torch.float64),dim=-1))
    loss,_=joint_loss(x,r);gradient=torch.autograd.grad(loss,x)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    q,_=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype));q[:,0]*=torch.linalg.det(q)
    assert float(joint_loss(x@q,r@q)[0].detach())==pytest.approx(float(loss.detach()),abs=1e-12)
    collapsed=paths(torch.tensor([0.,0.,1.],dtype=x.dtype).expand(8,4,3)).clone().requires_grad_()
    assert torch.isfinite(torch.autograd.grad(joint_loss(collapsed,collapsed.detach())[0],collapsed)[0]).all()
    eps=1e-6;a=x.detach().clone();b=a.clone();a[0,1,0,0]+=eps;b[0,1,0,0]-=eps
    fd=(joint_loss(a,r)[0]-joint_loss(b,r)[0])/(2*eps)
    assert float(gradient[0,1,0,0])==pytest.approx(float(fd),rel=1e-5,abs=1e-7)


def test_reject_anchor_and_evaluator_can_be_negative():
    r=paths(torch.tensor([0.,0.,1.]).expand(4,2,3)).clone();g=r.clone();g[:,0,0,0]=.1
    with pytest.raises(ValueError,match='anchor'):joint_loss(g,r)
    assert evaluate_distance(r,r)==pytest.approx(0.)
