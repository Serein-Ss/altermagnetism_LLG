import unittest
import torch
from tests.conditional_moment_repair.objectives import moment_loss,sample
from tests.chain_reversal_repair.model import ReversalGauge
from .objective import loss,cross_covariance


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(2)
        a=torch.nn.functional.normalize(torch.randn(1,16,3,dtype=torch.float64),dim=-1)
        self.g=torch.nn.functional.normalize(torch.randn(64,2,16,3,dtype=torch.float64),dim=-1);self.g[:,0]=a
        self.r=torch.nn.functional.normalize(torch.randn(64,2,16,3,dtype=torch.float64),dim=-1);self.r[:,0]=a

    def test_local_exact_and_nonlocal_formula(self):
        torch.testing.assert_close(loss(self.g,self.r,False)[0],moment_loss(self.g,self.r)[0],rtol=0,atol=0)
        d=self.g[:,1]-self.g[:,0];d=d-d.mean(0)
        target=torch.stack([d[:,i].T@d[:,i+2]/63 for i in range(14)])
        torch.testing.assert_close(cross_covariance(self.g,2),target)

    def test_rotation_reversal_and_reference_no_grad(self):
        q,_=torch.linalg.qr(torch.randn(3,3,dtype=self.g.dtype))
        a=loss(self.g,self.r,True)[0]
        torch.testing.assert_close(loss(self.g@q,self.r@q,True)[0],a,rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(loss(self.g.flip(2),self.r.flip(2),True)[0],a,rtol=1e-10,atol=1e-10)
        g=self.g.clone().requires_grad_();r=self.r.clone().requires_grad_()
        gg,gr=torch.autograd.grad(loss(g,r,True)[0],(g,r),allow_unused=True)
        self.assertIsNone(gr);self.assertTrue(torch.isfinite(gg).all());self.assertGreater(float(gg.norm()),0.)

    def test_input_gradient_finite_difference(self):
        g=self.g.clone().requires_grad_();v=loss(g,self.r,True)[0];grad=torch.autograd.grad(v,g)[0]
        eps=1e-6;a=g.detach().clone();b=a.clone();a[0,1,3,0]+=eps;b[0,1,3,0]-=eps
        fd=(loss(a,self.r,True)[0]-loss(b,self.r,True)[0])/(2*eps)
        self.assertAlmostEqual(float(fd),float(grad[0,1,3,0]),places=7)

    def test_actual_sampler_gradient(self):
        net=ReversalGauge('reversal').double().eval()
        with torch.no_grad():net.output.weight.normal_(0,.05)
        initial=self.r[:8,0];z=self.g[:8].clone()
        generated=sample(net,initial,z,steps=8,use_checkpoint=True)
        value,_=loss(generated,self.r,True);value.backward()
        norm=torch.stack([p.grad.square().sum() for p in net.parameters() if p.grad is not None]).sum().sqrt()
        self.assertTrue(torch.isfinite(norm));self.assertGreater(float(norm),0.)
        self.assertTrue(torch.equal(generated[:,0],initial))
        self.assertLess(float((generated.norm(dim=-1)-1).abs().max()),1e-12)


if __name__=='__main__':unittest.main()
