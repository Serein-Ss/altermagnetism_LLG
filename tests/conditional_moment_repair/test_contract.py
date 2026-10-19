import unittest
import numpy as np
import torch
from torch.nn import functional as F
from .objectives import moments,moment_loss,sample
from tests.transport_audit.run import integrate


class Field(torch.nn.Module):
    def __init__(self):
        super().__init__();self.p=torch.nn.Parameter(torch.tensor(.3,dtype=torch.float64))
    def velocity(self,x,tau,initial,times):
        axis=x.new_tensor([.2,.3,.7]);v=self.p*(1+tau[:,None,None,None])*torch.cross(x,axis.expand_as(x),dim=-1)
        return torch.cat((torch.zeros_like(v[:,:1]),v[:,1:]),1)


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.a=F.normalize(torch.randn(1,5,3,dtype=torch.float64),dim=-1).expand(16,-1,-1)
        self.z=torch.stack((self.a,F.normalize(torch.randn_like(self.a),dim=-1)),1)
    def test_covariance(self):
        m,c=moments(self.z);d=(self.z[:,1]-self.z[:,0]).numpy()
        np.testing.assert_allclose(m.numpy(),d.mean(0),atol=1e-14)
        for i in range(5):np.testing.assert_allclose(c[i].numpy(),np.cov(d[:,i],rowvar=False),atol=1e-14)
    def test_sampler_and_checkpoint_gradient(self):
        net=Field();a=sample(net,self.a,self.z,24,False);a[:,1,:,0].sum().backward();grad=net.p.grad.clone()
        net.p.grad=None;b=sample(net,self.a,self.z,24,True);b[:,1,:,0].sum().backward()
        torch.testing.assert_close(a,b);torch.testing.assert_close(grad,net.p.grad)
        expected=integrate(lambda x,t:net.velocity(x,t,self.a,x.new_tensor([0.,.1])),self.z,0,1,24,'midpoint')
        torch.testing.assert_close(a,expected,atol=1e-12,rtol=1e-12)
        self.assertTrue(torch.equal(b[:,0],self.a));self.assertLess(float((b.norm(dim=-1)-1).abs().max()),1e-12)
    def test_gradient_finite_difference(self):
        net=Field();loss,_=moment_loss(sample(net,self.a,self.z,8),self.z);loss.backward();grad=net.p.grad.item();values=[]
        for p in (.3-1e-5,.3+1e-5):
            with torch.no_grad():net.p.fill_(p)
            values.append(moment_loss(sample(net,self.a,self.z,8,False),self.z)[0].item())
        self.assertAlmostEqual(grad,(values[1]-values[0])/2e-5,places=6)
    def test_rotation_invariance(self):
        q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64));q[:,0]*=torch.linalg.det(q)
        g=sample(Field(),self.a,self.z,8,False)
        torch.testing.assert_close(moment_loss(g,self.z)[0],moment_loss(g@q,self.z@q)[0])
    def test_different_initial_rejected(self):
        z=self.z.clone();z[1,0]*=-1
        with self.assertRaises(ValueError):moment_loss(z,self.z)
    def test_exact_replicated_ensemble_zero(self):
        # Each half has the identical empirical moments of the target.
        loss,_=moment_loss(torch.cat((self.z,self.z)),self.z)
        self.assertEqual(float(loss),0.)


if __name__=='__main__':unittest.main()
