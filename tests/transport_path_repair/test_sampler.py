import unittest
import numpy as np
import torch
from torch.nn import functional as F
from tests.transport_audit.run import integrate as old_integrate
from .run import grid,integrate,CUTS


class SamplerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37);self.x=F.normalize(torch.randn(3,11,7,3,dtype=torch.float64),dim=-1)
        self.axis=F.normalize(torch.tensor([1.,2.,3.],dtype=torch.float64),dim=0)

    def field(self,x,t):return torch.linalg.cross(self.axis.expand_as(x),x)*(1+t[:,None,None,None])

    def test_uniform_regression_and_trace(self):
        y,tr=integrate(self.field,self.x,grid(128),True)
        old=old_integrate(self.field,self.x,0,1,128,'midpoint')
        self.assertTrue(torch.equal(y,old));self.assertEqual(set(tr),set(CUTS))

    def test_structure_covariance_convergence(self):
        q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64));q[:,0]*=torch.linalg.det(q)
        for late in (False,True):
            y,_=integrate(self.field,self.x,grid(128,late))
            self.assertLess(float(abs(y.norm(dim=-1)-1).max()),1e-12)
            self.assertTrue(torch.equal(y[:,0],self.x[:,0]))
            axis=self.axis@q
            rot=lambda x,t:torch.linalg.cross(axis.expand_as(x),x)*(1+t[:,None,None,None])
            yr,_=integrate(rot,self.x@q,grid(128,late))
            self.assertLess(float(abs(yr-y@q).max()),1e-12)
            # Integrated angle = integral_0^1(1+t)dt = 1.5 rad.
            exact=self.x*np.cos(1.5)+torch.linalg.cross(self.axis.expand_as(self.x),self.x)*np.sin(1.5)+self.axis*(self.x*self.axis).sum(-1,keepdim=True)*(1-np.cos(1.5))
            coarse,_=integrate(self.field,self.x,grid(16,late))
            self.assertLess(float(abs(y[:,1:]-exact[:,1:]).max()),float(abs(coarse[:,1:]-exact[:,1:]).max())/10)

    def test_restart_and_invalid(self):
        y,_=integrate(self.field,self.x,grid(128))
        z,_=integrate(self.field,self.x,grid(64)*.5)
        z,_=integrate(self.field,z,grid(64,start=.5))
        self.assertTrue(torch.equal(y,z))
        for edges in ([0,0,1],[0,float('nan'),1],[-1,1],[0,1.1]):
            with self.assertRaises(ValueError):integrate(self.field,self.x,edges)


if __name__=='__main__':unittest.main()
