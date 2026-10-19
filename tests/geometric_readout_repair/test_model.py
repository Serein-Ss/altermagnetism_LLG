import unittest
import torch
from torch.nn import functional as F
from .model import basis,initialize,GeometricNet
from tests.transition_block_repair.model import TransitionNet

class Tests(unittest.TestCase):
    def inputs(self):
        g=torch.Generator().manual_seed(31)
        a=F.normalize(torch.randn(2,16,3,generator=g),dim=-1)
        x=F.normalize(torch.randn(2,2,16,3,generator=g),dim=-1);x[:,0]=a
        return x,a,torch.tensor([.2,.8]),torch.tensor([0.,.1])
    def test_zero_head_identity(self):
        x,a,t,time=self.inputs();base=TransitionNet().eval()
        net=initialize({'model':base.state_dict()},True).eval()
        with torch.no_grad():self.assertTrue(torch.equal(base.velocity(x,t,a,time),net.velocity(x,t,a,time)))
    def test_covariance_tangent_anchor(self):
        x,a,t,time=self.inputs();net=GeometricNet().eval()
        torch.nn.init.normal_(net.geometric_head[-1].weight,std=.1)
        q,_=torch.linalg.qr(torch.randn(3,3));q[:,0]*=torch.linalg.det(q)
        with torch.no_grad():
            v=net.velocity(x,t,a,time)
            w=net.velocity(x@q.T,t,a@q.T,time,frame=torch.eye(3)@q.T)
        self.assertTrue(torch.allclose(w,v@q.T,atol=2e-6,rtol=2e-5))
        self.assertLess(float((v*x).sum(-1).abs().max()),2e-6)
        self.assertTrue(torch.equal(v[:,0],torch.zeros_like(v[:,0])))
    def test_coincident_and_antipode(self):
        a=torch.tensor([[[0.,0.,1.]]]);x=a[:,None].expand(1,2,1,3)
        log,cross=basis(x,a)
        self.assertTrue(torch.equal(log,torch.zeros_like(log)))
        self.assertTrue(torch.equal(cross,torch.zeros_like(cross)))
        with self.assertRaises(ValueError):basis(-x,a)
    def test_contract_rejects_old(self):
        with self.assertRaises(ValueError):GeometricNet().load_state_dict(TransitionNet().state_dict())
