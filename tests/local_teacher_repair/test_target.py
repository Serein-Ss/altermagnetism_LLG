import unittest
import torch
from torch.nn import functional as F
from tests.temporal_source_repair.source import draw
from tests.conditional_teacher_repair.target import labels as old_labels
from tests.empirical_transport_repair.oracle import conditional
from .target import patch_conditional,labels


class PatchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);gen=torch.Generator().manual_seed(543)
        a=F.normalize(torch.randn(1,7,3,dtype=torch.float64,generator=gen),dim=-1)
        end=F.normalize(a+torch.randn(64,7,3,dtype=torch.float64,generator=gen)*.3,dim=-1)
        self.bank=torch.stack((a.expand(64,-1,-1),end),1)
        self.y=self.bank[:8];self.z=draw(self.y[:,0],2,gen,0.);self.tau=torch.linspace(0,.99,8)

    def test_whole_arm_exact_existing_labels_and_rng(self):
        a=torch.Generator().manual_seed(321);b=torch.Generator().manual_seed(321)
        old=old_labels(self.z,self.y,self.tau,self.bank,a);new=labels(self.z,self.y,self.tau,self.bank,b)
        for x,y in zip(old[:3],new[:3]):self.assertTrue(torch.equal(x,y))
        self.assertTrue(torch.equal(a.get_state(),b.get_state()))

    def test_full_radius_matches_whole(self):
        # Use tau zero so support is independent of interpolation roundoff.
        xx=self.z;tt=torch.zeros(8,dtype=torch.float64)
        expected,weights,bad=conditional(xx,tt,self.bank,self.y[:,0]);got,ess=patch_conditional(xx,tt,self.bank,self.y[:,0],7)
        self.assertFalse(bad.any());self.assertTrue(torch.allclose(got,expected,atol=1e-12,rtol=0))
        self.assertTrue(torch.allclose(ess,(1/weights.square().sum(-1))[:,None].expand_as(ess),atol=1e-12))

    def test_tangent_anchor_joint_rotation_and_reversal(self):
        tau=torch.zeros(8,dtype=torch.float64);v,ess=patch_conditional(self.z,tau,self.bank,self.y[:,0],2)
        self.assertLess(float((v*self.z).sum(-1).abs().max()),1e-12)
        self.assertTrue(torch.equal(v[:,0],torch.zeros_like(v[:,0])))
        q,_=torch.linalg.qr(torch.tensor([[1.,2.,3.],[3.,1.,2.],[2.,3.,1.]],dtype=torch.float64));q=q*torch.linalg.det(q)
        rotated,_=patch_conditional(self.z@q,tau,self.bank@q,self.y[:,0]@q,2)
        reversed_,re=patch_conditional(self.z.flip(2),tau,self.bank.flip(2),self.y[:,0].flip(1),2)
        self.assertTrue(torch.allclose(rotated,v@q,atol=1e-11,rtol=0))
        self.assertTrue(torch.allclose(reversed_.flip(2),v,atol=1e-11,rtol=0))
        self.assertTrue(torch.allclose(re.flip(1),ess,atol=1e-11,rtol=0))

    def test_boundary_no_wrap_and_radius_zero(self):
        tau=torch.zeros(8,dtype=torch.float64);v,_=patch_conditional(self.z,tau,self.bank,self.y[:,0],2)
        direct,_,_=conditional(self.z[:,:,:3],tau,self.bank[:,:,:3],self.y[:,0,:3])
        self.assertTrue(torch.allclose(v[:,:,0],direct[:,:,0],atol=1e-12,rtol=0))
        zero,_=patch_conditional(self.z,tau,self.bank,self.y[:,0],0)
        direct,_,_=conditional(self.z[:,:,3:4],tau,self.bank[:,:,3:4],self.y[:,0,3:4])
        self.assertTrue(torch.allclose(zero[:,:,3],direct[:,:,0],atol=1e-12,rtol=0))

    def test_late_tau_and_pairing(self):
        a=torch.Generator().manual_seed(321);b=torch.Generator().manual_seed(321)
        whole=labels(self.z,self.y,self.tau,self.bank,a);local=labels(self.z,self.y,self.tau,self.bank,b,True)
        self.assertTrue(torch.equal(whole[0],local[0]));self.assertTrue(torch.equal(a.get_state(),b.get_state()))
        self.assertTrue(torch.equal(local[2][-1],local[1][-1]));self.assertTrue(torch.equal(whole[2][-1],whole[1][-1]))
        self.assertGreater(local[3]['target_difference'],0)
        with self.assertRaises(ValueError):patch_conditional(self.z,self.tau,self.bank,self.y[:,0],-1)


if __name__=='__main__':unittest.main()
