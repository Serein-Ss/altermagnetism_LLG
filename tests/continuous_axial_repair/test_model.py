import unittest
import torch
from torch.nn import functional as F
from tests.transition_block_repair.model import TransitionNet
from tests.transport_audit.run import integrate
from tests.temporal_source_repair.source import draw
from tests.axial_symmetry_repair.symmetry import rotate
from tests.bauer_p2_20260923.common import energy
from .model import AxialGauge,phases


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(8);self.base=TransitionNet().double().eval()
        with torch.no_grad():
            self.base.output.weight.normal_(0,.1)
            for b in self.base.blocks:b.modulation[-1].bias.normal_(0,.2)
        self.a=F.normalize(torch.randn(2,25,3,dtype=torch.float64),dim=-1)
        self.x=F.normalize(torch.randn(2,2,25,3,dtype=torch.float64),dim=-1);self.x[:,0]=self.a
        self.tau=self.x.new_tensor([.3,.7]);self.times=self.x.new_tensor([0.,.1])

    def net(self,arm):
        net=AxialGauge(arm).double().eval();s=self.base.state_dict();s['_extra_state']=net.get_extra_state();net.load_state_dict(s);return net

    def test_control_identity(self):
        a=self.net('control').velocity(self.x,self.tau,self.a,self.times)
        b=self.base.velocity(self.x,self.tau,self.a,self.times)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_continuous_axial_and_joint_covariance(self):
        net=self.net('gauge2');v=net.velocity(self.x,self.tau,self.a,self.times)
        for angles in (self.x.new_tensor([.371,1.123]),self.x.new_tensor([2.021,-.73])):
            vr=net.velocity(rotate(self.x,angles),self.tau,rotate(self.a,angles),self.times)
            torch.testing.assert_close(vr,rotate(v,angles),rtol=1e-9,atol=1e-9)
        q,_=torch.linalg.qr(torch.randn(3,3,dtype=self.x.dtype));q[:,0]*=torch.linalg.det(q)
        vr=net.velocity(self.x@q,self.tau,self.a@q,self.times,frame=q)
        torch.testing.assert_close(vr,v@q,rtol=1e-9,atol=1e-9)
        self.assertLess(float((v*self.x).sum(-1).abs().max()),1e-10);self.assertEqual(float(v[:,0].abs().max()),0.)

    def test_matched_latent_sampling_covariance_and_norm(self):
        net=self.net('gauge2');z=draw(self.a,2,torch.Generator().manual_seed(6),0.)
        angle=self.x.new_tensor([.333,1.157]);ar=rotate(self.a,angle)
        f=lambda x,t:net.velocity(x,t,self.a,self.times)
        fr=lambda x,t:net.velocity(x,t,ar,self.times)
        a=integrate(f,z,0.,1.,8,'midpoint');b=integrate(fr,rotate(z,angle),0.,1.,8,'midpoint')
        torch.testing.assert_close(b,rotate(a,angle),rtol=1e-8,atol=1e-8)
        self.assertTrue(torch.equal(a[:,0],self.a));self.assertLess(float((a.norm(dim=-1)-1).abs().max()),1e-12)

    def test_zero_weight_and_unsupported_poles(self):
        a=self.a.clone();a[:,0]=a.new_tensor([0.,0.,1.]);x=self.x.clone();x[:,0]=a
        phase,w=phases(a);self.assertTrue(torch.isfinite(phase).all());self.assertEqual(float(w[:,0].abs().max()),0.)
        net=self.net('gauge2');v=net.velocity(x,self.tau,a,self.times);self.assertTrue(torch.isfinite(v).all())
        angle=a.new_tensor([.431,2.38]);vr=net.velocity(rotate(x,angle),self.tau,rotate(a,angle),self.times)
        torch.testing.assert_close(vr,rotate(v,angle),rtol=1e-9,atol=1e-9)
        a[:,1]=a.new_tensor([0.,0.,-1.]);x[:,0]=a
        with self.assertRaisesRegex(ValueError,'unsupported initial'):net.velocity(x,self.tau,a,self.times)

    def test_finite_gradient_and_phase_continuity(self):
        net=self.net('gauge2');x=self.x.clone().requires_grad_();v=net.velocity(x,self.tau,self.a,self.times)
        gradient=torch.autograd.grad(v.square().sum(),x)[0];self.assertTrue(torch.isfinite(gradient).all())
        eps=1e-6;xp=x.detach().clone();xm=x.detach().clone();xp[0,1,3,0]+=eps;xm[0,1,3,0]-=eps
        fd=(net.velocity(xp,self.tau,self.a,self.times).square().sum()-net.velocity(xm,self.tau,self.a,self.times).square().sum())/(2*eps)
        self.assertAlmostEqual(float(fd),float(gradient[0,1,3,0]),places=6)
        a=self.a.clone();a[:,0]=a.new_tensor([.8,0.,.6]);a[:,1]=a.new_tensor([0.,.8,.6]);norms=[]
        for eps in (1e-3,1e-5):
            plus=a.clone();minus=a.clone();plus[:,1,1]+=eps;minus[:,1,1]-=eps
            plus=F.normalize(plus,dim=-1);minus=F.normalize(minus,dim=-1);xx=self.x.clone();xx[:,0]=plus
            vp=net.velocity(xx,self.tau,plus,self.times);xx[:,0]=minus;vm=net.velocity(xx,self.tau,minus,self.times);norms.append(float((vp-vm).norm()))
        self.assertLess(norms[1],norms[0]*.02)

    def test_metadata_and_physics_symmetry(self):
        a=self.net('gauge2');b=self.net('gauge2');b.load_state_dict(a.state_dict())
        with self.assertRaises(ValueError):self.net('control').load_state_dict(a.state_dict())
        with self.assertRaises(ValueError):self.base.load_state_dict(a.state_dict())
        angles=self.a.new_tensor([.313,1.999]);torch.testing.assert_close(energy(rotate(self.a,angles)),energy(self.a),rtol=1e-12,atol=1e-12)
        with self.assertRaises(ValueError):a.velocity(self.x,self.tau,self.a,self.times,temperature=12.5)


if __name__=='__main__':unittest.main()
