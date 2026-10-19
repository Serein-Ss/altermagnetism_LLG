import unittest
import torch
from tests.continuous_axial_repair.test_model import Tests as Previous
from tests.continuous_axial_repair.model import AxialGauge
from tests.bauer_p2_20260923.common import energy
from tests.transport_audit.run import integrate
from .model import ReversalGauge,gauge_probe


class Tests(Previous):
    def net(self,arm):
        if arm in ('control','gauge2'):return super().net(arm)
        net=ReversalGauge(arm).double().eval();state=self.base.state_dict()
        state['_extra_state']=net.get_extra_state();net.load_state_dict(state);return net

    def test_reversal_covariance_and_sample(self):
        net=self.net('reversal')
        for size in (16,25,32):
            a=torch.nn.functional.normalize(torch.randn(2,size,3,dtype=torch.float64),dim=-1)
            x=torch.nn.functional.normalize(torch.randn(2,2,size,3,dtype=torch.float64),dim=-1);x[:,0]=a
            v=net.velocity(x,self.tau,a,self.times)
            vr=net.velocity(x.flip(2),self.tau,a.flip(1),self.times).flip(2)
            torch.testing.assert_close(v,vr,rtol=1e-10,atol=1e-10)
            self.assertLess(float((v*x).sum(-1).abs().max()),1e-10)
            self.assertEqual(float(v[:,0].abs().max()),0.)
            out=integrate(lambda y,t:net.velocity(y,t,a,self.times),x,0.,1.,4,'midpoint')
            rev=integrate(lambda y,t:net.velocity(y,t,a.flip(1),self.times),x.flip(2),0.,1.,4,'midpoint')
            torch.testing.assert_close(out,rev.flip(2),rtol=1e-9,atol=1e-9)
            self.assertTrue(torch.equal(out[:,0],a))
            self.assertLess(float((out.norm(dim=-1)-1).abs().max()),1e-12)

    def test_control_identity_with_gauge2(self):
        control=ReversalGauge('control').double().eval();base=self.net('gauge2')
        state=base.state_dict();state['_extra_state']=control.get_extra_state();control.load_state_dict(state)
        torch.testing.assert_close(control.velocity(self.x,self.tau,self.a,self.times),
            base.velocity(self.x,self.tau,self.a,self.times),rtol=0,atol=0)
        with self.assertRaises(ValueError):control.load_state_dict(base.state_dict())
        with self.assertRaises(ValueError):ReversalGauge('reversal').load_state_dict(control.state_dict())

    def test_projected_axial_joint_and_gradient(self):
        from tests.axial_symmetry_repair.symmetry import rotate
        net=self.net('reversal');v=net.velocity(self.x,self.tau,self.a,self.times)
        angles=self.a.new_tensor([.371,1.127])
        torch.testing.assert_close(net.velocity(rotate(self.x,angles),self.tau,rotate(self.a,angles),self.times),rotate(v,angles),rtol=1e-9,atol=1e-9)
        q,_=torch.linalg.qr(torch.randn(3,3,dtype=self.x.dtype));q[:,0]*=torch.linalg.det(q)
        torch.testing.assert_close(net.velocity(self.x@q,self.tau,self.a@q,self.times,frame=q),v@q,rtol=1e-9,atol=1e-9)
        x=self.x.clone().requires_grad_();loss=net.velocity(x,self.tau,self.a,self.times).square().sum()
        grad=torch.autograd.grad(loss,x)[0];self.assertTrue(torch.isfinite(grad).all())
        eps=1e-6;xp=x.detach().clone();xm=x.detach().clone();xp[0,1,3,0]+=eps;xm[0,1,3,0]-=eps
        fd=(net.velocity(xp,self.tau,self.a,self.times).square().sum()-net.velocity(xm,self.tau,self.a,self.times).square().sum())/(2*eps)
        self.assertAlmostEqual(float(fd),float(grad[0,1,3,0]),places=6)

    def test_hamiltonian_and_gradient_permutation(self):
        a=self.a.clone().requires_grad_();rev=a.detach().flip(1).requires_grad_()
        torch.testing.assert_close(energy(a),energy(rev),rtol=1e-12,atol=1e-12)
        ga=torch.autograd.grad(energy(a).sum(),a)[0];gr=torch.autograd.grad(energy(rev).sum(),rev)[0]
        torch.testing.assert_close(ga,gr.flip(1),rtol=1e-12,atol=1e-12)

    def test_right_anchor_support_rejection(self):
        a=self.a.clone();a[:,-2:]=a.new_tensor([0.,0.,1.]);x=self.x.clone();x[:,0]=a
        self.assertTrue(torch.isfinite(self.net('gauge2').velocity(x,self.tau,a,self.times)).all())
        with self.assertRaisesRegex(ValueError,'unsupported initial'):self.net('reversal').velocity(x,self.tau,a,self.times)

    def test_synthetic_probe_obeys_two_end_contract(self):
        probe=gauge_probe(self.net('reversal'))
        self.assertEqual(len(probe['smooth_equal_weight_probe']),4)
        self.assertLess(probe['smooth_equal_weight_probe'][-1]['velocity_difference'],
            probe['smooth_equal_weight_probe'][0]['velocity_difference']*.002)


if __name__=='__main__':unittest.main()
