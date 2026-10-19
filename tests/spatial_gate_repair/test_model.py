import unittest
import torch
from torch.nn import functional as F
from tests.transition_block_repair.model import TransitionNet,rollout
from .model import GatedTransition,remote_probe


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(37)
        self.base=TransitionNet().double().eval()
        with torch.no_grad():
            self.base.output.weight.normal_(0,.1)
            for b in self.base.blocks:b.modulation[-1].bias.normal_(0,.2)
        self.a=F.normalize(torch.randn(2,25,3,dtype=torch.float64),dim=-1)
        self.x=F.normalize(torch.randn(2,2,25,3,dtype=torch.float64),dim=-1);self.x[:,0]=self.a

    def net(self,scale):
        net=GatedTransition(scale).double().eval();state=self.base.state_dict();state['_extra_state']=net.get_extra_state();net.load_state_dict(state);return net

    def call(self,net):return net.velocity(self.x,self.x.new_full((2,),.4),self.a,self.x.new_tensor([0.,.1]))

    def test_identity_and_local_equivalence(self):
        torch.testing.assert_close(self.call(self.net(1.)),self.call(self.base),rtol=0,atol=0)
        for b in self.base.blocks:b.arm='local'
        torch.testing.assert_close(self.call(self.net(0.)),self.call(self.base),rtol=0,atol=0)

    def test_remote_and_gradient(self):
        local=self.net(0.);global_net=self.net(1.)
        self.assertLess(max(r['max_change'] for r in remote_probe(local)),1e-12)
        self.assertGreater(max(r['max_change'] for r in remote_probe(global_net)),1e-6)
        self.call(local).square().sum().backward()
        self.assertEqual(float(local.blocks[0].coarse.weight.grad.abs().max()),0.)

    def test_joint_covariance_tangent_anchor(self):
        for scale in (0.,.25,1.):
            net=self.net(scale);v=self.call(net)
            q,_=torch.linalg.qr(torch.randn(3,3,dtype=self.x.dtype));q[:,0]*=torch.linalg.det(q)
            vr=net.velocity(self.x@q,self.x.new_full((2,),.4),self.a@q,self.x.new_tensor([0.,.1]),frame=q)
            torch.testing.assert_close(vr,v@q,rtol=1e-9,atol=1e-9)
            self.assertLess(float((v*self.x).sum(-1).abs().max()),1e-10)
            self.assertEqual(float(v[:,0].abs().max()),0.)

    def test_package_and_reload(self):
        a=self.net(.25);b=self.net(.25);b.load_state_dict(a.state_dict());torch.testing.assert_close(self.call(a),self.call(b))
        with self.assertRaises(ValueError):self.net(0.).load_state_dict(a.state_dict())
        with self.assertRaises(ValueError):TransitionNet().load_state_dict(a.state_dict())

    def test_spherical_rollout(self):
        for scale in (0.,.25,1.):
            y=rollout(self.net(scale),self.a,torch.Generator().manual_seed(77),2,2)
            self.assertTrue(torch.equal(y[:,0],self.a));self.assertLess(float((y.norm(dim=-1)-1).abs().max()),1e-12)


if __name__=='__main__':unittest.main()
