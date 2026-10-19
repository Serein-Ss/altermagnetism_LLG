import math
import unittest
import torch
from torch.nn import functional as F
from tests.transition_block_repair.model import TransitionNet
from tests.transport_audit.run import integrate
from tests.axial_symmetry_repair.symmetry import rotate
from .model import AxialProjection


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)
        self.base = TransitionNet().double().eval()
        with torch.no_grad():
            self.base.output.weight.normal_(0, .02)
        self.x = F.normalize(torch.randn(2, 2, 7, 3, dtype=torch.float64), dim=-1)
        self.a = self.x[:, 0].clone()
        self.t = self.x.new_tensor([0., .1])
        self.tau = self.x.new_tensor([.2, .7])

    def test_identity(self):
        torch.testing.assert_close(AxialProjection(self.base, 1).velocity(self.x, self.tau, self.a, self.t),
                                   self.base.velocity(self.x, self.tau, self.a, self.t), rtol=0, atol=0)

    def test_discrete_covariance_tangent_anchor(self):
        for order in (4, 8):
            net = AxialProjection(self.base, order)
            v = net.velocity(self.x, self.tau, self.a, self.t)
            angle = 2 * math.pi / order
            w = net.velocity(rotate(self.x, angle), self.tau, rotate(self.a, angle), self.t)
            torch.testing.assert_close(w, rotate(v, angle), rtol=1e-9, atol=1e-10)
            self.assertEqual(float(v[:, 0].abs().max()), 0.)
            self.assertLess(float((v * self.x).sum(-1).abs().max()), 1e-12)

    def test_joint_rotation(self):
        q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
        if torch.det(q) < 0: q[:, 0] *= -1
        net = AxialProjection(self.base, 4)
        v = net.velocity(self.x, self.tau, self.a, self.t)
        w = net.velocity(self.x @ q.T, self.tau, self.a @ q.T, self.t, frame=q.T)
        torch.testing.assert_close(w, v @ q.T, atol=1e-10, rtol=1e-9)

    def test_sampler_and_no_mutation(self):
        net = AxialProjection(self.base, 4)
        before = [p.detach().clone() for p in self.base.parameters()]
        def sample(x, a):
            return integrate(lambda y,t: net.velocity(y,t,a,self.t),x,0.,1.,4,'midpoint')
        y = sample(self.x, self.a)
        yr = sample(rotate(self.x, math.pi/2), rotate(self.a, math.pi/2))
        torch.testing.assert_close(yr, rotate(y, math.pi/2), atol=1e-9, rtol=1e-8)
        torch.testing.assert_close(y.norm(dim=-1), torch.ones_like(y[...,0]), atol=1e-12, rtol=0)
        self.assertTrue(torch.equal(y[:,0], self.a))
        for p,b in zip(self.base.parameters(),before): self.assertTrue(torch.equal(p,b))

    def test_contract_rejections(self):
        with self.assertRaises(ValueError): AxialProjection(self.base, 3)
        net = AxialProjection(self.base, 4)
        with self.assertRaises(ValueError): net.velocity(self.x,self.tau,self.a,self.t,temperature=0.)
        with self.assertRaises(ValueError): net.velocity(self.x,self.tau,self.a,self.t*2)
        self.base.periodic = (True,)
        with self.assertRaises(ValueError): AxialProjection(self.base,4)


if __name__ == '__main__': unittest.main()
