import unittest
import numpy as np
import torch
from torch.nn import functional as F
from tests.bauer_p2_20260923.common import energy
from tests.bauer_fig2_10ns_20260917.kernel import drift,heun_step
from .symmetry import rotate


class Tests(unittest.TestCase):
    def test_rotation_energy_drift_and_noise_step(self):
        torch.manual_seed(1);s=F.normalize(torch.randn(3,16,3,dtype=torch.float64),dim=-1)
        angle=torch.tensor([.3,1.2,2.8],dtype=torch.float64);r=rotate(s,angle)
        torch.testing.assert_close(energy(s),energy(r),atol=1e-13,rtol=1e-13)
        torch.testing.assert_close(rotate(r,-angle),s,atol=1e-13,rtol=1e-13)
        for i in range(3):
            a=drift(s[i].numpy(),.1,.1);b=drift(r[i].numpy(),.1,.1)
            np.testing.assert_allclose(b,rotate(torch.tensor(a),angle[i]).numpy(),atol=1e-13)
            dw=torch.randn_like(s[i])*.01
            a,_=heun_step(s[i].numpy(),dw.numpy(),.001,.1,.1,.11)
            b,_=heun_step(r[i].numpy(),rotate(dw,angle[i]).numpy(),.001,.1,.1,.11)
            np.testing.assert_allclose(b,rotate(torch.tensor(a),angle[i]).numpy(),atol=1e-13)

    def test_path_anchor_tangency_and_no_cross_path_rotation(self):
        torch.manual_seed(2);x=F.normalize(torch.randn(5,2,7,3),dim=-1);v=torch.randn_like(x);v-=x*(x*v).sum(-1,keepdim=True)
        angle=torch.arange(5)*.4;r=rotate(x,angle);w=rotate(v,angle)
        torch.testing.assert_close(r.norm(dim=-1),x.norm(dim=-1))
        torch.testing.assert_close(r[:,0],rotate(x[:,0],angle))
        torch.testing.assert_close((r*w).sum(-1),(x*v).sum(-1),atol=1e-6,rtol=1e-5)
        for i in range(5):torch.testing.assert_close(r[i],rotate(x[i],angle[i]))

    def test_general_spin_rotation_not_assumed(self):
        z=torch.tensor([[[0.,0.,1.]]*16]);x=torch.tensor([[[1.,0.,0.]]*16])
        self.assertGreater(float(abs(energy(z)-energy(x))),.09)

    def test_augmentation_rng_and_anchor_contract(self):
        from .run import augment
        x=F.normalize(torch.randn(4,2,16,3),dim=-1);v=torch.randn_like(x);v[:,0]=0
        g1=torch.Generator().manual_seed(12);g2=torch.Generator().manual_seed(12)
        control=augment(x,v,x[:,0],g1,False);r,w,a=augment(x,v,x[:,0],g2,True)
        self.assertTrue(torch.equal(g1.get_state(),g2.get_state()))
        self.assertTrue(torch.equal(control[0],x));self.assertTrue(torch.equal(r[:,0],a))
        self.assertTrue(torch.equal(w[:,0],v[:,0]))
        torch.testing.assert_close(energy(r),energy(x))


if __name__=='__main__':unittest.main()
