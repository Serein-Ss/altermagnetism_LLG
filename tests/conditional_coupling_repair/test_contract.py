import unittest
import numpy as np
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.axial_symmetry_repair.symmetry import rotate
from .coupling import match,schedule


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.y=F.normalize(torch.randn(32,2,7,3,dtype=torch.float64),dim=-1)
        for i in (0,16):self.y[i:i+16,0]=self.y[i,0].clone()
        self.z=F.normalize(torch.randn_like(self.y),dim=-1);self.z[:,0]=self.y[:,0]

    def test_permutation_cost_and_no_mutation(self):
        old=self.y.clone();out,record=match(self.z,self.y)
        p=record['permutation']
        self.assertEqual(sorted(p[:16]),list(range(16)))
        self.assertEqual(sorted(p[16:]),list(range(16,32)))
        self.assertTrue(torch.equal(self.y,old));self.assertTrue(torch.equal(out,self.y[p]))
        self.assertLessEqual(record['assignment_cost'],record['random_cost']+1e-12)
        self.assertTrue(torch.equal(out[:,0],self.y[:,0]))

    def test_rotation_invariance(self):
        out,a=match(self.z,self.y)
        r,b=match(rotate(self.z,.79),rotate(self.y,.79))
        self.assertEqual(a['permutation'],b['permutation'])
        torch.testing.assert_close(r,rotate(out,.79))

    def test_reject_mixed_conditions(self):
        y=self.y.clone();y[1,0]=y[16,0]
        with self.assertRaises(ValueError):match(self.z,y)
        z=self.z.clone();z[1,0]=y[1,0]
        with self.assertRaises(ValueError):match(z,y)

    def test_zero_cost_permuted_targets(self):
        z=self.y.clone();z[:,1]=self.y[torch.arange(32).reshape(2,16).flip(1).flatten(),1]
        out,record=match(z,self.y)
        torch.testing.assert_close(out,z)
        self.assertLess(record['assignment_cost'],1e-14)

    def test_schedule_exposure(self):
        a,b=schedule(12,32);aa,bb=schedule(12,32)
        np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb)
        np.testing.assert_array_equal(np.bincount(a.flatten()),np.full(48,32))
        for x,y in zip(a,b):
            self.assertEqual(len(set(x)),3)
            for i in (0,16,32):self.assertEqual(len(set(y[i:i+16])),16)
        with self.assertRaises(ValueError):schedule(12,17)

    def test_spherical_interpolation_contract(self):
        y,_=match(self.z,self.y)
        x,v,_=path_interpolate(self.z,y,torch.linspace(.1,.9,32,dtype=torch.float64),torch.Generator().manual_seed(1))
        torch.testing.assert_close(x.norm(dim=-1),torch.ones_like(x[...,0]))
        self.assertTrue(torch.equal(x[:,0],y[:,0]))
        self.assertLess(float((x*v).sum(-1).abs().max()),1e-10)


if __name__=='__main__':unittest.main()
