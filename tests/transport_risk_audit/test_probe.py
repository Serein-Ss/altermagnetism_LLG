import unittest
import torch
from .probe import times,oriented,scores,draft

class Tests(unittest.TestCase):
    def test_time_coverage(self):
        g=torch.Generator().manual_seed(1)
        grid=torch.stack([times(64,k,g,'cpu') for k in range(10)])
        for k in range(10):self.assertTrue(((grid[k]>=k/10)&(grid[k]<(k+1)/10)).all())
        self.assertTrue((grid[-1]>.9).all())
        with self.assertRaises(ValueError):times(64,10,g,'cpu')
    def test_decomposition(self):
        g=torch.Generator().manual_seed(2)
        p,s,c=[torch.randn(64,2,16,3,generator=g) for _ in range(3)]
        r=scores(p,s,c)
        self.assertAlmostEqual(r[1],r[0]+r[2]+r[3],places=5)
        for a,b in zip(r,scores(p.flip(0),s.flip(0),c.flip(0))):self.assertAlmostEqual(a,b,places=5)
    def test_rotation(self):
        p,s,c=[torch.randn(8,2,16,3) for _ in range(3)]
        out=oriented(p,s,c,.37,True)
        for a,b in zip(scores(p,s,c),scores(*out)):self.assertAlmostEqual(a,b,places=5)
        self.assertIs(oriented(p,s,c,.37,False)[0],p)
    def test_wrong_branch_bank(self):
        with self.assertRaises(ValueError):draft(torch.zeros(8,2,16,3),0,0,0)
