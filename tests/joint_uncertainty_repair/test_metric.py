import itertools
import unittest
import numpy as np
import torch
from tests.joint_score_repair.objective import evaluate_distance
from .metric import matrices,values,summarize,distance_with_uncertainty


def paths(end):
    a=np.zeros_like(end);a[...,2]=1;return np.stack((a,end),1)


class Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_exact_bootstrap_center(self):
        g=paths(np.array([[[0.,0.,1.]],[[0.,0.,-1.]]]))
        r=paths(np.array([[[1.,0.,0.]],[[0.,1.,0.]]]))
        gg,gr,rr=matrices(g,r)
        counts=np.array([np.bincount(i,minlength=2) for i in itertools.product(range(2),repeat=2)])
        cg=np.repeat(counts,4,axis=0);cr=np.tile(counts,(4,1))
        boot=values(gg,gr,rr,cg,cr)
        self.assertAlmostEqual(boot.mean(),2*gr.mean()-gg.mean()-rr.mean(),places=12)
        self.assertGreater(boot.mean()-values(gg,gr,rr,np.ones((1,2)),np.ones((1,2)))[0],0.)

    def test_point_estimate_unchanged_and_duplicate_counts(self):
        rng=np.random.default_rng(6)
        g=paths(rng.normal(size=(6,4,3)));r=paths(rng.normal(size=(7,4,3)))
        result=distance_with_uncertainty(g,r,repeats=99,permutations=99)
        self.assertAlmostEqual(result['estimate'],evaluate_distance(torch.tensor(g),torch.tensor(r)),places=12)
        i=np.array([0,0,2,3,3,5]);j=np.array([1,1,1,3,4,5,6])
        v=values(*matrices(g,r),np.bincount(i,minlength=6)[None],np.bincount(j,minlength=7)[None])[0]
        self.assertAlmostEqual(v,evaluate_distance(torch.tensor(g[i]),torch.tensor(r[j])),places=12)
        expected=matrices(g,r)[0].mean()/5+matrices(g,r)[2].mean()/6
        self.assertAlmostEqual(result['exact_bootstrap_center_shift'],expected,places=12)

    def test_degenerate_and_rotation(self):
        x=paths(np.array([[[0.,0.,1.]]]*8))
        a=distance_with_uncertainty(x,x,repeats=99,permutations=99)
        self.assertEqual(a['estimate'],0);self.assertEqual(a['permutation_equality_p'],1.)
        self.assertEqual(a['centered_basic_approx95'],[0.,0.])
        rng=np.random.default_rng(19);g=paths(rng.normal(size=(8,4,3)));r=paths(rng.normal(size=(9,4,3)))
        q,_=np.linalg.qr(rng.normal(size=(3,3)))
        a=distance_with_uncertainty(g,r,repeats=99,permutations=99);b=distance_with_uncertainty(g@q,r@q,repeats=99,permutations=99)
        np.testing.assert_allclose(a['centered_basic_approx95'],b['centered_basic_approx95'],atol=1e-12)
        self.assertEqual(a['permutation_equality_p'],b['permutation_equality_p'])

    def test_reproducibility_and_shape(self):
        x=paths(np.array([[[0.,0.,1.]],[[0.,0.,-1.]]]*4))
        a=distance_with_uncertainty(x,x,repeats=99,permutations=99)
        self.assertEqual(a,distance_with_uncertainty(x,x,repeats=99,permutations=99))
        with self.assertRaises(ValueError):distance_with_uncertainty(x,x[:,:1],repeats=99,permutations=99)


if __name__=='__main__':unittest.main()
