import unittest
import torch
from .gradients import merge


class Tests(unittest.TestCase):
    def test_sum(self):
        a=torch.tensor([1.,2.]);b=torch.tensor([-3.,4.])
        c,_=merge([a],[b],'sum');torch.testing.assert_close(c[0],a+b)
    def test_conflict_projection(self):
        a=torch.tensor([2.,0.]);b=torch.tensor([-3.,4.])
        c,s=merge([a],[b],'project');torch.testing.assert_close(c[0],torch.tensor([2.,4.]))
        self.assertTrue(s['conflict']);self.assertAlmostEqual(s['projected_replay_norm_ratio'],.8,places=6)
    def test_rescale_matched_norm(self):
        a=torch.tensor([2.,0.]);b=torch.tensor([-3.,4.])
        p,_=merge([a],[b],'project');r,_=merge([a],[b],'rescale')
        torch.testing.assert_close((p[0]-a).norm(),(r[0]-a).norm())
        self.assertLess(float(torch.dot(r[0]-a,a)),0)
    def test_nonconflict_unchanged(self):
        a=torch.tensor([1.,0.]);b=torch.tensor([2.,4.])
        for mode in ('sum','project','rescale'):
            c,_=merge([a],[b],mode);torch.testing.assert_close(c[0],a+b)
    def test_zero_and_unused(self):
        a=[torch.zeros(2),None,None];b=[torch.ones(2),torch.ones(3),None]
        for mode in ('sum','project','rescale'):
            c,_=merge(a,b,mode);torch.testing.assert_close(c[0],b[0]);torch.testing.assert_close(c[1],b[1]);self.assertIsNone(c[2])
    def test_antiparallel(self):
        a=torch.ones(3);b=-2*a
        for mode in ('project','rescale'):
            c,_=merge([a],[b],mode);torch.testing.assert_close(c[0],a)
    def test_sum_matches_autograd(self):
        x=torch.tensor([.2,.7],requires_grad=True)
        a=(x-1).square().sum();b=(x+2).square().sum()
        ga=torch.autograd.grad(a,x,retain_graph=True)[0]
        gb=torch.autograd.grad(b,x,retain_graph=True)[0]
        total=torch.autograd.grad(a+b,x)[0]
        combined,_=merge([ga],[gb],'sum');torch.testing.assert_close(combined[0],total)
    def test_paired_selection(self):
        import numpy as np
        from .run import selection, ARMS, tasks
        self.assertEqual(len(tasks()),9)
        draws=[selection(arm,1,np.random.default_rng(1)) for arm in ARMS]
        for ids,branches,weight in draws:
            np.testing.assert_array_equal(ids,draws[0][0]);np.testing.assert_array_equal(branches,draws[0][1])
            np.testing.assert_array_equal(np.sort(ids),np.arange(48));self.assertEqual(weight,1.)


if __name__=='__main__':unittest.main()
