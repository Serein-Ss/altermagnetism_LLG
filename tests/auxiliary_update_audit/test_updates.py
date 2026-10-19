import copy,unittest
import torch
from .updates import shadow,flat,combine


class Tests(unittest.TestCase):
    def test_matches_real_adam_clipping_and_no_mutation(self):
        torch.manual_seed(3);p=[torch.nn.Parameter(torch.randn(12)),torch.nn.Parameter(torch.randn(7))]
        opt=torch.optim.AdamW(p,lr=3e-4,weight_decay=1e-5)
        for _ in range(3):
            for v in p:v.grad=torch.randn_like(v)
            opt.step()
        saved=copy.deepcopy(opt.state_dict());before=[v.detach().clone() for v in p];grads=[torch.randn_like(v)*100 for v in p]
        delta,info=shadow(p,saved,grads)
        for v,g in zip(p,grads):v.grad=g.clone()
        torch.nn.utils.clip_grad_norm_(p,10.);opt.step()
        torch.testing.assert_close(delta,torch.cat([(v.detach()-b).flatten() for v,b in zip(p,before)]),rtol=0,atol=0)
        self.assertLess(info['clip_factor'],1)
        self.assertEqual(int(saved['state'][0]['step']),3)

    def test_unused_parameters_and_zero_increment(self):
        p=[torch.nn.Parameter(torch.ones(3)),torch.nn.Parameter(torch.ones(2))]
        opt=torch.optim.AdamW(p,lr=.01);s=opt.state_dict();before=[v.detach().clone() for v in p]
        d,_=shadow(p,s,[torch.ones(3),None])
        self.assertEqual(float(d[-2:].abs().max()),0.)
        for x,y in zip(p,before):torch.testing.assert_close(x,y,rtol=0,atol=0)
        a=[torch.ones(3),None];b=[torch.zeros(3),None]
        da,_=shadow(p,s,a);db,_=shadow(p,s,combine(a,b,.1));torch.testing.assert_close(da,db,rtol=0,atol=0)

    def test_known_sgd_gradient_ratio_not_adam_equivalence(self):
        p=[torch.ones(3)];a=[torch.ones(3)];b=[torch.ones(3)*2]
        self.assertAlmostEqual(float((flat(combine(a,b,.1),p)-flat(a,p)).norm()/flat(a,p).norm()),.2,places=6)


if __name__=='__main__':unittest.main()
