import unittest
import torch
from torch.nn import functional as F
from tests.temporal_source_repair.source import draw
from .target import labels


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(18);torch.set_num_threads(1)
        self.bank=F.normalize(torch.randn(64,2,5,3),dim=-1)
        self.bank[:,0]=self.bank[0,0].clone();self.y=self.bank[:8]
        self.z=draw(self.y[:,0],2,torch.Generator().manual_seed(19),0.)

    def test_late_labels_unchanged(self):
        x,v,rb,d=labels(self.z,self.y,torch.full((8,),.95),self.bank,torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(v,rb));self.assertEqual(d['teacher_count'],0)

    def test_tau_zero_conditional_mean(self):
        from scripts.model.gate_f import path_interpolate
        x,v,rb,d=labels(self.z,self.y,torch.zeros(8),self.bank,torch.Generator().manual_seed(1))
        y=F.normalize(self.bank.double(),dim=-1)
        for i in range(8):
            z=F.normalize(self.z[i:i+1].double(),dim=-1).expand_as(y).clone();z[:,0]=y[:,0]
            _,allv,_=path_interpolate(z,y,torch.zeros(64,dtype=torch.float64),torch.Generator().manual_seed(1))
            torch.testing.assert_close(rb[i],allv.mean(0).float(),atol=2e-6,rtol=2e-6)
        self.assertAlmostEqual(d['effective_modes'],64.,places=6)

    def test_tangent_anchor_and_rng_pair(self):
        a=torch.Generator().manual_seed(3);b=torch.Generator().manual_seed(3)
        result=labels(self.z,self.y,torch.linspace(.05,.99,8),self.bank,a)
        other=labels(self.z,self.y,torch.linspace(.05,.99,8),self.bank,b)
        x,v,rb,_=result
        self.assertTrue(torch.equal(a.get_state(),b.get_state()))
        for p,q in zip(result[:3],other[:3]):self.assertTrue(torch.equal(p,q))
        torch.testing.assert_close(x[:,0],F.normalize(self.y[:,0].double(),dim=-1).float())
        self.assertLess(float((x*rb).sum(-1).abs().max()),2e-6)
        self.assertEqual(float(rb[:,0].abs().max()),0.)

    def test_wrong_bank_rejected(self):
        bank=self.bank.clone();bank[1,0]=bank[1,1]
        with self.assertRaises(ValueError):labels(self.z,self.y,torch.zeros(8),bank,torch.Generator())


if __name__=='__main__':unittest.main()
