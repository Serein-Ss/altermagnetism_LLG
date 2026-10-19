import unittest
import torch
from .objective import weight,loss
class Tests(unittest.TestCase):
    def test_normalized_positive(self):
        t=torch.linspace(0,1,10001,dtype=torch.float64);w=weight(t,'early_tau')
        self.assertAlmostEqual(float(torch.trapezoid(w,t)),1.,places=7)
        self.assertGreater(float(w.min()),0)
        self.assertAlmostEqual(float(w[0]),51/19)
        self.assertAlmostEqual(float(w[-1]),3/19)
        self.assertTrue(torch.equal(weight(t,'control'),torch.ones_like(t)))
    def test_loss_and_anchor(self):
        p=torch.zeros(2,2,4,3,requires_grad=True);y=torch.ones_like(p)
        t=torch.tensor([0.,1.])
        ordinary=loss(p,y,t,'control')
        self.assertAlmostEqual(float(ordinary.detach()),3.)
        weighted=loss(p,y,t,'early_tau');weighted.backward()
        self.assertTrue(torch.equal(p.grad[:,0],torch.zeros_like(p.grad[:,0])))
        self.assertAlmostEqual(float(p.grad[0,1,0,0]/p.grad[1,1,0,0]),17.,places=5)
    def test_rejections(self):
        for t in (torch.tensor([1.01]),torch.tensor([float('nan')])):
            with self.assertRaises(ValueError):weight(t,'early_tau')
