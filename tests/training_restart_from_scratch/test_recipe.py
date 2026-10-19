import unittest
import torch
from tests.transition_block_repair.model import TransitionNet
from .recipe import initialize,rate

class Tests(unittest.TestCase):
    def test_warm_fresh_and_empty_optimizer(self):
        base=TransitionNet();torch.nn.init.normal_(base.output.weight,std=.1)
        checkpoint={'model':base.state_dict()}
        warm,wo=initialize(checkpoint,'warm',11)
        fresh,fo=initialize(checkpoint,'fresh',11)
        again,_=initialize(checkpoint,'fresh',11)
        self.assertFalse(wo.state);self.assertFalse(fo.state)
        for name,p in base.named_parameters():
            self.assertTrue(torch.equal(p,dict(warm.named_parameters())[name]))
        self.assertTrue(torch.equal(fresh.input.weight,again.input.weight))
        self.assertFalse(torch.equal(warm.input.weight,fresh.input.weight))
        self.assertTrue(torch.equal(fresh.output.weight,torch.zeros_like(fresh.output.weight)))
        self.assertEqual(sum(p.numel() for p in fresh.parameters()),sum(p.numel() for p in warm.parameters()))
        self.assertEqual({id(p) for g in wo.param_groups for p in g['params']},{id(p) for p in warm.parameters()})
    def test_seed_and_global_rng(self):
        before=torch.random.get_rng_state().clone()
        a,_=initialize({},'fresh',11)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        b,_=initialize({},'fresh',22)
        self.assertFalse(torch.equal(a.input.weight,b.input.weight))
    def test_learning_schedule(self):
        self.assertAlmostEqual(rate(200,12000),1e-3)
        self.assertAlmostEqual(rate(12000,12000),3e-5)
        self.assertGreater(rate(1,12000),0)
        self.assertAlmostEqual(rate(8,16),1e-3)
        self.assertAlmostEqual(rate(16,16),3e-5)
        self.assertTrue(all(0<rate(s,12000)<=1e-3 for s in range(1,12001)))
