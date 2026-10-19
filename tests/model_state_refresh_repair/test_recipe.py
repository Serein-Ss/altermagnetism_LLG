import unittest
import numpy as np
from .recipe import replay_roles,rate
from .data import seeds,ROLES
from tests.model_state_coverage_repair.data import seeds as old_seeds

class Tests(unittest.TestCase):
    def test_group_balance(self):
        for arm,role in (('stale_replay',1),('refreshed_replay',2)):
            x=np.concatenate([replay_roles(i,arm) for i in range(1,17)])
            self.assertEqual((x==0).sum(),384);self.assertEqual((x==role).sum(),384)
            for group in x.reshape(-1,16):self.assertEqual(len(set(group)),1)
        with self.assertRaises(ValueError):replay_roles(0,'stale_replay')
        with self.assertRaises(ValueError):replay_roles(1,'unknown')
    def test_fresh_noise_disjoint(self):
        prior=set();fresh=set()
        for sid in range(48):
            for role in ROLES:prior.update(map(int,old_seeds(sid,role)))
        for sid in range(48):
            for role,(n,_,_) in ROLES.items():
                values=set(map(int,seeds(sid,role)))
                self.assertEqual(len(values),n);self.assertFalse(values&(fresh|prior));fresh|=values
        self.assertEqual(len(fresh),15360)
    def test_preserved_learning_rate(self):
        self.assertAlmostEqual(rate(200,6000),3e-4);self.assertAlmostEqual(rate(6000,6000),9e-6)
