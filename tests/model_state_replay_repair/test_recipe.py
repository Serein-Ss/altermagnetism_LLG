import unittest
import numpy as np
from .recipe import replay_roles,rate
from tests.conditional_coupling_repair.coupling import schedule

class Tests(unittest.TestCase):
    def test_role_balance_and_shared_groups(self):
        ids,branches=schedule(968311,16)
        counts=np.zeros((2,48,64),dtype=int)
        for step,(i,b) in enumerate(zip(ids,branches),1):
            roles=replay_roles(step,'model_replay')
            self.assertEqual(roles.shape,(48,))
            for start in (0,16,32):
                self.assertEqual(len(set(roles[start:start+16])),1)
                self.assertEqual(len(set(i[start:start+16])),1)
            np.add.at(counts,(roles,i,b),1)
            self.assertTrue((replay_roles(step,'real_control')==0).all())
        self.assertEqual(counts[0].sum(),384);self.assertEqual(counts[1].sum(),384)
        self.assertTrue((counts.sum((0,2))==16).all())
        with self.assertRaises(ValueError):replay_roles(0,'real_control')
        with self.assertRaises(ValueError):replay_roles(1,'unknown')
    def test_learning_rate(self):
        self.assertAlmostEqual(rate(200,6000),3e-4)
        self.assertAlmostEqual(rate(6000,6000),9e-6)
        self.assertTrue(all(0<rate(i,6000)<=3e-4 for i in range(1,6001)))
