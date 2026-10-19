import unittest
import numpy as np
import torch
from .run import learning_rate,tasks,STAGES


class ScheduleTests(unittest.TestCase):
    def test_endpoints_monotonic(self):
        for arm,peak in [('restart_high',3e-4),('restart_low',3e-5),('continue_floor',3e-6)]:
            values=[learning_rate(arm,i) for i in range(1,6001)]
            self.assertAlmostEqual(values[0],peak);self.assertAlmostEqual(values[-1],3e-6)
            self.assertTrue(np.all(np.diff(values)<=0))

    def test_previous_control_formula(self):
        for step in (1,500,1500,3000,6000):
            expected=3e-6+(3e-4-3e-6)*.5*(1+np.cos(np.pi*(step-1)/5999))
            self.assertAlmostEqual(learning_rate('restart_high',step),expected)

    def test_invalid(self):
        for args in [('bad',1),('restart_low',0),('continue_floor',6001)]:
            with self.assertRaises(ValueError):learning_rate(*args)

    def test_design(self):
        self.assertEqual(len(tasks()),9);self.assertEqual(STAGES,(500,1500,3000,6000))
        for seed in (11,22,33):self.assertEqual(len([r for r in tasks() if r['seed']==seed]),3)

    def test_rng_guard(self):
        state=torch.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):torch.randn(100)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
