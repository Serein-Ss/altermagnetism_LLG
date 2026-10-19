import unittest
import numpy as np
from .data import seeds,ROLES
from tests.intermediate_restart_repair.run import normalize_state

class Tests(unittest.TestCase):
    def test_noise_roles_disjoint(self):
        used=set()
        for sid in range(48):
            for role,(count,_,_) in ROLES.items():
                s=set(map(int,seeds(sid,role)))
                self.assertEqual(len(s),count);self.assertFalse(used&s);used|=s
        self.assertEqual(len(used),15360)
        with self.assertRaises(ValueError):seeds(48,'train')
        with self.assertRaises(ValueError):seeds(0,'dev')
    def test_state_normalization(self):
        a=np.array([[.5,.2,.1],[1.,2.,3.]])
        b=normalize_state(a)
        np.testing.assert_allclose(np.linalg.norm(b,axis=-1),1.,atol=1e-15)
        np.testing.assert_allclose(np.cross(a,b),0.,atol=1e-15)
        with self.assertRaises(ValueError):normalize_state(np.zeros((16,3)))
