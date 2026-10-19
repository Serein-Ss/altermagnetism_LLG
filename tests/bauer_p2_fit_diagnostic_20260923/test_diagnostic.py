import unittest
import numpy as np
from .run import decompose


class TestDecomposition(unittest.TestCase):
    def test_identity(self):
        s=np.random.default_rng(7).normal(size=(16,11,16,3))
        d=decompose(s)
        np.testing.assert_allclose(d['raw_dot'],d['drift_dot']+d['centered_cov'])

    def test_scale(self):
        s=np.random.default_rng(7).normal(size=(16,11,16,3))
        a,b=decompose(s),decompose(3*s)
        np.testing.assert_allclose(a['normalized_cov'],b['normalized_cov'])
        np.testing.assert_allclose(a['direction_cosine'],b['direction_cosine'])
        np.testing.assert_allclose(a['variance']*9,b['variance'])

    def test_static(self):
        d=decompose(np.ones((8,11,16,3)))
        for value in d.values():np.testing.assert_array_equal(value,0)

    def test_known_alternating(self):
        d=np.ones((8,10,4,3));d[:,1::2]*=-1
        s=np.concatenate([np.zeros((8,1,4,3)),np.cumsum(d,axis=1)],axis=1)
        a=decompose(s)
        np.testing.assert_allclose(a['direction_cosine'],-1)
        np.testing.assert_allclose(a['centered_cov'],0)
