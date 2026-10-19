import unittest
import numpy as np
from .run import seeds,normalize_state


class RestartTests(unittest.TestCase):
    def test_seed_independence(self):
        values=np.concatenate([seeds(i,fine,128) for i in range(216) for fine in (False,True)])
        self.assertEqual(len(np.unique(values)),55296)
        self.assertLess(values.max(),2**31)
        for i in (0,215):np.testing.assert_array_equal(seeds(i,False,4),seeds(i,False,128)[:4])

    def test_normalization(self):
        x=np.random.default_rng(1).normal(size=(16,3));x/=np.linalg.norm(x,axis=-1,keepdims=True)
        a=normalize_state(x.astype('f4'));self.assertEqual(a.dtype,np.float64)
        np.testing.assert_allclose(np.linalg.norm(a,axis=-1),1,atol=1e-15)
        with self.assertRaises(ValueError):normalize_state(np.zeros((16,3)))
        with self.assertRaises(ValueError):normalize_state(np.full((16,3),np.nan))


if __name__=='__main__':unittest.main()
