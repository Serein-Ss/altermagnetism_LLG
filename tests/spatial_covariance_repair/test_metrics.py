import unittest
import numpy as np
from .metrics import compare,moments


class Tests(unittest.TestCase):
    def paths(self,d):
        return np.stack((np.zeros_like(d),d),1)

    def test_matches_direct_covariance(self):
        d=np.random.default_rng(2).normal(size=(32,5,3));_,c=moments(self.paths(d))
        np.testing.assert_allclose(c.transpose(0,2,1,3).reshape(15,15),np.cov(d.reshape(32,15),rowvar=False))

    def test_local_equal_but_cross_site_different(self):
        rng=np.random.default_rng(1);d=rng.normal(size=(128,8,3));permuted=d.copy()
        for site in range(8):permuted[:,site]=d[rng.permutation(128),site]
        result=compare(self.paths(d),self.paths(permuted))
        self.assertLess(result['lags'][0]['covariance_rmse'],1e-12)
        self.assertGreater(result['lags'][1]['covariance_rmse'],.03)

    def test_rotation_and_reversal(self):
        rng=np.random.default_rng(4);g=self.paths(rng.normal(size=(128,8,3)));r=self.paths(rng.normal(size=(128,8,3)))
        q,_=np.linalg.qr(rng.normal(size=(3,3)));a=compare(g,r)
        for b in (compare(g@q,r@q),compare(g[:,:,::-1],r[:,:,::-1])):
            np.testing.assert_allclose([x['covariance_squared_bias_crossfold'] for x in a['lags']],
                [x['covariance_squared_bias_crossfold'] for x in b['lags']],atol=1e-12)

    def test_known_gaussian_squared_bias(self):
        rng=np.random.default_rng(5);values=[]
        for _ in range(400):
            g=self.paths(rng.normal(size=(128,5,3))*np.sqrt(1.5));r=self.paths(rng.normal(size=(128,5,3)))
            values.append(compare(g,r)['lags'][0]['covariance_squared_bias_crossfold'])
        self.assertLess(abs(np.mean(values)-3*.5**2),.05)

    def test_anchor_rejection(self):
        d=np.zeros((8,5,3));g=self.paths(d);r=g.copy();r[:,0]=1
        with self.assertRaises(ValueError):compare(g,r)


if __name__=='__main__':unittest.main()
