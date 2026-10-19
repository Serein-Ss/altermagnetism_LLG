import unittest
import numpy as np
from .coverage import descriptors,nearest
from .report import empirical
from .run import chosen,tasks


class Tests(unittest.TestCase):
    def test_selection(self):
        r=chosen();self.assertEqual([x['index'] for x in r],[1,17,33])
        self.assertTrue(all(x['split']=='train' and x['frame']==8 for x in r))
        self.assertEqual(len(tasks()),6)
    def test_covariance_ddof0(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(64,2,16,3));x/=np.linalg.norm(x,axis=-1,keepdims=True);x[:,0]=x[0,0]
        out=empirical(x,x);d=x[:,1]-x[:,0]
        for site in range(16):np.testing.assert_allclose(out['increment_cov']['reference'][site],np.cov(d[:,site],rowvar=False,ddof=0),atol=1e-12)
        self.assertNotIn('pointwise_ci95',out['increment_cov'])
    def test_joint_rotation(self):
        rng=np.random.default_rng(1);s=rng.normal(size=(4,16,3));s/=np.linalg.norm(s,axis=-1,keepdims=True)
        q,_=np.linalg.qr(rng.normal(size=(3,3)));q[:,0]*=np.linalg.det(q)
        np.testing.assert_allclose(descriptors(s),descriptors(s@q,np.array([0.,0.,1.])@q),atol=1e-12)
    def test_open_boundary(self):
        s=np.zeros((16,3));s[:,2]=1;d=descriptors(s)
        self.assertEqual(d[0,8],.5);self.assertEqual(d[1,8],1.)
        self.assertEqual(d[0,9],0.);self.assertEqual(d[5,9],1.)
        s[-1]=[1.,0.,0.];np.testing.assert_array_equal(d[0],descriptors(s)[0])
    def test_augmented_distance(self):
        rng=np.random.default_rng(3);a=rng.normal(size=(30,10));b=rng.normal(size=(6,10));scale=np.ones(10)
        x=nearest(a,b,scale);y=nearest(np.concatenate((a,b)),b,scale)
        self.assertTrue(np.all(y<=x));np.testing.assert_allclose(y,0,atol=1e-12)


if __name__=='__main__':unittest.main()
