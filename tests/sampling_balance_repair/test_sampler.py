import hashlib
import unittest
from collections import Counter
import numpy as np
import torch
from tests.bauer_closed_loop_20260919.run import batch
from .run import RefillWindows


class FakeWindows:
    def __init__(self):
        self.records=[('fake',dict(parent_id='L16_test',family='tilted'),range(0,30,10),10)]
        self.ends=[24]
        rng=np.random.default_rng(1);x=rng.normal(size=(8,31,16,3));self.x=(x/np.linalg.norm(x,axis=-1,keepdims=True)).astype(np.float32)
        self.x[:,0]=self.x[0,0]
    def __len__(self):return 24
    def __getitem__(self,i):
        b,w=divmod(i,3)
        return dict(spins=self.x[b,w*10:w*10+11].copy())


def audit():return dict(counts=Counter(),hash=hashlib.sha256())


class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.ds=FakeWindows();self.pool={'L16_test':np.repeat(self.ds.x[:,:11],9,axis=0)}

    def test_native_old_exact_legacy(self):
        a=np.random.default_rng(17);b=np.random.default_rng(17)
        w=RefillWindows(self.ds,self.pool,np.random.default_rng(3),False,audit())
        for _ in range(100):
            y,t=batch([self.ds],a,dict(batch=8,save_ps=.1),'cpu','B')
            z,u=batch([w],b,dict(batch=8,save_ps=.1),'cpu','B')
            self.assertTrue(torch.equal(y,z));self.assertTrue(torch.equal(t,u))
        self.assertEqual(a.bit_generator.state,b.bit_generator.state)

    def test_paired_schedule_and_noise(self):
        a=np.random.default_rng(4);b=np.random.default_rng(4);ra=np.random.default_rng(5);rb=np.random.default_rng(5)
        ca=audit();cb=audit();wa=RefillWindows(self.ds,self.pool,ra,False,ca);wb=RefillWindows(self.ds,self.pool,rb,True,cb)
        for _ in range(100):
            batch([wa],a,dict(batch=8,save_ps=.1),'cpu','B');batch([wb],b,dict(batch=8,save_ps=.1),'cpu','B')
        self.assertEqual(ca['hash'].hexdigest(),cb['hash'].hexdigest())
        self.assertEqual(a.bit_generator.state,b.bit_generator.state);self.assertEqual(ra.bit_generator.state,rb.bit_generator.state)
        self.assertGreater(cb['counts']['new_branch'],0)

    def test_later_windows_unchanged_and_anchor_fixed(self):
        w=RefillWindows(self.ds,self.pool,np.random.default_rng(8),True,audit())
        for i in range(len(w)):
            x=w[i]['spins'];original=self.ds[i]['spins']
            if i%3:np.testing.assert_array_equal(x,original)
            else:np.testing.assert_array_equal(x[0],original[0])

    def test_ineligible_parent_unchanged(self):
        w=RefillWindows(self.ds,{},np.random.default_rng(8),True,audit())
        for i in range(len(w)):np.testing.assert_array_equal(w[i]['spins'],self.ds[i]['spins'])


if __name__=='__main__':unittest.main()
