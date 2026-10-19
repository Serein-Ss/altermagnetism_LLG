import copy
import unittest
import numpy as np
from .run import selection, tasks, ARMS
from tests.training_restart_repair.data import rows, load_train


class Tests(unittest.TestCase):
    def test_tasks(self):
        self.assertEqual(len(tasks()),9)
        for seed in (11,22,33):
            self.assertEqual({x['repair'] for x in tasks() if x['seed']==seed},set(ARMS))

    def test_exact_balance(self):
        rng=np.random.default_rng(1)
        for step in range(1,21):
            ids,branches,w=selection('dense_balanced',step,rng)
            np.testing.assert_array_equal(np.sort(ids),np.arange(48))
            self.assertTrue(((branches>=0)&(branches<64)).all());self.assertEqual(w,1.)

    def test_sparse_contract(self):
        rng=np.random.default_rng(2)
        before=copy.deepcopy(rng.bit_generator.state)
        ids,_,w=selection('sparse_iid',1,rng)
        self.assertEqual(len(ids),0);self.assertEqual(w,0);self.assertEqual(before,rng.bit_generator.state)
        ids,branches,w=selection('sparse_iid',2,rng)
        self.assertEqual(len(ids),8);self.assertEqual(len(branches),8);self.assertEqual(w,.5)

    def test_dense_iid_and_resume(self):
        rng=np.random.default_rng(3);state=copy.deepcopy(rng.bit_generator.state)
        a=selection('dense_iid',1,rng);rng.bit_generator.state=state;b=selection('dense_iid',1,rng)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        self.assertEqual(len(a[0]),48);self.assertLess(len(set(a[0])),48);self.assertEqual(a[2],1.)

    def test_train_only(self):
        rr=rows();bank,hashes=load_train()
        self.assertEqual(len(bank),48);self.assertEqual(len(hashes),48)
        self.assertTrue(all(r['split']=='train' for r in rr))
        self.assertTrue(all(p.endswith('_train.h5') for p in hashes))
        self.assertEqual(len({r['source_group'] for r in rr}),24)

    def test_reject_arm(self):
        with self.assertRaises(ValueError):selection('unknown',1,np.random.default_rng())


if __name__=='__main__':unittest.main()
