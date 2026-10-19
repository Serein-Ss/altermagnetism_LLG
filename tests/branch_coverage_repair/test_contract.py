import json
import unittest
from unittest.mock import patch
from collections import Counter
import numpy as np
import h5py
import torch
from numba import set_num_threads
from .run import ROOT, parents, tasks, branch_seeds, checkpoint, targeted_batch, build, reference_inputs, simulate, pools


class Contracts(unittest.TestCase):
    def test_parent_selection(self):
        rows=parents()
        self.assertEqual(Counter(r['family'] for r in rows),dict(tilted=8,interior_domain=8,single_wall=8))
        dev=json.loads((ROOT/'tests/bauer_p2_20260923/manifest_dev.json').read_text())
        self.assertFalse({r['parent_id'] for r in rows}&{r['parent_id'] for r in dev})
        self.assertTrue(all(r['split']=='train' and r['L']==16 for r in rows))

    def test_separate_random_streams(self):
        values=np.concatenate([branch_seeds(i,role,n) for i in range(24) for role,n in [('train',64),('probe',128)]])
        self.assertEqual(len(values),4608)
        self.assertEqual(len(np.unique(values)),len(values))

    def test_matched_parent_and_quantile_draws(self):
        old=[np.tile(np.array([0.,0.,1.],dtype=np.float32),(8,11,16,1)) for _ in range(24)]
        new=[np.repeat(v,9,axis=0) for v in old]
        a=np.random.default_rng(7);b=np.random.default_rng(7)
        for _ in range(12):
            x,t,ids,u=targeted_batch(old,a,8,'cpu');y,q,j,v=targeted_batch(new,b,8,'cpu')
            np.testing.assert_array_equal(ids,j);np.testing.assert_array_equal(u,v)
            torch.testing.assert_close(x,y);torch.testing.assert_close(t,q)
        self.assertEqual(a.bit_generator.state,b.bit_generator.state)

    def test_actual_checkpoint_warm_start(self):
        torch.set_num_threads(1)
        for seed in (11,22,33):
            task=next(t for t in tasks() if t['seed']==seed)
            net,opt,_=build(dict(task,repair='control'),'cpu')
            ck=torch.load(checkpoint(seed),map_location='cpu',weights_only=True)
            self.assertEqual(ck['step'],24000)
            net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
            self.assertEqual(len(opt.state),len(ck['optimizer']['state']))
            for k,v in net.state_dict().items():
                if isinstance(v,torch.Tensor):self.assertTrue(torch.equal(v,ck['model'][k]))

    def test_original_solver_replay(self):
        set_num_threads(2);row=parents()[0];initial,config=reference_inputs(row)
        with h5py.File(ROOT/row['path']) as h: seeds=h['seeds'][:2];expected=h['spins'][:2,:11]
        x,_=simulate(np.repeat(initial[None],2,axis=0),seeds,config,.25,1.)
        np.testing.assert_allclose(x,expected,atol=7e-8,rtol=0)
        self.assertLess(np.max(abs(np.linalg.norm(x,axis=-1)-1)),1e-10)

    def test_training_never_opens_probe(self):
        class TrainOnlyAccess(Exception):pass
        accessed=[]
        def intercept(path,*args,**kwargs):
            accessed.append(str(path))
            if str(path).endswith('/train.h5'):raise TrainOnlyAccess()
            self.assertFalse(str(path).endswith('/probe.h5'))
            return original(path,*args,**kwargs)
        original=h5py.File
        with patch('tests.branch_coverage_repair.run.h5py.File',side_effect=intercept):
            with self.assertRaises(TrainOnlyAccess):pools(True)
        self.assertEqual(len(accessed),2)


if __name__=='__main__':unittest.main()
