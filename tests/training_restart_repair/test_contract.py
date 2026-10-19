import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
import json
import hashlib
import h5py
import numpy as np
from . import data
from .run import restart_batch,tasks


class Tests(unittest.TestCase):
    def test_disjoint_seed_roles(self):
        used=set()
        for i in range(48):
            for role,(count,_,_) in data.ROLES.items():
                s=data.seeds(i,role);self.assertEqual(len(s),count)
                self.assertFalse(used&set(s));used.update(s)
                self.assertLess(int(s.max()),2**31)
        self.assertEqual(len(used),15360)
        with self.assertRaises(ValueError):data.seeds(48,'train')
        with self.assertRaises(ValueError):data.seeds(0,'dev')
    def test_paired_batches(self):
        bank=[np.full((64,2,16,3),i,dtype=np.float32) for i in range(48)]
        a=restart_batch(bank,np.random.default_rng(13));b=restart_batch(bank,np.random.default_rng(13))
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        self.assertEqual(a[0].shape,(8,2,16,3));self.assertTrue(np.all(a[0][:,0,0,0]==a[1]))
    def test_loader_rejects_diagnostic_role(self):
        with TemporaryDirectory() as d:
            root=Path(d);a=np.zeros((16,3));a[:,2]=1
            row=dict(index=0,parent_index=0,state_sha256=hashlib.sha256(a.tobytes()).hexdigest())
            (root/'manifest.json').write_text(json.dumps([row]))
            with h5py.File(root/'states.h5','w') as h:h.create_dataset('state000',data=a)
            p=root/'reference/00/state000_train.h5';p.parent.mkdir(parents=True)
            with h5py.File(p,'w') as h:
                h.attrs.update(complete=True,role='probe',split='train',metadata=json.dumps(row))
                h.create_dataset('spins',data=np.tile(a,(64,2,1,1)))
                h.create_dataset('seeds',data=data.seeds(0,'train'));h.create_dataset('time_ps',data=[0.,.1])
            with patch.object(data,'HERE',root),patch.object(data,'ROOT',root):
                with self.assertRaises(AssertionError):data.load_train()
                with h5py.File(p,'a') as h:h.attrs['role']='train'
                bank,_=data.load_train();self.assertEqual(bank[0].shape,(64,2,16,3))
    def test_task_controls(self):
        self.assertEqual(len(tasks()),9)
        for s in (11,22,33):
            self.assertEqual({x['repair'] for x in tasks() if x['seed']==s},{'native','restart_fm','restart_moments'})


if __name__=='__main__':unittest.main()
