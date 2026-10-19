import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import h5py
import numpy as np
from . import data
from .run import banks
from .recipe import call_due,weight_at


class DataTests(unittest.TestCase):
    def test_seed_namespaces_disjoint(self):
        seen=set();count=0
        for seed in data.SEEDS:
            for sid in range(48):
                for role in data.ROLES:
                    values=set(map(int,data.seeds(seed,sid,role)))
                    self.assertFalse(seen&values);seen.update(values);count+=len(values)
        self.assertEqual(count,46080);self.assertLess(max(seen),2**31)
        with self.assertRaises(ValueError):data.seeds(11,48,'train')
        with self.assertRaises(ValueError):data.seeds(12,0,'train')
        with self.assertRaises(ValueError):data.seeds(11,0,'test')

    def test_descriptor_axial_invariance(self):
        a=np.random.default_rng(6).normal(size=(16,3));a/=np.linalg.norm(a,axis=-1,keepdims=True)
        angle=.731;q=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        np.testing.assert_allclose(data.descriptor(a@q),data.descriptor(a),atol=1e-14)
        self.assertTrue(np.isfinite(data.descriptor(a)).all())

    def test_bank_only_model_slot_changes(self):
        old=(['real'],['unused'],['stale']);fresh=['fresh']
        with patch('tests.on_policy_refresh_repair.run.old_banks',return_value=(old,{'old':'hash'})),patch.object(data,'load_train',return_value=(fresh,{'new':'hash'})) as loader:
            control,hashes=banks(11,'stale',True);self.assertIs(control,old);loader.assert_not_called()
            candidate,hashes=banks(11,'current',True);self.assertIs(candidate[0],old[0]);self.assertIs(candidate[1],old[1]);self.assertIs(candidate[2],fresh)
            self.assertEqual(hashes,{'old':'hash','new':'hash'})
        for step in range(1,2049):
            self.assertEqual(call_due(step,'stale'),call_due(step,'current'))
            self.assertEqual(weight_at(step,'stale'),weight_at(step,'current'))

    def test_loader_rejects_role_split_and_wrong_anchor(self):
        temporary=data.HERE/'tmp';temporary.mkdir(exist_ok=True)
        row=dict(index=0,model_seed=11,parent_index=0,source_group='parent',split='train')
        a=np.zeros((16,3));a[:,2]=1;spins=np.repeat(a[None,None],64,axis=0);spins=np.repeat(spins,2,axis=1)
        with tempfile.TemporaryDirectory(dir=temporary) as folder:
            p=Path(folder)/'train.h5'
            with h5py.File(p,'w') as h:
                h.attrs.update(complete=True,role='probe',split='train',metadata=json.dumps(row))
                h.create_dataset('spins',data=spins);h.create_dataset('seeds',data=data.seeds(11,0,'train'))
            with patch.object(data,'rows',return_value=[row]),patch.object(data,'path',return_value=p),patch.object(data,'state',return_value=a):
                with self.assertRaises(AssertionError):data.load_train(11,True)
                with h5py.File(p,'r+') as h:h.attrs['role']='train';h.attrs['split']='dev'
                with self.assertRaises(AssertionError):data.load_train(11,True)
                with h5py.File(p,'r+') as h:h.attrs['split']='train';h['spins'][0,0,0,2]=-1
                with self.assertRaises(AssertionError):data.load_train(11,True)
                with h5py.File(p,'r+') as h:h['spins'][0,0,0,2]=1
                bank,hashes=data.load_train(11,True);self.assertEqual(bank[0].shape,(64,2,16,3));self.assertEqual(len(hashes),1)


if __name__=='__main__':unittest.main()
