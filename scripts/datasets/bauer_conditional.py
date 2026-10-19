"""Explicit TRAIN-only banks and branch refill for the Bauer candidate."""
import json
from pathlib import Path
import h5py
import numpy as np


def load_bank(manifest):
    manifest=Path(manifest);rows=json.loads(manifest.read_text());bank=[]
    for row in rows:
        path=(manifest.parent/row['file']).resolve()
        with h5py.File(path) as h:
            if not h.attrs.get('complete') or h.attrs['role']!='train' or h.attrs['split']!='train':
                raise ValueError('only completed TRAIN conditional banks may supervise training')
            metadata=json.loads(h.attrs['metadata'])
            if metadata['parent']['split']!='train' or metadata['parent']['parent_id']!=row['parent_id']:
                raise ValueError('conditional source-group mismatch')
            if not np.allclose(h['time_ps'][:],[0.,.1],atol=1e-12,rtol=0):raise ValueError('native0.1ps bank required')
            x=h['spins'][:].astype(np.float32)
            if x.shape!=(64,2,16,3) or not np.allclose(x[:,0],x[:1,0],atol=1e-6,rtol=0):
                raise ValueError('64 independent futures at one L16 initial required')
            bank.append(x)
    if len(bank)!=48:raise ValueError('registered recipe requires48 TRAIN states')
    return bank


class RefillWindows:
    def __init__(self,base,pool,rng):
        self.base=base;self.pool=pool;self.rng=rng;self.records=base.records;self.ends=base.ends

    def __len__(self):return len(self.base)

    def __getitem__(self,index):
        item=self.base[index];g=int(np.searchsorted(self.ends,index,side='right'))
        offset=index-(self.ends[g-1] if g else 0);_,row,starts,_=self.records[g]
        _,window=divmod(offset,len(starts));parent=row['parent_id']
        if starts[window]==0 and parent in self.pool:
            item['spins']=self.pool[parent][int(self.rng.integers(len(self.pool[parent])))].copy()
        return item
