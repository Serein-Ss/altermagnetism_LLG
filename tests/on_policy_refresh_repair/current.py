import json
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from tests.transition_block_repair.model import rollout
from . import data


@torch.no_grad()
def evaluate_current(net,out,seed,smoke=False):
    with h5py.File(out,'x') as h:
        h.attrs.update(complete=False,scope='TRAIN-source matched-state independent-noise diagnostic')
        for row in data.rows(seed,smoke):
            n=4 if smoke else 64;sid=row['index']
            a=F.normalize(torch.tensor(data.state(row),device='cuda',dtype=torch.float32),dim=-1)[None].expand(n,-1,-1)
            gen=torch.Generator(device='cuda').manual_seed(985000+sid)
            x=rollout(net,a,gen,8 if smoke else 128,1).cpu().numpy()
            assert np.isfinite(x).all() and abs(np.linalg.norm(x,axis=-1)-1).max()<2e-6
            assert np.array_equal(x[:,0],a.cpu().numpy())
            g=h.create_group(f'state{sid:03d}');g.attrs['metadata']=json.dumps(row)
            g.create_dataset('spins',data=x,compression='lzf');g.create_dataset('time_ps',data=[0.,.1])
        h.attrs['complete']=True
