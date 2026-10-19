"""Frozen warm models evaluated at current model-visited TRAIN states."""
import argparse,json,time
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from tests.transition_block_repair.model import TransitionNet,rollout
from tests.bauer_comprehensive_20260923.run import require_gpu
from tests.bauer_p2_20260923.common import dump
from .data import HERE,rows,state,source,verify

@torch.no_grad()
def run(index,smoke):
    require_gpu('cuda');seed=(11,22,33)[index]
    if not smoke:assert json.loads((HERE/'data_audit.json').read_text())['passed']
    ck=torch.load(source(seed)/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert ck['task']['repair']=='warm' and ck['step']==72144
    net=TransitionNet().cuda();net.load_state_dict(ck['model']);net.eval()
    out=HERE/('smoke' if smoke else 'models')/str(seed);out.mkdir(parents=True,exist_ok=False)
    tick=time.perf_counter();records=[]
    with h5py.File(out/'paths.h5','x') as h:
        h.attrs.update(complete=False,checkpoint_contract=json.dumps(net.get_extra_state()))
        for row in rows(smoke):
            sid=row['index'];n=4 if smoke else 64
            initial=state(row);a=F.normalize(torch.tensor(initial,device='cuda',dtype=torch.float32),dim=-1)[None].expand(n,-1,-1)
            g=h.create_group(f'state{sid:03d}');g.attrs['metadata']=json.dumps(row);g.create_dataset('time_ps',data=[0.,.1])
            for steps in ((8,16) if smoke else (128,512)):
                gen=torch.Generator(device='cuda').manual_seed(966000+sid)
                torch.cuda.synchronize();began=time.perf_counter()
                x=rollout(net,a,gen,steps,1).cpu().numpy();torch.cuda.synchronize()
                norm=float(abs(np.linalg.norm(x,axis=-1)-1).max());anchor=float(abs(x[:,0]-a.cpu().numpy()).max())
                assert np.isfinite(x).all() and norm<2e-6 and anchor==0
                g.create_dataset(f'midpoint{steps}',data=x,compression='lzf')
                g.create_dataset(f'rng_final{steps}',data=gen.get_state().cpu().numpy())
                records.append(dict(state=sid,sampler=steps,seconds=time.perf_counter()-began,norm_error=norm,
                    anchor_error=anchor,initial_cast_max=float(abs(a[0].cpu().numpy()-initial).max())))
        h.attrs['complete']=True
    dump(out/'complete.json',dict(seed=seed,records=records,seconds=time.perf_counter()-tick,
        generating_seed=11,own_rollout_states=seed==11,scientific_certification=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run']);p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    verify();run(a.index,a.smoke)
