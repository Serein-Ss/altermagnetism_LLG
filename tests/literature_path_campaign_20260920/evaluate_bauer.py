"""Selected repaired model, diagnostic size ladder, independent noise branches."""
import argparse
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from tests.bauer_closed_loop_20260919.metrics import compare
from tests.bauer_transition_20260920.run import config, rollout, temporal_errors
from tests.bauer_two_stage_20260919.common import allocation, dump
from .models import ReversalNet
from .long import selected

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(); p.add_argument('--device', required=True, choices=['cuda']); a = p.parse_args()
    allocation(a.device); torch.set_num_threads(1); cfg = config()
    checkpoint = selected('bauer','fixed'); state = torch.load(checkpoint,map_location='cuda',weights_only=False)
    net = ReversalNet('C',64,3).cuda(); net.load_state_dict(state['model']); net.eval()
    out = HERE/'bauer_evaluation'; out.mkdir(exist_ok=False); rows = []
    for length in [16,25,32,50,100]:
        with h5py.File(out/f'L{length}.h5','x') as h:
            h.attrs['complete'] = False
            for parent in range(6):
                source = HERE.parent/'bauer_two_stage_20260919/data'/f'L{length}'/f'test_future_{parent}.h5'
                with h5py.File(source) as ref:
                    if not ref.attrs['complete']: raise RuntimeError('Incomplete reference')
                    truth = ref['spins'][:,:101].astype('f8'); initial = ref['initial'][:]
                s = F.normalize(torch.tensor(initial,device='cuda',dtype=torch.float32),dim=-1)[None].repeat(16,1,1)
                started = time.perf_counter()
                x = rollout(net,s,10.,260920000+length*100+parent,64,'transition',cfg).cpu().numpy()
                group = h.create_group(f'parent_{parent}'); group.create_dataset('generated',data=x,compression='lzf')
                group.create_dataset('reference',data=truth,compression='lzf')
                for frames in [11,101]:
                    rows.append(dict(L=length,parent=parent,horizon_ps=(frames-1)*.1,
                       raw=compare(x[:,:frames].astype('f8'),truth[:16,:frames]),
                       reference_floor=compare(truth[16:,:frames],truth[:16,:frames]),
                       temporal=temporal_errors(x[:,:frames],truth[:,:frames]),
                       seconds_10ps=time.perf_counter()-started))
                print(length,parent,flush=True)
            h['time_ps'] = np.arange(101)*.1; h.attrs['complete'] = True
    dump(out/'metrics.json',dict(checkpoint=str(checkpoint),rows=rows,scientific_certification=False,
         note='Previously inspected diagnostic test parents, not a fresh final holdout. Development-only checkpoint selection.'))


if __name__ == '__main__': main()
