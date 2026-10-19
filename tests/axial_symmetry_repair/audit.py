import torch
from pathlib import Path
from tests.transition_block_repair.model import TransitionNet
from tests.bauer_p2_20260923.common import dump
from .symmetry import probe


if __name__=='__main__':
    torch.set_num_threads(1);here=Path(__file__).resolve().parent;root=here.parents[1];result=[]
    for i,seed in enumerate((11,22,33)):
        net=TransitionNet();ck=torch.load(root/f'tests/local_condition_repair/train/{i*3:02d}/checkpoint.pt',map_location='cpu',weights_only=True)
        net.load_state_dict(ck['model']);net.eval();r=probe(net)
        result.extend([dict(seed=seed,**x) for x in r])
        print(seed,'relative RMS min/mean/max',min(x['relative_rms'] for x in r),sum(x['relative_rms'] for x in r)/len(r),max(x['relative_rms'] for x in r))
    dump(here/'source_audit.json',dict(rows=result,scope='TRAIN fixed transport probes only;path-distribution accuracy not established'))
