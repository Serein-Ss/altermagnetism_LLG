"""Training-only geometric coupling audit before launching optimization."""
import numpy as np
import torch
from torch.nn import functional as F
from tests.training_restart_repair.data import load_train
from tests.temporal_source_repair.source import draw
from tests.bauer_p2_20260923.common import dump
from .coupling import match,schedule
from pathlib import Path

torch.set_num_threads(1)
bank,_=load_train();ids,branches=schedule(954300,16)
g=torch.Generator().manual_seed(954200);records=[]
for a,b in zip(ids,branches):
    y=F.normalize(torch.tensor(np.stack([bank[i][j] for i,j in zip(a,b)])),dim=-1)
    z=draw(y[:,0],2,g,0.);out,r=match(z,y)
    assert torch.equal(out[:,0],y[:,0]) and len(set(r['permutation']))==48
    records.append({k:v for k,v in r.items() if k!='permutation'})
dump(Path(__file__).parent/'coupling_audit.json',dict(rows=records,
    mean_random_cost=float(np.mean([r['random_cost'] for r in records])),
    mean_assignment_cost=float(np.mean([r['assignment_cost'] for r in records])),
    fixed_conditioning_states=48,source_parents=24,training_only=True,physical_certification=False))
