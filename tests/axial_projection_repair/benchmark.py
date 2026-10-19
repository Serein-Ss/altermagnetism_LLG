"""Short production-batch memory/timing preflight; not an accuracy benchmark."""
import time
import h5py
import torch
from torch.nn import functional as F
from tests.transition_block_repair.model import TransitionNet,rollout
from .model import AxialProjection
from .run import HERE,ROOT,source,dev_rows,verify,require_gpu,dump

verify();require_gpu('cuda');torch.set_num_threads(1)
ck=torch.load(source(11)/'checkpoint.pt',map_location='cpu',weights_only=True)
base=TransitionNet().cuda().eval();base.load_state_dict(ck['model'])
net=AxialProjection(base,8).eval();i,_=dev_rows()[0]
with h5py.File(ROOT/f'tests/reference_resolution/reference/{i:02d}/paths.h5') as h:
    a=F.normalize(torch.tensor(h['base/spins'][0,0],device='cuda',dtype=torch.float32),dim=-1)[None].expand(64,-1,-1)
g=torch.Generator(device='cuda').manual_seed(953100)
rollout(net,a,g,2,1);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
start=time.perf_counter();x=rollout(net,a,g,8,1);torch.cuda.synchronize();seconds=time.perf_counter()-start
assert torch.isfinite(x).all() and torch.equal(x[:,0],a)
#6 parents*(10 native+4 refined transition equivalents)+24 middle*(1+4),2chunks each,128/8steps.
estimated=seconds*2*(6*14+24*5)*16
dump(HERE/'benchmark.json',dict(batch=64,L=16,order=8,flow_steps=8,seconds=seconds,
    peak_allocated_bytes=torch.cuda.max_memory_allocated(),device=torch.cuda.get_device_name(),
    formal_C8_estimated_seconds=estimated,formal_C4_estimated_seconds=estimated/2,
    formal_C1_estimated_seconds=estimated/8,accuracy_claim=False))
