"""Small end-to-end tests before releasing long arrays."""
import time
import torch
from .run import HERE,run_toy,require_gpu
from .oracle import conditional,targets
from .model import source
from scripts.model.gate_f import path_interpolate
from tests.bauer_p2_20260923.common import dump


def main():
    require_gpu('cuda');torch.set_num_threads(1)
    out=HERE/'toy_smoke';out.mkdir(exist_ok=False)
    result=run_toy(dict(domain='toy',arm='oracle_supervision',modes=2,seed=11,updates=8),out,True)
    g=torch.Generator(device='cuda').manual_seed(929)
    initial=torch.nn.functional.normalize(torch.randn(8,16,3,device='cuda',generator=g),dim=-1)
    ys=targets(initial);bad_count=0
    for _ in range(100):
        z=source(initial,11,g,0.);tau=torch.rand(8,device='cuda',generator=g)*.95
        chosen=torch.randint(2,(8,),device='cuda',generator=g)
        x,_,_=path_interpolate(z,ys[torch.arange(8,device='cuda'),chosen],tau,g)
        _,_,bad=conditional(x,tau,ys);bad_count+=int(bad.sum())
    result['fresh_interpolant_support_failures']=bad_count
    if bad_count:raise RuntimeError('analytic support audit failed')
    dump(out/'complete.json',result)


if __name__=='__main__':main()
