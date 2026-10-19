"""TRAIN-only geometric residual audit, with fresh-latent validation."""
import json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tests.training_restart_repair.data import load_train
from tests.conditional_teacher_repair.target import labels
from tests.conditional_teacher_repair.run import source as previous_source
from tests.transport_risk_audit.run import verify as verify_previous
from tests.transition_block_repair.model import TransitionNet
from tests.temporal_source_repair.source import draw
from tests.bauer_comprehensive_20260923.run import require_gpu
from tests.bauer_p2_20260923.common import dump
from .model import basis
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

@torch.no_grad()
def main():
    require_gpu('cuda');verify_previous();bank,_=load_train();tick=time.perf_counter();records=[]
    for seed in (11,22,33):
        # The source is conditional_teacher step54000, NOT optimizer highLR.
        p=ROOT/f'tests/conditional_teacher_repair/train/{(11,22,33).index(seed)*2+1:02d}/checkpoint.pt'
        ck=torch.load(p,map_location='cpu',weights_only=True);assert ck['step']==54000
        net=TransitionNet().cuda();net.load_state_dict(ck['model']);net.eval()
        for tv in (.1,.3,.5,.7,.9):
            stats=[]
            for rep in (0,1):
                gram=torch.zeros(2,2,device='cuda',dtype=torch.float64);rhs=torch.zeros(2,device='cuda',dtype=torch.float64)
                sse=0.;count=0
                for sid,b in enumerate(bank):
                    y=F.normalize(torch.tensor(b,device='cuda'),dim=-1)
                    g=torch.Generator(device='cuda').manual_seed(958000+rep*1000+sid)
                    z=draw(y[:,0],2,g,0.);t=torch.full((64,),tv,device='cuda')
                    x,_,c,_=labels(z,y,t,y,g)
                    p=net.velocity(x,t,x[:,0],x.new_tensor([0.,.1]))
                    error=(c[:,1:]-p[:,1:]).double().flatten()
                    log,cross=basis(x,x[:,0])
                    design=torch.stack((log[:,1:].double().flatten(),cross[:,1:].double().flatten()),-1)
                    gram+=design.T@design;rhs+=design.T@error;sse+=float(error.square().sum())
                    count+=error.numel()/3
                stats.append(dict(gram=gram.cpu().numpy(),rhs=rhs.cpu().numpy(),sse=sse,count=count))
            coef=np.linalg.lstsq(stats[0]['gram'],stats[0]['rhs'],rcond=None)[0]
            test=stats[1];before=test['sse']/test['count']
            after=(test['sse']-2*coef@test['rhs']+coef@test['gram']@coef)/test['count']
            records.append(dict(seed=seed,tau=tv,coefficients=coef.tolist(),validation_before=before,
                validation_after=after,fraction_reduced=(before-after)/before))
    torch.cuda.synchronize()
    dump(HERE/'geometry_audit.json',dict(records=records,seconds=time.perf_counter()-tick,
        estimand='Empirical TRAIN teacher residual; coefficients fitrep0,testrep1 freshlatents;not unseenparent physics',
        scientific_certification=False))
if __name__=='__main__':main()
