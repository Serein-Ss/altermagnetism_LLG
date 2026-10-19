"""TRAIN-only fixed transport probes; not evidence of physical accuracy."""
import math
import torch
from tests.temporal_source_repair.source import draw
from tests.conditional_teacher_repair.target import labels

def rate(arm,step,total):
    peak=3e-4 if arm.endswith('high') else 3e-5
    return peak*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/max(1,total-1))))

def optimizer(net,state,arm):
    opt=torch.optim.AdamW(net.parameters(),lr=rate(arm,1,512),weight_decay=1e-5)
    if arm.startswith('restore'):opt.load_state_dict(state)
    return opt

@torch.no_grad()
def risk(net,bank,smoke=False):
    training=net.training;net.eval();result=[]
    for sid,b in enumerate(bank[:2] if smoke else bank):
        g=torch.Generator(device=b.device).manual_seed(956900+sid)
        y=b[:8];z=draw(y[:,0],2,g,0.)
        tau=torch.linspace(.05,.85,8,device=b.device)
        x,v,c,_=labels(z,y,tau,b,g)
        pred=net.velocity(x,tau,x[:,0],x.new_tensor([0.,.1]))
        values=[float((pred[:,1:]-t[:,1:]).square().sum(-1).mean()) for t in (c,v)]
        assert all(math.isfinite(v) for v in values)
        result.append(dict(state=sid,conditional_risk=values[0],sampled_risk=values[1]))
    net.train(training)
    return result
