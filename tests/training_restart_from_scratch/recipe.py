"""Matched full optimizer restart, differing only in model initialization."""
import math
import torch
from tests.transition_block_repair.model import TransitionNet
from tests.transport_risk_audit.probe import draft,oriented,scores

def initialize(checkpoint,arm,seed,device='cpu'):
    if arm not in ('warm','fresh'):raise ValueError(arm)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(963000+seed)
        net=TransitionNet()
    if arm=='warm':net.load_state_dict(checkpoint['model'])
    net=net.to(device)
    # Reset BOTH arms. Historical Adam state is intentionally not loaded.
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-5)
    return net,opt

def rate(step,total):
    warmup=8 if total==16 else 200
    if step<=warmup:return 1e-3*step/warmup
    return 3e-5+(1e-3-3e-5)*.5*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))

@torch.no_grad()
def restart_risk(net,bank,smoke=False):
    training=net.training;net.eval();records=[]
    for sid,b in enumerate(bank[:2] if smoke else bank):
        for k in ((0,5,9) if smoke else range(10)):
            x,t,s,c,angle,_=draft(b,sid,0,k)
            x,s,c=oriented(x,s,c,angle,True)
            p=net.velocity(x,t,x[:,0],x.new_tensor([0.,.1]))
            records.append(dict(state=sid,bin=k,values=scores(p,s,c)))
    net.train(training)
    return records
