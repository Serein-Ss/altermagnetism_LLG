"""Experimental flow-time weighting, with unchanged interpolation and RNG order."""
import torch
from scripts.model.gate_f import path_interpolate
from tests.experiments.conditional_source import conditional_source


def weighted_flow_loss(net,y,condition,time,graph,axis,generator,*,power=2,source_scale=None):
    if power not in (0,2):raise ValueError('only uniform or quadratic weighting supported')
    tau=torch.rand(len(y),device=y.device,dtype=y.dtype,generator=generator)
    source=conditional_source(y[:,0],y.shape[1],generator,source_scale)
    x,target,diagnostics=path_interpolate(source,y,tau,generator)
    error=net(x,tau,y[:,0],condition,time,graph,axis)-target
    per_path=error[:,1:].square().sum(-1).flatten(1).mean(1)
    weight=(power+1)*(1-tau).pow(power)
    return (weight*per_path).mean(),diagnostics
