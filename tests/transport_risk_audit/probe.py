"""Balanced empirical-TRAIN transport-risk probes, not physical validation."""
import math
import torch
from tests.temporal_source_repair.source import draw
from tests.conditional_teacher_repair.target import labels
from tests.axial_symmetry_repair.symmetry import rotate

def times(n,bin_index,g,device):
    if not 0<=bin_index<10:raise ValueError('ten transport-time bins required')
    return (bin_index+torch.rand(n,device=device,generator=g))/10

@torch.no_grad()
def draft(bank,sid,rep,bin_index):
    if len(bank)!=64:raise ValueError('all 64 TRAIN branches required')
    # Every branch occurs in every time bin, with fresh latent and jitter.
    g=torch.Generator(device=bank.device).manual_seed(957000+sid*100+rep*10+bin_index)
    y=bank;z=draw(y[:,0],2,g,0.);tau=times(64,bin_index,g,bank.device)
    x,sampled,teacher,diag=labels(z,y,tau,bank,g)
    angle=torch.rand((),device=bank.device,generator=g)*2*math.pi
    return x,tau,sampled,teacher,angle,diag

def oriented(x,sampled,teacher,angle,augmented):
    if not augmented:return x,sampled,teacher
    # Match the training common-angle operation for a same-parent group.
    return rotate(x,angle),rotate(sampled,angle),rotate(teacher,angle)

def scores(pred,sampled,teacher):
    p,s,c=(a[:,1:] for a in (pred,sampled,teacher))
    mse=lambda a:a.square().sum(-1).mean()
    vals=torch.stack((mse(p-c),mse(p-s),mse(s-c),2*((p-c)*(c-s)).sum(-1).mean(),mse(p)))
    if not torch.isfinite(vals).all():raise ValueError('nonfinite transport risk')
    return vals.detach().cpu().tolist()
