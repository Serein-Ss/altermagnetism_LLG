"""Positive site-metric reweighting; same conditional sphere flow targets."""
import torch
from tests.experiments.local_equivariant_transition import source
from scripts.model.gate_f import path_interpolate


def site_weights(sites,*,dtype,device):
    if sites<=6: raise ValueError('requires two three-site ends and nonempty interior')
    weights=torch.full((sites,),sites/(2*(sites-6)),dtype=dtype,device=device)
    weights[:3]=sites/12; weights[-3:]=sites/12
    return weights


def loss(model,initial,target,theta,lag,axis,generator,scale,weights=None):
    z=source(initial,generator,scale)
    tau=torch.rand(initial.shape[0],dtype=initial.dtype,device=initial.device,generator=generator)
    x,velocity,diagnostics=path_interpolate(torch.stack((initial,z),1),torch.stack((initial,target),1),tau,generator)
    predicted=model(x[:,1],tau,initial,theta,lag,axis)
    if weights is None: weights=site_weights(initial.shape[1],dtype=initial.dtype,device=initial.device)
    return ((predicted-velocity[:,1]).square().sum(-1)*weights).mean(),diagnostics


def train_step(model,optimizer,initial,target,theta,lag,axis,generator,scale):
    optimizer.zero_grad(set_to_none=True)
    value,diagnostics=loss(model,initial,target,theta,lag,axis,generator,scale)
    if not torch.isfinite(value): raise ValueError('nonfinite weighted training loss')
    value.backward()
    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    return float(value.detach()),float(grad),diagnostics
