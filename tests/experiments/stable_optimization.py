"""Explicit cosine learning rate and EMA state for experimental training."""
import math
import torch


def cosine_lr(step,total,start,end):
    if not 1<=step<=total or not 0<end<=start:raise ValueError('invalid schedule')
    progress=(step-1)/max(1,total-1)
    return end+(start-end)*.5*(1+math.cos(math.pi*progress))


@torch.no_grad()
def update_ema(average,current,decay):
    if not 0<=decay<1:raise ValueError('EMA decay must be in [0,1)')
    for key,value in current.items():
        if value.is_floating_point():average[key].lerp_(value,1-decay)
        else:average[key].copy_(value)
