"""Nishino Fig.1: independent unit moments, h=-field_h sum sz."""
from __future__ import annotations
import math
import torch


class FreeMoments:
    def __init__(self,reduced):
        if reduced['moment'] != 1.:
            raise ValueError('this Fig.1 model requires reduced moment=1')
        self.field_h=float(reduced['field_h'])

    def energy(self,s): return -self.field_h*s[...,2].sum(-1)

    def field(self,s):
        b=torch.zeros_like(s);b[...,2]=self.field_h
        return b

    def exact_mean(self,theta):
        if theta<=0: return 1.
        x=self.field_h/theta
        return 1./math.tanh(x)-1./x


def case_alpha(reduced,case,theta):
    if theta<=0: raise ValueError('Fig.1 temperature must be positive')
    if case=='A': return reduced['alpha_A']
    if case=='B': return reduced['D_B']/theta
    raise ValueError('case must be A or B')
