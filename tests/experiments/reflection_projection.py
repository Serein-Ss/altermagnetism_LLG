"""Index-reversal group projection of a frozen director-equivariant flow."""
from torch import nn


class ReflectionProjection(nn.Module):
    def __init__(self,base):
        super().__init__()
        self.base=base

    def forward(self,x,tau,initial,theta,lag,axis):
        direct=self.base(x,tau,initial,theta,lag,axis)
        reverse=self.base(x.flip(1),tau,initial.flip(1),theta,lag,axis).flip(1)
        return (direct+reverse)/2
