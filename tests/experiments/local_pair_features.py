"""Data-derived local dot features; no coupling/field/target-statistic input.

Only a representation hypothesis, not an established remedy. Retains the
Cartesian backbone, so this does not establish full rotational equivariance.
"""
import torch
from torch import nn
from tests.experiments.velocity_controls import CartesianPathNet


def open_pair_features(spins):
    """[B,3,F,L] -> left/right dot and existence masks [B,4,F,L]."""
    if spins.ndim!=4 or spins.shape[1]!=3 or spins.shape[-1]<2:
        raise ValueError('expected [B,3,F,L], L>=2')
    dot=(spins[...,:-1]*spins[...,1:]).sum(1,keepdim=True)
    zero=torch.zeros_like(dot[...,:1]); one=torch.ones_like(dot)
    return torch.cat((torch.cat((zero,dot),-1),torch.cat((dot,zero),-1),
                      torch.cat((zero,one),-1),torch.cat((one,zero),-1)),1)


class PairFeatureInput(nn.Conv2d):
    def __init__(self,width):
        super().__init__(27,width,1)
        self.pair_features=nn.Conv2d(4,width,1,bias=False)
        nn.init.zeros_(self.pair_features.weight)

    def forward(self,features):
        return super().forward(features)+self.pair_features(open_pair_features(features[:,:3]))


class LocalPairPathNet(CartesianPathNet):
    def __init__(self,width=64,blocks=4):
        super().__init__(width=width,blocks=blocks)
        original=self.input
        self.input=PairFeatureInput(width)
        self.input.load_state_dict(original.state_dict(),strict=False)
