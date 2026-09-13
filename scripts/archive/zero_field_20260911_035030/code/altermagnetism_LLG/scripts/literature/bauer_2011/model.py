"""Bauer Eq.(1)/(3): unit spins, open nearest-neighbour chain, uniaxial K."""
import torch


class OpenChain:
    def __init__(self,reduced):
        if reduced['equation_convention']!='bauer_ll':
            raise ValueError('Bauer uses its LL convention, not Gilbert')
        self.exchange=float(reduced['exchange'])
        self.anisotropy=float(reduced['anisotropy'])

    def energy(self,s):
        return -self.exchange*(s[:,:-1]*s[:,1:]).sum((-1,-2))-self.anisotropy*s[...,2].square().sum(-1)

    def field(self,s):
        b=torch.zeros_like(s)
        b[:,:-1]+=self.exchange*s[:,1:]
        b[:,1:]+=self.exchange*s[:,:-1]
        b[...,2]+=2*self.anisotropy*s[...,2]
        return b
