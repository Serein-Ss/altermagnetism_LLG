"""Mn2Au four-Mn conventional-cell ASD, published PRB 106, 094402 Fig.1.

Layer ordering: corner A, corner B, body-offset B, body-offset A.
J1 connects adjacent corner/body layers (four bonds); J4 the paired
corner/corner or body/body layers (one); J3 their four lateral neighbors.
J2 connects the four in-plane neighbors of each Mn. No mean-field shortcut.
Fractional Mn internal z is NOT inferred here: topology suffices for ASD;
metric-dependent comparisons need separately audited site coordinates.
"""
import torch
from scripts.core.cell_hamiltonian import CellHamiltonian
from scripts.core.reduced_llg import ReducedLLG


class Mn2Au(CellHamiltonian):
    signs = (1., -1., -1., 1.)

    def __init__(self, reduced, shape=(30,30,30), periodic=(True,True,True)):
        templates = []
        for a,b in ((0,2),(1,3)):
            for x,y in ((0,0),(-1,0),(0,-1),(-1,-1)):
                templates.append((a,b,(x,y,0),reduced['J1']))
        for a in range(4):
            for d in ((1,0,0),(0,1,0)):
                templates.append((a,a,d,reduced['J2']))
        for a,b,z in ((0,1,0),(2,3,-1)):
            templates.append((a,b,(0,0,z),reduced['J4']))
            for x,y in ((1,0),(-1,0),(0,1),(0,-1)):
                templates.append((a,b,(x,y,z),reduced['J3']))
        super().__init__(shape,4,templates,periodic=periodic,
                         anisotropy=(reduced['d_x'],0.,reduced['d_z']))

    def ground_state(self, **kwargs):
        return self.uniform(self.signs,axis=0,**kwargs)

    def order_parameters(self, s):
        spatial = tuple(range(1,4))
        cells = s.mean(spatial)
        a = cells[:,(0,3)].mean(1)
        b = cells[:,(1,2)].mean(1)
        return {'m_a':a, 'm_b':b, 'neel':(a-b)/2, 'magnetization':(a+b)/2}

    def tilted_state(self, angle, **kwargs):
        """Rigid rotation of both opposite sublattices out of easy x-y plane."""
        s = self.ground_state(**kwargs)
        s[...,2] = s[...,0]*torch.sin(s.new_tensor(angle))
        s[...,0] *= torch.cos(s.new_tensor(angle))
        return s


class PinnedEndLLG(ReducedLLG):
    """Paper domain-wall protocol fixes opposite end layers at every stage."""
    def increment(self,s,dt,dw):
        increment=super().increment(s,dt,dw)
        increment[:,0]=0.; increment[:,-1]=0.
        return increment

