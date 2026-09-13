"""Reduced double-layer Hamiltonian, official supplement Eq.S1; LLG Eq.S44.

The published S8 is the linear dynamical matrix, S9 its dispersion, not
the nonlinear Hamiltonian. Basis offset (1/2,1/2) is retained explicitly.
"""
import math
import numpy as np
import torch
from scripts.core.cell_hamiltonian import CellHamiltonian


class DoubleLayer(CellHamiltonian):
    def __init__(self, reduced, shape=(500,500), *, wall=False,
                 periodic=(True,True), orientation='100'):
        j1,j2,jt = (reduced[k] for k in ('J1','J2','J_tilde'))
        templates = [(0,1,(x,y),-j1) for x,y in ((0,0),(-1,0),(0,-1),(-1,-1))]
        for a,sign in ((0,1),(1,-1)):
            templates += [(a,a,d,j2) for d in ((1,0),(0,1))]
            templates += [(a,a,(1,-1),-sign*jt),(a,a,(1,1),sign*jt)]
        self.orientation = orientation
        if orientation == '110':
            # Exact determinant-two supercell: e1=(1,1), e2=(-1,1).
            # Representatives (0,0),(1,0); never impose a tilted wall on
            # a rectangular periodic seam that does not match the lattice.
            transformed=[]
            for r in (0,1):
                for a,b,(dx,dy),j in templates:
                    x,y = r+dx,dy
                    rb = (x+y)%2
                    d = ((x-rb+y)//2,(y-x+rb)//2)
                    transformed.append((2*r+a,2*rb+b,d,j))
            templates=transformed
            basis=4
        elif orientation == '100':
            basis=2
        else:
            raise ValueError('orientation must be 100 or 110')
        self.reduced=dict(reduced)
        self.k = reduced['K_DW' if wall else 'K_SW']
        super().__init__(shape,basis,templates,periodic=periodic,anisotropy=(0.,0.,self.k))

    def coordinates(self, *, device='cpu', dtype=torch.float64):
        x,y=torch.meshgrid(torch.arange(self.shape[0],device=device,dtype=dtype),
                           torch.arange(self.shape[1],device=device,dtype=dtype),indexing='ij')
        if self.orientation == '100':
            origin=torch.stack((x,y),-1)[...,None,:]
            offsets=origin.new_tensor(((0.,0.),(.5,.5)))
        else:
            origin=torch.stack((x-y,x+y),-1)[...,None,:]
            offsets=origin.new_tensor(((0.,0.),(.5,.5),(1.,0.),(1.5,.5)))
        return origin+offsets

    def ground_state(self, **kwargs):
        return self.uniform([1.,-1.]*(self.basis//2),**kwargs)

    def wall_state(self, *, velocity=0., phase=-math.pi/2, **kwargs):
        """S26/S28/S31 and S19b; approximate traveling-wave INITIAL condition.

        velocity is in a0/t0. Subsequent motion is unforced, alpha=0.
        Valid for small canting and deformation; normalized per lattice site.
        """
        s=self.ground_state(**kwargs)
        coords=self.coordinates(device=s.device,dtype=s.dtype)
        xi=coords[...,0] if self.orientation=='100' else coords.sum(-1)/math.sqrt(2)
        xi=xi-(xi.max()+xi.min())/2
        j1,j2,jt=(self.reduced[k] for k in ('J1','J2','J_tilde'))
        c=2*math.sqrt(j1*(j1+2*j2))
        if self.k<=0 or abs(velocity)>=c:
            raise ValueError('positive wall anisotropy and |velocity|<c required')
        ell=math.sqrt((j1/2+j2)/(2*self.k))
        width=ell*math.sqrt(1-(velocity/c)**2)
        lam=4*jt if self.orientation=='110' else 0.
        eps=lam*4*math.sqrt(j1*self.k)/c**2
        sigma=eps*(velocity/c)/(1-(velocity/c)**2)**1.5
        # Autograd in the spatial coordinate supplies independent continuum
        # derivatives for S19b; dynamics itself uses the discrete Hamiltonian.
        with torch.enable_grad():
            q=xi.detach().requires_grad_(True)
            u=q/width
            phi=phase+sigma*torch.logaddexp(u,-u)-sigma*math.log(2)
            sech=2*torch.exp(-torch.logaddexp(u,-u))
            n=torch.stack((sech*phi.cos(),sech*phi.sin(),-u.tanh()),-1)
            dn=torch.stack([torch.autograd.grad(n[...,a].sum(),q,create_graph=True,retain_graph=True)[0] for a in range(3)],-1)
            d2n=torch.stack([torch.autograd.grad(dn[...,a].sum(),q,retain_graph=True)[0] for a in range(3)],-1)
            mixed=.5*d2n if self.orientation=='110' else torch.zeros_like(d2n)
            m=torch.linalg.cross(-velocity*dn+lam*torch.linalg.cross(n,mixed),n)/(8*j1)
            if m.norm(dim=-1).max() >= .3:
                raise ValueError('continuum initial state exceeds small-canting regime')
            spins=n[None]*s[...,2,None]+m[None]
        return (spins/spins.norm(dim=-1,keepdim=True)).detach()


def dispersion(kx,ky,reduced, *, wall=False):
    """S7b/S9, branch labels preserve signed splitting under C4 rotation."""
    j1,j2,jt=(reduced[k] for k in ('J1','J2','J_tilde'))
    k=reduced['K_DW' if wall else 'K_SW']
    a=np.cos(np.asarray(kx)/2)*np.cos(np.asarray(ky)/2)
    b=1+k/(2*j1)+(j2/j1)*(np.sin(np.asarray(kx)/2)**2+np.sin(np.asarray(ky)/2)**2)
    c=(jt/j1)*np.sin(kx)*np.sin(ky)
    base=np.sqrt(np.maximum(b*b-a*a,0.))
    return 4*j1*(base+c),4*j1*(base-c)

