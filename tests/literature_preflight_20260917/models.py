"""Test-only physical parameter conversion and open triangular Co/Pt model."""
import itertools
import math
import torch
from scipy import constants
from scripts.core.cell_hamiltonian import CellHamiltonian, shifted
from scripts.core.literature_config import UREG, quantity, atomic_scales
from scripts.literature.hirst_mn2au_2022.model import Mn2Au


def parameters(config, name):
    row = config['systems'][name]
    p = row['physics']
    energy = p['J'] if name != 'hirst' else {'value': 30.804, 'unit': 'mRy'}
    e0 = quantity(energy)
    if name == 'bauer':
        time_s = (constants.hbar * UREG.joule * UREG.second / e0).m_as('second')
        scales = {'energy_J': e0.m_as('joule'), 'time_s': time_s}
    else:
        scales = atomic_scales(energy, p['moment'], config['gamma_convention'], p['a'])
    reduced = {'alpha': p['alpha'], 'theta': (constants.Boltzmann * UREG.joule / UREG.kelvin
                * p['temperature_K'] * UREG.kelvin / e0).m_as('dimensionless')}
    keys = {'bauer': ['barrier_low', 'barrier_high'], 'miltat': ['D', 'K'],
            'hirst': ['J1', 'J2', 'J3', 'J4', 'd_z', 'd_x']}[name]
    reduced.update({k: (quantity(p[k])/e0).m_as('dimensionless') for k in keys})
    if name == 'miltat':
        # mu0*mu_s^2/(4*pi*a^3*E0); every pair is retained, no demag approximation.
        mu0 = constants.mu_0 * UREG.newton / UREG.ampere**2
        reduced['dipole'] = (mu0 * quantity(p['moment'])**2 /
                            (4*math.pi*quantity(p['a'])**3*e0)).m_as('dimensionless')
        reduced['a_nm'] = quantity(p['a']).m_as('nanometer')
    return reduced, scales


class TriangularCo(CellHamiltonian):
    """Two-site rectangular cell; free edges; full finite-sample dipolar field."""
    def __init__(self, reduced, shape):
        self.reduced = reduced
        self.offsets = ((0., 0., 0.), (.5, math.sqrt(3)/2, 0.))
        templates, dm = [], []
        for a,b in itertools.product(range(2), repeat=2):
            for dx,dy in itertools.product(range(-1,2), repeat=2):
                if (a,b,dx,dy) <= (b,a,-dx,-dy):
                    continue
                r = (dx+self.offsets[b][0]-self.offsets[a][0],
                     math.sqrt(3)*dy+self.offsets[b][1]-self.offsets[a][1], 0.)
                if abs(sum(x*x for x in r)-1) < 1e-12:
                    templates.append((a,b,(dx,dy),1.))
                    dm.append(tuple(reduced['D']*x for x in (-r[1], r[0], 0.)))
        super().__init__(shape, 2, templates, periodic=(False,False),
                         anisotropy=(0.,0.,reduced['K']))
        self.dm = dm
        self._kernel = None

    def positions(self, s):
        x,y = torch.meshgrid(torch.arange(self.shape[0],device=s.device,dtype=s.dtype),
                             torch.arange(self.shape[1],device=s.device,dtype=s.dtype),indexing='ij')
        return torch.stack((x,math.sqrt(3)*y,torch.zeros_like(x)),-1)[...,None,:]+s.new_tensor(self.offsets)

    def dipole_field(self, s):
        nx,ny = self.shape
        px,py = 2*nx-1,2*ny-1
        cache_key = (s.device, s.dtype)
        if self._kernel is None or self._kernel[0] != cache_key:
            xx,yy = torch.meshgrid(torch.arange(px,device=s.device),torch.arange(py,device=s.device),indexing='ij')
            xx = torch.where(xx < nx,xx,xx-px).to(s.dtype)
            yy = torch.where(yy < ny,yy,yy-py).to(s.dtype)
            kernel = torch.zeros((2,2,3,3,px,py),device=s.device,dtype=s.dtype)
            for a,b in itertools.product(range(2),repeat=2):
                r = torch.stack((xx+self.offsets[a][0]-self.offsets[b][0],
                                 math.sqrt(3)*yy+self.offsets[a][1]-self.offsets[b][1],torch.zeros_like(xx)),0)
                r2 = r.square().sum(0)
                mask = r2 > 0
                inv = torch.where(mask,r2.clamp_min(1e-30).rsqrt(),0.)
                tensor = 3*r[:,None]*r[None,:]*inv**5-torch.eye(3,device=s.device,dtype=s.dtype)[:,:,None,None]*inv**3
                kernel[a,b] = self.reduced['dipole']*tensor
            self._kernel = (cache_key, torch.fft.rfft2(kernel))
        spectrum = torch.fft.rfft2(s.permute(0,3,4,1,2),s=(px,py))
        field = torch.fft.irfft2(torch.einsum('abcdxy,nbdxy->nacxy',self._kernel[1],spectrum),s=(px,py))
        return field[...,:nx,:ny].permute(0,3,4,1,2)

    def energy(self, s):
        e = super().energy(s)-.5*(s*self.dipole_field(s)).flatten(1).sum(1)
        for (a,b,d,_),v in zip(self.templates,self.dm):
            e -= (torch.linalg.cross(s[...,a,:],shifted(s[...,b,:],d,self.periodic))*s.new_tensor(v)).flatten(1).sum(1)
        return e

    def field(self, s):
        f = super().field(s)+self.dipole_field(s)
        for (a,b,d,_),v in zip(self.templates,self.dm):
            vec=s.new_tensor(v)
            f[...,a,:] += torch.linalg.cross(shifted(s[...,b,:],d,self.periodic),vec.expand_as(s[...,a,:]))
            f[...,b,:] += torch.linalg.cross(vec.expand_as(s[...,b,:]),shifted(s[...,a,:],tuple(-x for x in d),self.periodic))
        return f

    def initial_texture(self, s, radius_nm, width_nm):
        pos = self.positions(s)
        center = pos.reshape(-1,3).mean(0)
        delta = (pos-center)*self.reduced['a_nm']
        r = delta.norm(dim=-1)
        angle = math.pi*.5*(1+torch.tanh((r-radius_nm)/width_nm))
        radial = delta/r.clamp_min(1e-12)[...,None]
        radial[...,2] = 0
        candidates=[]
        for chirality in (-1,1):
            trial=chirality*angle.sin()[...,None]*radial
            trial[...,2]=angle.cos()
            trial=trial/trial.norm(dim=-1,keepdim=True)
            candidates.append(trial[None].expand_as(s).clone())
        return min(candidates,key=lambda x:self.energy(x).mean().item())


def build(config, name, *, shape=None):
    r,scales=parameters(config,name)
    row=config['systems'][name]
    shape=tuple(shape or row['geometry']['shape'])
    if name=='bauer':
        # Set K_z=high, K_x=high-low, K_y=0: easy-axis gaps are low and high.
        model=CellHamiltonian(shape,1,[(0,0,(1,),1.)],periodic=(False,),
                              anisotropy=(r['barrier_high']-r['barrier_low'],0.,r['barrier_high']))
    elif name=='hirst':
        model=Mn2Au(r,shape,tuple(row['geometry']['periodic']))
    else:
        model=TriangularCo(r,shape)
    return model,r,scales
