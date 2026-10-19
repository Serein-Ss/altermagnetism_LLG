import math
from pathlib import Path
import pytest
import torch
import yaml
from scipy import constants
from scripts.core.reduced_llg import ReducedLLG
from tests.literature_preflight_20260917.models import build, parameters

CONFIG=yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())


@pytest.mark.parametrize('name,shape',[('bauer',(6,)),('miltat',(4,3)),('hirst',(3,3,3))])
def test_energy_gradient(name,shape):
    model,_,_=build(CONFIG,name,shape=shape)
    s=torch.randn((2,*shape,model.basis,3),dtype=torch.float64,requires_grad=True)
    gradient=torch.autograd.grad(model.energy(s).sum(),s)[0]
    torch.testing.assert_close(model.field(s),-gradient,atol=1e-9,rtol=1e-11)


def test_open_dipoles_against_direct_all_pairs():
    model,_,_=build(CONFIG,'miltat',shape=(4,3))
    s=torch.randn((2,4,3,2,3),dtype=torch.float64)
    p=model.positions(s).reshape(-1,3)
    dr=p[:,None]-p[None,:]
    rr=dr.norm(dim=-1)
    inv=torch.where(rr>0,1/rr.clamp_min(1e-20),0.)
    kernel=model.reduced['dipole']*(3*dr[..., :,None]*dr[...,None,:]*inv[...,None,None]**5
                                  -torch.eye(3)*inv[...,None,None]**3)
    expected=torch.einsum('ijcd,bjd->bic',kernel,s.reshape(2,-1,3)).reshape_as(s)
    torch.testing.assert_close(model.dipole_field(s),expected,atol=1e-12,rtol=1e-11)


def test_triangular_geometry_and_coordination():
    model,_,_=build(CONFIG,'miltat',shape=(6,5))
    assert len(model.templates)==6
    counts=torch.zeros((1,6,5,2,3),dtype=torch.float64)
    from scripts.core.cell_hamiltonian import shifted
    ones=torch.ones((1,6,5,3),dtype=torch.float64)
    for a,b,d,_ in model.templates:
        counts[...,a,:]+=shifted(ones,d,model.periodic)
        counts[...,b,:]+=shifted(ones,tuple(-v for v in d),model.periodic)
    assert torch.all(counts[:,1:-1,1:-1]==6)
    assert counts.min()<6


def test_bauer_barrier_and_si_time():
    model,_,scales=build(CONFIG,'bauer',shape=(6,))
    ez=model.energy(model.uniform([1],axis=2))
    ex=model.energy(model.uniform([1],axis=0))
    ey=model.energy(model.uniform([1],axis=1))
    torch.testing.assert_close((ex-ez)/6,torch.tensor([.8/50],dtype=torch.float64))
    torch.testing.assert_close((ey-ez)/6,torch.tensor([1/50],dtype=torch.float64))
    assert math.isclose(scales['time_s'],constants.hbar/(50e-3*constants.e),rel_tol=1e-12)


def test_published_hirst_not_preprint_and_counts():
    r,scales=parameters(CONFIG,'hirst')
    assert math.isclose(r['d_z'],-.0663/30.804,rel_tol=1e-12)
    assert math.isclose(r['d_x'],.0026/30.804,rel_tol=1e-12)
    assert scales['time_s']>0
    m,_,_=build(CONFIG,'hirst',shape=(3,3,3))
    counts=[[0]*4 for _ in range(4)]
    for a,b,d,j in m.templates:
        k=min(range(4),key=lambda k:abs(j-r[f'J{k+1}']))
        counts[a][k]+=1; counts[b][k]+=1
    assert counts==[[4,4,4,1]]*4


@pytest.mark.parametrize('name,shape',[('bauer',(6,)),('miltat',(4,3)),('hirst',(3,3,3))])
def test_short_heun_and_norm(name,shape):
    model,r,_=build(CONFIG,name,shape=shape)
    s=model.ground_state() if name=='hirst' else model.uniform([1]*model.basis)
    llg=ReducedLLG(model,alpha=r['alpha'],theta=r['theta'],equation_convention='bauer_ll' if name=='bauer' else 'gilbert')
    g=torch.Generator().manual_seed(7103)
    for _ in range(4):
        s,errors=llg.step(s,1e-4,llg.noise_increment(s,1e-4,g),method='heun')
    assert torch.isfinite(s).all()
    assert (s.norm(dim=-1)-1).abs().max()<1e-12


def test_no_training_release_from_preflight():
    assert CONFIG['production_enabled'] is False
    assert not CONFIG['gamma_convention']['strict_physical_time_certified']
    assert all(row['certification_holds'] for row in CONFIG['systems'].values())
