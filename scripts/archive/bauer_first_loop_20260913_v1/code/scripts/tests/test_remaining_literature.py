from pathlib import Path
import numpy as np
import pytest
import torch
from scripts.core.literature_config import load_runtime
from scripts.core.reduced_llg import ReducedLLG
from scripts.literature.hirst_mn2au_2022.model import Mn2Au
from scripts.literature.gomonay_2024.model import DoubleLayer, dispersion
from scripts.literature.laliena_crnb3s6_2020.model import ChiralChain, CurrentLLG, solve_profile, soliton, profile_spins

ROOT=Path(__file__).resolve().parents[2]


def params(paper):
    return load_runtime(ROOT/'conf/literature'/f'{paper}.yaml').reduced


@pytest.mark.parametrize('periodic',[True,False])
@pytest.mark.parametrize('kind',['hirst','gomonay100','gomonay110','laliena'])
def test_field_is_negative_energy_gradient(kind,periodic):
    if kind=='hirst':
        model=Mn2Au(params('hirst_mn2au_2022'),(3,3,3),(periodic,True,periodic))
        shape=(2,3,3,3,4,3)
    elif kind.startswith('gomonay'):
        model=DoubleLayer(params('gomonay_2024'),(4,5),wall=True,
                          orientation=kind[-3:],periodic=(periodic,True))
        shape=(2,4,5,model.basis,3)
    else:
        model=ChiralChain(params('laliena_crnb3s6_2020'),dx=.1,periodic=periodic)
        shape=(2,13,3)
    s=torch.randn(shape,dtype=torch.float64,requires_grad=True)
    gradient=torch.autograd.grad(model.energy(s).sum(),s)[0]
    torch.testing.assert_close(model.field(s),-gradient,atol=1e-10,rtol=1e-12)


def test_hirst_neighbors_sums_and_ground_state():
    r=params('hirst_mn2au_2022'); model=Mn2Au(r,(3,3,3))
    counts=np.zeros((4,4),dtype=int)
    for a,b,d,j in model.templates:
        col=min(range(4),key=lambda k:abs(j-r[f'J{k+1}']))
        counts[a,col]+=1; counts[b,col]+=1
    np.testing.assert_array_equal(counts,np.tile([4,4,4,1],(4,1)))
    s=model.ground_state()
    exchange=4*r['J2']-4*r['J1']-4*r['J3']-r['J4']
    torch.testing.assert_close(model.field(s),s*(exchange+2*r['d_x']))
    assert model.energy(s)<model.energy(torch.ones_like(s)*s.new_tensor([1.,0.,0.]))
    for axis in (1,2,3):
        random=torch.randn_like(s)
        torch.testing.assert_close(model.energy(random),model.energy(random.roll(1,axis)))


def test_projected_heun_uses_same_noise_and_normalizes_predictor():
    model=Mn2Au(params('hirst_mn2au_2022'),(2,2,2))
    s=model.ground_state(); llg=ReducedLLG(model,alpha=1.,theta=.1)
    dw=llg.noise_increment(s,.001,torch.Generator().manual_seed(2))
    first=llg.increment(s,.001,dw); pred=s+first; pred/=pred.norm(dim=-1,keepdim=True)
    expected=s+.5*(first+llg.increment(pred,.001,dw)); expected/=expected.norm(dim=-1,keepdim=True)
    got,errors=llg.step(s,.001,dw,method='heun_projected_predictor')
    torch.testing.assert_close(got,expected)
    torch.testing.assert_close(errors[0],((s+first).norm(dim=-1)-1).abs().max())
    assert errors[0]>1e-8
    assert (pred.norm(dim=-1)-1).abs().max()<1e-14


def test_gomonay_branches_and_rotated_bulk():
    r=params('gomonay_2024'); k=np.linspace(-np.pi,np.pi,33)
    plus,minus=dispersion(k,.7,r); rot=dispersion(-.7,k,r)
    np.testing.assert_allclose(plus-minus,-(rot[0]-rot[1]),atol=1e-14)
    p,m=dispersion(k,0.,r); np.testing.assert_allclose(p,m)
    p,m=dispersion(k,.7,dict(r,J_tilde=0.)); np.testing.assert_allclose(p,m)
    a=DoubleLayer(r,(4,4)); b=DoubleLayer(r,(4,4),orientation='110')
    torch.testing.assert_close(a.energy(a.ground_state())/32,b.energy(b.ground_state())/64)
    wall=DoubleLayer(r,(16,3),wall=True,orientation='110',periodic=(False,True))
    s=wall.wall_state(velocity=.05)
    assert torch.isfinite(s).all()
    assert (s.norm(dim=-1)-1).abs().max()<1e-14


def test_laliena_corrected_bvp_zero_and_finite_current():
    r=params('laliena_crnb3s6_2020')
    result=solve_profile(r,0.)
    x=np.linspace(0,15,151)
    np.testing.assert_allclose(result.sol(x),soliton(x,r['h_y']),atol=2e-6)
    for gamma in np.arange(.1,.9,.1):
        result=solve_profile(r,float(gamma),previous=result)
    result=solve_profile(r,.89,previous=result)
    assert result.rms_residuals.max()<1e-6
    spins=profile_spins(np.linspace(-15,15,301),result)
    np.testing.assert_allclose(np.linalg.norm(spins,axis=-1),1.,atol=1e-14)


def test_laliena_equal_alpha_beta_is_advection():
    r=params('laliena_crnb3s6_2020'); model=ChiralChain(r,dx=.1)
    s=torch.randn((1,20,3),dtype=torch.float64); s/=s.norm(dim=-1,keepdim=True)
    a=.03; u=.1; dt=.001; dw=torch.zeros_like(s)
    base=ReducedLLG(model,alpha=a,theta=0.)
    driven=CurrentLLG(model,alpha=a,beta=a,u=u)
    torch.testing.assert_close(driven.increment(s,dt,dw)-base.increment(s,dt,dw),
                               -dt*u*model.derivative(s),atol=1e-14,rtol=1e-10)
