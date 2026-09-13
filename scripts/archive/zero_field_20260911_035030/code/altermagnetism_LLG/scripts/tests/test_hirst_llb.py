import numpy as np
import torch
from scripts.literature.hirst_mn2au_2022.llb import AFMLLB
from scripts.literature.hirst_mn2au_2022.model import Mn2Au,PinnedEndLLG
from scripts.literature.hirst_mn2au_2022.wall_analysis import fit_wall
from scripts.tests.test_remaining_literature import params


def test_transverse_llb_conserves_macrospin_length_not_unit_length():
    r=params('hirst_mn2au_2022')
    model=AFMLLB(r,theta=.1,theta_n=.27449,me=.8,lam=.01)
    m=torch.tensor([[[.6,0.,.2],[-.6,0.,-.2]]],dtype=torch.float64)
    before=m.norm(dim=-1)
    for _ in range(30): m=model.step(m,.002)
    torch.testing.assert_close(m.norm(dim=-1),before,atol=1e-14,rtol=1e-12)


def test_longitudinal_llb_relaxes_towards_me():
    r=params('hirst_mn2au_2022')
    model=AFMLLB(r,theta=.1,theta_n=.27449,me=.8,lam=.01,
                 chi_parallel=.2,enhancement_z=6.,enhancement_exponent=1.)
    for magnitude in (.5,1.):
        m=torch.tensor([[[magnitude,0.,0.],[-magnitude,0.,0.]]],dtype=torch.float64)
        derivative=model.rhs(m)[0,0,0]
        assert float(derivative)*(.8-magnitude)>0
    m=torch.tensor([[[.8,0.,0.],[-.8,0.,0.]]],dtype=torch.float64)
    torch.testing.assert_close(model.rhs(m),torch.zeros_like(m),atol=1e-14,rtol=0)


def test_asd_wall_end_spins_stay_fixed_in_thermal_heun():
    model=Mn2Au(params('hirst_mn2au_2022'),(4,2,2),(False,True,True))
    s=model.ground_state(); s[:,-1]*=-1
    solver=PinnedEndLLG(model,alpha=1.,theta=.1)
    dw=solver.noise_increment(s,.01,torch.Generator().manual_seed(8))
    after,_=solver.step(s,.01,dw,method='heun_projected_predictor')
    torch.testing.assert_close(after[:,0],s[:,0]); torch.testing.assert_close(after[:,-1],s[:,-1])


def test_tanh_parameter_is_not_neel_width():
    x=np.arange(100.); signs=np.array([1.,-1.,-1.,1.]); offsets=np.array([0.,0.,.5,.5])
    s=np.zeros((1,100,2,2,4,3))
    s[...,0]=-np.tanh((x[:,None]+offsets[None,:]-49.5)/7.)[None,:,None,None,:]*signs
    result=fit_wall(s)[0]
    assert abs(result['delta0_over_a']-7.)<.02
    assert result['neel_width_over_a']==np.pi*result['delta0_over_a']
