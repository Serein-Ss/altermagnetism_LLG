import h5py
import numpy as np
import pytest
import torch
from scripts.core.reduced_llg import ReducedLLG
from scripts.literature.workflow import simulate,require_complete,rk4_step
from scripts.literature.bauer_2011.model import OpenChain
from scripts.literature.bauer_2011.analyze import reversals,kaplan_meier,restricted_mean,statistics
from scripts.literature.bauer_2011.fit import fit_scaling
from scripts.literature.gomonay_2024.model import DoubleLayer,dispersion
from scripts.literature.gomonay_2024.analyze import spectral_paths,wall_observables
from scripts.literature.laliena_crnb3s6_2020.model import continue_branch,ChiralChain,CurrentLLG,solve_profile,profile_spins
from scripts.tests.test_remaining_literature import params


def test_survival_ties_censoring_and_dwell():
    time=np.arange(9.)
    m=np.array([1.,-.9,.1,-.9,-.9,.9,.9,.9,.9])
    np.testing.assert_array_equal(reversals(time,m,.8,1.),[4.,6.])
    t,s=kaplan_meier([1.,1.,2.],[True,False,True])
    np.testing.assert_allclose(s,[1.,2/3,0.])
    assert restricted_mean([1.,1.,2.],[True,False,True],2.)==pytest.approx(5/3)
    stats=statistics(time,np.stack((m,np.ones_like(m)),1),threshold=.8,dwell=1.,bootstrap=50)
    assert stats['right_censored']==1 and stats['completed_events']==2
    assert stats['unrestricted_mean_lifetime'] is None


def test_scaling_rejects_rmst_and_recovers_synthetic_barrier():
    rows=[dict(theta=t,length=100,lifetime=np.exp(2+.7/t),standard_error=.01*np.exp(2+.7/t),
               estimator='unrestricted_mean_lifetime',reference_source='synthetic_unit_test') for t in (.1,.12,.14,.16)]
    result=fit_scaling(rows)
    assert result['slope']==pytest.approx(.7)
    rows[0]['estimator']='RMST'
    with pytest.raises(ValueError): fit_scaling(rows)


def test_hdf5_includes_all_trajectories_and_final_state(tmp_path):
    model=OpenChain(params('bauer_2011'))
    llg=ReducedLLG(model,alpha=.1,theta=.11,equation_convention='bauer_ll')
    s=torch.zeros((3,5,3),dtype=torch.float64); s[...,2]=1
    p=dict(dt=.001,steps=7,save_every=3,seed=10,method='heun')
    result=simulate(tmp_path/'all.h5',s,llg,p)
    with h5py.File(tmp_path/'all.h5') as h:
        require_complete(h)
        assert h['spins'].shape==(4,3,5,3)
        np.testing.assert_allclose(h['time'][:],[0,.003,.006,.007])
        np.testing.assert_array_equal(h['spins'][-1],result.numpy())
    with pytest.raises(FileExistsError): simulate(tmp_path/'all.h5',s,llg,p)
    with h5py.File(tmp_path/'partial.h5','w') as h:
        h.attrs['complete']=False
        with pytest.raises(ValueError): require_complete(h)


def test_gomonay_nonlinear_field_linearization_matches_paper_dispersion():
    r=params('gomonay_2024'); shape=(8,8); model=DoubleLayer(r,shape)
    s=model.ground_state().to(torch.complex128); b0=model.field(s)
    for ix,iy in ((1,0),(1,1),(1,7),(2,3)):
        kx,ky=2*np.pi*ix/8,2*np.pi*iy/8
        x,y=torch.meshgrid(torch.arange(8),torch.arange(8),indexing='ij')
        wave=torch.exp(1j*(kx*x.double()+ky*y.double()))
        matrix=np.zeros((4,4),complex)
        for col in range(4):
            ds=torch.zeros_like(s); ds[0,:,:,col//2,col%2]=wave
            rhs=-torch.linalg.cross(ds,b0)-torch.linalg.cross(s,model.field(ds))
            for row in range(4):
                matrix[row,col]=((rhs[0,:,:,row//2,row%2]/wave).mean()).item()
        numeric=np.sort(np.abs(np.linalg.eigvals(matrix).imag))
        p,m=dispersion(kx,ky,r)
        np.testing.assert_allclose(numeric,np.sort([p,p,m,m]),atol=1e-10)


def test_corrected_bvp_matches_independent_driven_llg_to_second_order():
    r=params('laliena_crnb3s6_2020'); result=None
    for gamma in np.linspace(0,.89,10): result=solve_profile(r,float(gamma),previous=result)
    errors=[]
    for dx in (.1,.05):
        x=np.arange(-12,12,dx); s=torch.tensor(profile_spins(x,result))[None]
        model=ChiralChain(r,dx=dx); llg=CurrentLLG(model,alpha=.01,beta=.02,u=.89)
        residual=llg.increment(s,1.,torch.zeros_like(s))+1.78*model.derivative(s)
        errors.append(float(residual[:,20:-20].abs().max()))
    assert 3.5<errors[0]/errors[1]<4.5


def test_arclength_continuation_has_small_residual():
    branch=continue_branch(params('laliena_crnb3s6_2020'),steps=3,points=101)
    assert len(branch['gamma'])==5
    assert np.max(branch['residual'])<1.01e-6
    assert np.all(np.diff(branch['gamma'])>0)


def test_wall_pipeline_uses_physical_sublattice_offsets(tmp_path):
    model=DoubleLayer(params('gomonay_2024'),(20,4),wall=True,orientation='110',periodic=(False,True))
    llg=ReducedLLG(model,alpha=.75,theta=0.)
    p=dict(dt=.001,steps=4,save_every=1,seed=1,method='rk4',orientation='110')
    simulate(tmp_path/'wall.h5',model.wall_state(),llg,p,step_fn=lambda s,t,dt,dw:rk4_step(llg,s,t,dt))
    with h5py.File(tmp_path/'wall.h5') as h:
        result=wall_observables(h)
        assert len(result['observables'])==5
        assert np.isfinite(result['velocity']).all()
