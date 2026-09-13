"""R0 tests: source boundary, Hamiltonian derivatives, geometry and SDE stages."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from scripts.core.literature_config import load_runtime, read_document, atomic_scales, UREG
from scripts.core.reduced_llg import BondHamiltonian, ReducedLLG
from scripts.literature.nishino_miyashita_2015.model import FreeMoments, case_alpha
from scripts.literature.nishino_miyashita_2015.run import simulate, write_raw
from scripts.literature.nishino_miyashita_2015.analyze import equivalent, holm

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'conf/literature/nishino_miyashita_2015.yaml'


@pytest.mark.parametrize('path',sorted((ROOT/'conf/literature').glob('*.yaml')),ids=lambda x:x.stem)
def test_conversion_and_runtime_boundary(path):
    read_document(path)
    runtime=load_runtime(path)
    assert not hasattr(runtime,'source_parameters')
    assert set(vars(runtime))=={'reduced','numerics'}


def test_conversion_rejects_wrong_reduced_value(tmp_path):
    doc=yaml.safe_load(CONFIG.read_text());doc['reduced']['field_h']=3.
    path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError,match='independently converted'): read_document(path)


def test_physical_scale_roundtrip_and_dimensions():
    result=atomic_scales({'value':1.,'unit':'meV'},{'value':2.,'unit':'mu_B'},
                         {'value':1.76e11,'unit':'1/(second*tesla)'},{'value':3.,'unit':'angstrom'})
    mu=(2*UREG.mu_B).m_as('joule/tesla')
    assert result['field_T']*mu==pytest.approx(result['energy_J'],rel=1e-13)
    assert result['time_s']*1.76e11*result['field_T']==pytest.approx(1.,rel=1e-13)
    with pytest.raises(Exception):
        atomic_scales({'value':1.,'unit':'second'},{'value':2.,'unit':'mu_B'},
                      {'value':1.76e11,'unit':'1/(second*tesla)'},{'value':3.,'unit':'angstrom'})


@pytest.mark.parametrize('bonds',[[[0,1],[1,2]],[[0,1],[1,2],[2,0]],
    [[0,1],[1,2],[2,0],[3,4],[4,5],[5,3],[0,3],[1,4],[2,5]]])
def test_field_autograd_bond_count_and_boundaries(bonds):
    sites=max(max(x) for x in bonds)+1
    s=torch.randn(2,sites,3,dtype=torch.float64,requires_grad=True)
    model=BondHamiltonian(bonds,.7,.13,(.2,-.1,.3))
    derivative=torch.autograd.grad(model.energy(s).sum(),s)[0]
    torch.testing.assert_close(model.field(s),-derivative,atol=1e-10,rtol=1e-10)
    up=torch.zeros_like(s);up[...,2]=1
    assert model.energy(up)[0].item()==pytest.approx(-.7*len(bonds)-(.13+.3)*sites)
    assert len(set(tuple(sorted(x)) for x in bonds))==len(bonds)
    with pytest.raises(ValueError): BondHamiltonian([[0,1],[1,0]],1.)


def fixture(alpha=.05,theta=2.):
    runtime=load_runtime(CONFIG)
    model=FreeMoments(runtime.reduced)
    solver=ReducedLLG(model,alpha=alpha,theta=theta)
    state=torch.tensor([[[.6,0.,.8]]],dtype=torch.float64)
    return model,solver,state


def test_free_field_derivative_and_case_b():
    model,solver,s=fixture()
    s.requires_grad_(True)
    torch.testing.assert_close(model.field(s),-torch.autograd.grad(model.energy(s).sum(),s)[0])
    cfg=load_runtime(CONFIG)
    for theta in [.5,2.,6.]:
        assert case_alpha(cfg.reduced,'B',theta)*theta==pytest.approx(1.)


@pytest.mark.parametrize('method',['paper_midpoint','heun'])
def test_same_wiener_stages_and_preprojection_errors(method):
    model,solver,s=fixture()
    dt=.005;dw=solver.noise_increment(s,dt,torch.Generator().manual_seed(9))
    first=solver.increment(s,dt,dw)
    predictor=s+(.5 if method=='paper_midpoint' else 1.)*first
    raw=s+(solver.increment(predictor,dt,dw) if method=='paper_midpoint' else .5*(first+solver.increment(predictor,dt,dw)))
    result,errors=solver.step(s,dt,dw,method=method)
    torch.testing.assert_close(result,raw/raw.norm(dim=-1,keepdim=True),atol=1e-14,rtol=1e-14)
    assert errors[0].item()==pytest.approx((predictor.norm(dim=-1)-1).abs().max().item())
    assert errors[1].item()==pytest.approx((raw.norm(dim=-1)-1).abs().max().item())
    assert errors.max()>0


def test_midpoint_geometric_residual_no_silent_fallback():
    _,solver,s=fixture()
    dw=solver.noise_increment(s,.005,torch.Generator().manual_seed(3))
    result,_=solver.step(s,.005,dw,method='geometric_midpoint')
    assert (result.norm(dim=-1)-1).abs().max()<1e-10
    assert (result-s-solver.increment((s+result)/2,.005,dw)).abs().max()<1e-12
    with pytest.raises(RuntimeError):
        solver.step(s,.005,dw,method='geometric_midpoint',max_iterations=1)


def test_brownian_aggregation_and_bauer_convention():
    model,gilbert,s=fixture()
    fine=torch.stack([gilbert.noise_increment(s,.001,torch.Generator().manual_seed(i)) for i in range(4)])
    torch.testing.assert_close(fine.sum(0),fine.reshape(2,2,*s.shape).sum(1).sum(0))
    bauer=ReducedLLG(model,alpha=.05,theta=2.,equation_convention='bauer_ll')
    dw=fine.sum(0)
    expected=-torch.linalg.cross(s,.004*model.field(s)+np.sqrt(.2)*dw)-.05*.004*torch.linalg.cross(s,torch.linalg.cross(s,model.field(s)))
    torch.testing.assert_close(bauer.increment(s,.004,dw),expected)
    assert not torch.allclose(gilbert.increment(s,.004,dw),expected)


def test_deterministic_energy_and_precession_convergence():
    errors=[]
    for dt in [.02,.01,.005]:
        model,solver,s=fixture(alpha=0.,theta=0.)
        for _ in range(round(1/dt)): s,_=solver.step(s,dt,torch.zeros_like(s))
        exact=torch.tensor([[[.6*np.cos(2),.6*np.sin(2),.8]]],dtype=torch.float64)
        errors.append((s-exact).norm().item())
    assert errors[1]<errors[0]/3 and errors[2]<errors[1]/3
    model,solver,s=fixture(alpha=.3,theta=0.)
    energies=[model.energy(s).item()]
    for _ in range(100):
        s,_=solver.step(s,.005,torch.zeros_like(s))
        energies.append(model.energy(s).item())
        assert (s.norm(dim=-1)-1).abs().max()<1e-10
    assert np.max(np.diff(energies))<=1e-12


def test_equivalence_not_just_mean_and_holm():
    assert not equivalent(np.array([-1.,1.]),0.,.01)['passed']
    assert equivalent(np.zeros(100),0.,.01)['passed']
    assert holm([.01,.04,.03])==pytest.approx([.03,.06,.06])


def test_smoke_raw_is_not_formal_and_no_overwrite(tmp_path):
    cfg=load_runtime(CONFIG)
    meta,results,tau,spins=simulate(cfg,'A',0,0,steps=16,moments=4)
    assert not meta['formal_settings']
    path=tmp_path/'task.h5';write_raw(path,meta,results,tau,spins)
    with pytest.raises(FileExistsError): write_raw(path,meta,results,tau,spins)
    assert spins.shape==(9,4,3)
