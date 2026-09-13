"""New source-independent field, RNG, recovery and real-shard contract tests."""
from pathlib import Path
import json
import h5py
import numpy as np
import pytest
import torch
import yaml
from scripts.core.reduced_llg import ReducedLLG,BondHamiltonian
from scripts.core.streaming_llg import simulate_resumable
from scripts.core.literature_config import load_runtime
from scripts.literature.gomonay_2024.model import DoubleLayer
from scripts.generation.generate_zero_field_gomonay import run,validate_config
from scripts.datasets.zero_field_paths import ZeroFieldPaths,audit_group_splits

ROOT=Path(__file__).resolve().parents[2]


def source_energy(s,r,periodic):
    # Supplement S1 with both orientations explicitly summed and halves for
    # intra-layer bonds; independent of CellHamiltonian templates/shifts.
    energy=-r['K_DW']*s[...,2].square().flatten(1).sum(1)
    nx,ny=s.shape[1:3]
    def at(x,y,a):
        if (not periodic[0] and not 0<=x<nx) or (not periodic[1] and not 0<=y<ny): return None
        return s[:,x%nx,y%ny,a]
    for x in range(nx):
        for y in range(ny):
            for dx,dy in ((0,0),(-1,0),(0,-1),(-1,-1)):
                b=at(x+dx,y+dy,1)
                if b is not None: energy=energy+r['J1']*(s[:,x,y,0]*b).sum(-1)
            for a,sign in ((0,1),(1,-1)):
                for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)):
                    b=at(x+dx,y+dy,a)
                    if b is None: continue
                    coefficient=-r['J2']/2 if dx*dy==0 else -sign*r['J_tilde']*dx*dy/2
                    energy=energy+coefficient*(s[:,x,y,a]*b).sum(-1)
    return energy


@pytest.mark.parametrize('periodic',[(True,True),(False,False),(False,True),(True,False)])
def test_source_s1_field(periodic):
    r=load_runtime(ROOT/'conf/literature/gomonay_2024.yaml').reduced
    m=DoubleLayer(r,(4,5),wall=True,periodic=periodic)
    g=torch.Generator().manual_seed(123)
    s=torch.randn((2,4,5,2,3),dtype=torch.float64,generator=g,requires_grad=True)
    energy=source_energy(s,r,periodic)
    gradient=torch.autograd.grad(energy.sum(),s)[0]
    torch.testing.assert_close(m.field(s),-gradient,rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(m.energy(s),energy,rtol=1e-10,atol=1e-10)
    state=m.ground_state()
    torch.testing.assert_close(m.energy(state),m.energy(-state),rtol=0,atol=1e-12)
    assert torch.linalg.cross(state,m.field(state)).abs().max()==0


def test_projected_predictor_diagnostic_and_shared_noise():
    llg=ReducedLLG(BondHamiltonian([],[],field=(1.,0.,0.)),alpha=.1,theta=.3)
    s=torch.tensor([[[0.,0.,1.]]],dtype=torch.float64)
    dw=torch.ones_like(s)*.2
    expected=(s+llg.increment(s,.01,dw)).norm(dim=-1).sub(1).abs().max()
    used=[];increment=llg.increment
    def record(state,dt,noise): used.append(noise.data_ptr());return increment(state,dt,noise)
    llg.increment=record
    _,errors=llg.step(s,.01,dw,method='heun_projected_predictor')
    assert errors[0]>1e-5
    assert errors[0]==expected
    assert used==[dw.data_ptr(),dw.data_ptr()]


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('method',['heun_projected_predictor','geometric_midpoint'])
def test_exact_resume_and_identity_rejection(tmp_path,method,device):
    import os
    if device=='cuda' and (not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available()):
        pytest.skip('requires allocated CUDA')
    llg=ReducedLLG(BondHamiltonian([[0,1]],1.,anisotropy=.1),alpha=.1,theta=.3)
    s=torch.tensor([[[0.,0.,1.],[0.,0.,-1.]]],dtype=torch.float64,device=device).repeat(2,1,1)
    p=dict(dt=.005,steps=19,save_every=3,checkpoint_every=5,seed=91,method=method)
    expected,done=simulate_resumable(tmp_path/'a.h5',s,llg,p,identity='fixture')
    _,done=simulate_resumable(tmp_path/'b.h5',s,llg,p,identity='fixture',stop_after=8)
    assert not done
    with pytest.raises(ValueError,match='identity'):
        simulate_resumable(tmp_path/'b.h5',s,llg,p,identity='changed')
    actual,done=simulate_resumable(tmp_path/'b.h5',s,llg,p,identity='fixture')
    assert done and torch.equal(expected,actual)
    with h5py.File(tmp_path/'a.h5') as a,h5py.File(tmp_path/'b.h5') as b:
        for key in ('spins','energy','raw_norm_errors','time','mask','max_norm_error'):
            np.testing.assert_array_equal(a[key][:],b[key][:])


@pytest.mark.parametrize('orientation,basis',[('100',2),('110',4)])
def test_real_llg_shard_loader_and_no_training(tmp_path,orientation,basis):
    c=yaml.safe_load((ROOT/'conf/zero_field_gomonay/software_validation.yaml').read_text())
    c.update(steps=5,save_every=2,orientation=orientation,material_config=str(ROOT/'conf/literature/gomonay_2024.yaml'))
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(c))
    path=tmp_path/'shard.h5';run(config,path)
    with pytest.raises(ValueError,match='diagnostic'): ZeroFieldPaths([path])
    dataset=ZeroFieldPaths([path],allow_diagnostic=True);row=dataset[0]
    assert len(dataset)==4 and row['spins'].shape==(4,4,4,basis,3)
    assert torch.equal(row['spins'][0],row['initial_state'])
    assert row['time'][-1]==.025 and row['sample_weight']==1
    assert row['mask'].all() and row['geometry']['basis']==basis
    c['drives']['sot']=1
    with pytest.raises(ValueError,match='zero'): validate_config(c)
    c['drives']['sot']=0;c['role']='production'
    with pytest.raises(RuntimeError,match='blocked'): validate_config(c)


def test_split_chain_and_initial_state_leakage():
    rows=[dict(parent_chain_id='a',initial_state_id='x',split='train'),
          dict(parent_chain_id='a',initial_state_id='y',split='test')]
    with pytest.raises(ValueError,match='leakage'): audit_group_splits(rows)


def test_independent_mc_matrix_matches_source_energy():
    from scripts.validation.gomonay_static_reference import coupling_matrix
    r=load_runtime(ROOT/'conf/literature/gomonay_2024.yaml').reduced
    s=torch.randn((2,4,4,2,3),dtype=torch.float64)
    matrix=coupling_matrix(4,r);flat=s.numpy().reshape(2,32,3)
    e=-.5*np.einsum('bia,ij,bja->b',flat,matrix,flat)-r['K_DW']*(flat[...,2]**2).sum(1)
    np.testing.assert_allclose(e,source_energy(s,r,(True,True)).numpy(),rtol=1e-12,atol=1e-12)
    assert np.array_equal(matrix,matrix.T)


def test_event_censor_return_disorder_and_recurrence():
    from scripts.analysis.zero_field_events import classify
    c=dict(status='pass',independent_equilibrium_sha256='synthetic_test_not_physical_certificate',
        positive_nz_min=.7,negative_nz_max=-.7,dwell_reduced_time=2.,order_min=.6,structure_max=.4)
    t=np.arange(10.);o=np.ones(10);q=np.zeros(10)
    a=classify(t,np.ones(10),o,q,c)
    assert a['outcome']=='no_crossing' and not a['event_observed'] and a['censor_time']==9
    z=np.ones(10);z[2]=-.1
    assert classify(t,z,o,q,c)['outcome']=='crossed_returned'
    z[2:6]=-1
    b=classify(t,z,o,q,c)
    assert b['event_observed'] and len(b['recurrent_events'])==2
    assert b['first_committed_switch_time']==4
    o[-1]=.1
    assert classify(t,z,o,q,c)['outcome']=='critical_disordering'


def test_certificate_rejects_boolean_and_changed_evidence(tmp_path):
    from scripts.validation.zero_field_certificates import verify
    from scripts.core.literature_config import sha256
    evidence=tmp_path/'raw';evidence.write_text('test')
    cert=tmp_path/'cert.json'
    c=dict(scope='fixture',bindings={},status='pass',metrics=[],evidence_sha256={})
    cert.write_text(json.dumps(c))
    with pytest.raises(ValueError): verify(cert,'fixture',{})
    c.update(metrics=[dict(value=.1,lower=0.,upper=.2)],evidence_sha256={str(evidence):sha256(evidence)})
    cert.write_text(json.dumps(c));verify(cert,'fixture',{})
    evidence.write_text('changed')
    with pytest.raises(ValueError): verify(cert,'fixture',{})


def test_arviz_diagnostics_do_not_certify_shifted_chains():
    from scripts.analysis.zero_field_equilibrium import diagnostics
    g=np.random.default_rng(20260911)
    x=g.normal(size=(4,4000))
    good=diagnostics(x,1.)
    assert good['rank_folded_rhat']<1.01 and good['bulk_ess']>1000
    x[0]+=2
    assert diagnostics(x,1.)['rank_folded_rhat']>1.1

