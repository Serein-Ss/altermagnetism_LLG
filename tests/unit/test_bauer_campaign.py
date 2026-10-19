import numpy as np
import h5py
import pytest
from scripts.workflow.bauer_campaign import run_reference, observables, event_rows, seed, initial_states


def test_observables_open_ground_state():
    s=np.zeros((2,4,5,3));s[...,2]=1
    obs=observables(s)
    np.testing.assert_allclose(obs[...,0],-.9)
    np.testing.assert_allclose(obs[...,3:7],1)
    np.testing.assert_array_equal(obs[...,7],0)


def test_reversal_confirmation_and_right_censoring():
    time=np.arange(11,dtype=float)
    mz=np.array([[1,1,-1,-1,-1,-1,1,1,1,1,1],np.ones(11)])
    rows,events=event_rows(mz,time,.6,2)
    np.testing.assert_array_equal(events[0],[4.,8.])
    np.testing.assert_allclose(rows[0,:3],[1,1,.4])
    np.testing.assert_allclose(rows[1,:3],[0,0,1])
    np.testing.assert_array_equal(rows[1,3:],1)


def test_initial_source_families_disjoint():
    a=initial_states(4,7,'train');b=initial_states(4,7,'test')
    assert not np.array_equal(a,b)
    np.testing.assert_array_equal(a,initial_states(4,7,'train'))
    assert len({seed(split,i) for split in ('train','dev','test') for i in range(100)})==300


def test_reference_resume_keeps_identical_frames(tmp_path):
    case=dict(id='tiny',L=5,theta=.11,dt=.01,paths=2,duration=1.,save_dt=.1)
    cfg=dict(reference_cases=[case])
    run_reference(cfg,tmp_path,0)
    path=tmp_path/'reference/tiny/observables.h5'
    with h5py.File(path) as h:before=h['values'][:];assert h.attrs['complete']
    # Simulate a crash after raw frames were written but before the checkpoint advanced.
    initial=np.zeros((2,5,3));initial[...,2]=1
    np.savez(tmp_path/'reference/tiny/checkpoint.npz',state=initial,next_block=0)
    with h5py.File(path,'r+') as h:h.attrs['complete']=False
    run_reference(cfg,tmp_path,0)
    with h5py.File(path) as h:
        np.testing.assert_array_equal(h['values'][:],before)
        assert h['spins'].shape==(2,11,5,3)
    changed=dict(case,theta=.12)
    with pytest.raises(ValueError,match='identity mismatch'):
        run_reference(dict(reference_cases=[changed]),tmp_path,0)


def test_static_wall_and_open_chain_modes():
    from scripts.validation.bauer_physics_checks import static_checks
    rows=static_checks()
    assert all(r['status']=='PASS' for r in rows),rows


def test_analysis_and_report_smoke(tmp_path):
    from scripts.workflow.bauer_campaign import analyze_reference, pilot, gate, report, save
    case=dict(id='original_dt002',L=5,theta=.11,dt=.01,paths=2,duration=1.,save_dt=.1)
    cfg=dict(reference_cases=[case],bootstrap=2000,model=dict(L=5,train_theta=[.11,.13],
        pilot_initials=2,pilot_noise=2,pilot_horizon=2.,pilot_save_dt=.1,candidate_windows=[1.,2.]))
    save(tmp_path/'config.json',cfg)
    run_reference(cfg,tmp_path,0);analyze_reference(cfg,tmp_path);pilot(cfg,tmp_path);gate(cfg,tmp_path);report(cfg,tmp_path)
    import json
    r=json.loads((tmp_path/'final_report.json').read_text())
    assert r['status']=='HOLD'
    assert not r['results']['release']['training_released']
    assert (tmp_path/'figures/pilot_events.png').exists()
