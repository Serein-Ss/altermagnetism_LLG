"""Offline workflow checks, including fail-closed behavior and plot outputs."""
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from scripts.core.literature_config import load_runtime, sha256
from scripts.core.reduced_llg import BondHamiltonian, ReducedLLG
from scripts.literature.nishino_miyashita_2015.run import simulate,write_raw
from scripts.literature.nishino_miyashita_2015.analyze import collect,analyze
from scripts.literature.nishino_miyashita_2015.plot import plot
from scripts.workflow.run_reduced_literature_campaign import freeze,verify_snapshot

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'conf/literature/nishino_miyashita_2015.yaml'


def test_exchange_rotation_and_energy_drift_convergence():
    s=torch.tensor([[[.6,0.,.8],[0.,.8,.6]]],dtype=torch.float64)
    isotropic=BondHamiltonian([[0,1]],.7)
    rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype=torch.float64)
    torch.testing.assert_close(isotropic.energy(s),isotropic.energy(s@rotation))
    model=BondHamiltonian([[0,1]],.7,.1,(.2,-.1,.3))
    solver=ReducedLLG(model,alpha=0.,theta=0.)
    errors=[]
    for dt in [.04,.02,.01]:
        state=s.clone();initial=model.energy(state).item()
        for _ in range(round(1/dt)):
            state,_=solver.step(state,dt,torch.zeros_like(state))
        errors.append(abs(model.energy(state).item()-initial)/abs(initial))
    assert errors[1]<errors[0]/2 and errors[2]<errors[1]/2


def test_freeze_hash_and_no_overwrite(tmp_path):
    from scripts.core.literature_config import read_document
    run_id,path=freeze(read_document(CONFIG),'convergence',tmp_path,'abcdef0123456789','a'*64)
    assert run_id.endswith(sha256(path)[:8])
    assert load_runtime(path).numerics['code_sha256']=='a'*64
    with pytest.raises(FileExistsError):
        freeze(read_document(CONFIG),'convergence',tmp_path,'abcdef0123456789','a'*64)
    with pytest.raises(ValueError,match='frozen source'):
        verify_snapshot({'files':{'scripts/core/reduced_llg.py':'wrong'}})


def test_incomplete_smoke_rejected_and_complete_analysis_plots(tmp_path):
    config=load_runtime(CONFIG)
    config.numerics['paper_settings'].update(moments=4,total_steps=8,burn_steps=4)
    config.numerics['numerical_choice']['save_every']=1
    config.numerics['validation_extension']['repetitions']=1
    config.numerics['code_sha256']='fixture'
    raw=tmp_path/'test_run';raw.mkdir()
    with pytest.raises(ValueError,match='missing task'): collect(raw,config,'primary','fixture')
    for case_idx,case in enumerate(('A','B')):
        for index in range(12):
            # Test fixture only: reduced-size protocol lives in tmp_path.
            meta,results,times,vectors=simulate(config,case,index,0)
            meta.update(config_sha256='fixture',run_id=raw.name,code_sha256='fixture')
            write_raw(raw/f'task_{case_idx*12+index:04d}.h5',meta,results,times,vectors)
    report=analyze(raw,config,'primary','fixture')
    assert report['status']=='fail'
    assert not report['production_enabled']
    files=plot(report,tmp_path/'no_reference',tmp_path/'assets')
    assert len(files)==3
    assert {Path(p).suffix for p in files}=={'.png','.gif'}
