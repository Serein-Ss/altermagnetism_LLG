import pytest
import json
import shutil
import numpy as np
from pathlib import Path
from scripts.workflow.bauer_campaign import digest
from tests.experiments import independent_moment_analysis as module
from tests.experiments.independent_moment_analysis import select_records


def test_selection_requires_every_registered_model_and_control():
    rows=[dict(kind='reference',case=dict(family=0,theta=.11,role='primary'))]
    rows += [dict(kind='reference',case=dict(family=0,theta=.11,role='numerical_control',method=m)) for m in ('mt','heun')]
    rows += [dict(kind='generated',family=0,theta=.11,seed=s) for s in (11,22,33)]
    primary,models,controls=select_records(rows,0,.11)
    assert len(models)==3 and len(controls)==2
    with pytest.raises(ValueError):select_records(rows[:-1],0,.11)
    with pytest.raises(ValueError):select_records(rows[1:],0,.11)
    with pytest.raises(ValueError):select_records(rows,0,.13)


def test_small_end_to_end_analysis_preserves_all_models(tmp_path,monkeypatch):
    source_root=Path(module.__file__).resolve().parents[2]
    for name in ('independent_comparison_statistics.py','trajectory_bootstrap.py','path_observables.py'):
        target=tmp_path/'tests/experiments'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source_root/'tests/experiments'/name,target)
    monkeypatch.chdir(tmp_path)
    n=3;rng=np.random.default_rng(37)
    def paths(count):
        a=rng.normal(size=(count,201,25,3));a/=np.linalg.norm(a,axis=-1,keepdims=True);a[:,0]=[0,0,1]
        return a
    records=[]
    np.savez_compressed('reference.npz',spins=paths(2*n))
    records.append(dict(kind='reference',case=dict(family=1,theta=.11,role='primary'),path='reference.npz',sha256=digest('reference.npz')))
    for seed in (11,22,33):
        path=f'model_{seed}.npz';np.savez_compressed(path,spins=paths(n).astype('f4'))
        records.append(dict(kind='generated',family=1,theta=.11,seed=seed,path=path,sha256=digest(path)))
    manifest=Path('tests/results/stable_independent_reference/manifest.json');manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(dict(config=dict(theta=[.11,.13],n_per_half=n,save_dt=80.))))
    audit=Path('tests/results/independent_input_audit/summary.json');audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps(dict(status='COMPLETE_INPUT_INTEGRITY_ONLY',manifest_sha256=digest(manifest),records=records)))
    original=module.compare_ensembles
    monkeypatch.setattr(module,'compare_ensembles',lambda *a,**kw:original(*a,**kw,replicates=2))
    module.run(2)
    out=Path('tests/results/independent_moment_analysis/family_1_T0.11')
    report=json.loads((out/'summary.json').read_text())
    assert report['status']=='COMPLETE_CONDITIONAL_MOMENTS_NOT_CERTIFIED'
    assert {r['comparison'] for r in report['rows']}=={'model_11','model_22','model_33'}
    for seed in (11,22,33):
        with np.load(out/f'model_{seed}_bootstrap.npz') as d:
            assert d['frame200/magnetization_mean/excess_rmse'].shape==(2,)
    with pytest.raises(FileExistsError):module.run(2)
