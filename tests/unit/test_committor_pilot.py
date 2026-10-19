import copy
import json
import numpy as np
import pytest
import yaml
from tests.experiments import committor_pilot_analysis as a


def test_competing_arrivals_and_censoring():
    t=np.arange(5)*.1
    mz=np.array([[0,-.9,-.9,.9,.9],[0,.9,.9,-.9,-.9],[0,.1,0,-.1,0],[0,-.9,0,-.9,0]])
    labels,tau=a.classify(mz,t,-.8,.8,.1)
    np.testing.assert_array_equal(labels,[-1,1,0,0])
    np.testing.assert_allclose(tau[:2],[.2,.2])
    s=a.probabilities(labels,tau,.4)
    assert (s['A'],s['B'],s['unresolved'])==(1,1,2)
    assert s['eventual_core_probability_bounds']==[.25,.75]
    assert s['restricted_mean_first_arrival']==pytest.approx(.3)
    assert a.probabilities(labels,tau,.1)['unresolved']==4


def test_initial_core_and_invalid_data():
    labels,tau=a.classify(np.array([[1.,1.],[-1.,-1.]]),np.array([0.,.1]),-.8,.8,0)
    np.testing.assert_array_equal(labels,[1,-1]);np.testing.assert_array_equal(tau,[0,0])
    with pytest.raises(ValueError):a.classify(np.array([[0.,np.nan]]),np.array([0.,.1]),-.8,.8,0)
    assert a.core_proposals(np.ones(10),[.1])==[]
    assert a.wilson(0,128)[1]>0 and a.wilson(128,128)[0]<1


def test_complete_analysis_with_synthetic_paths(tmp_path,monkeypatch):
    cfg=yaml.safe_load(a.Path('tests/conf/committor_pilot.yaml').read_text())
    rows=[dict(kind='wall',theta=cfg['theta'][i//6]) for i in range(12)]
    manifest=dict(rows=rows);(tmp_path/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(a,'OUT',tmp_path);monkeypatch.setattr(a,'load',lambda:(manifest,copy.deepcopy(cfg)))
    monkeypatch.setenv('SLURM_JOB_ID','synthetic-analysis-only')
    def read(name):
        calibration=name.startswith('calibration')
        t=np.arange(2001)*.1 if calibration else np.arange(801)*.01
        p=np.zeros((4,len(t),25,3))
        z=np.full((4,len(t)),.9) if calibration else np.tile(np.minimum(t/2,1),(4,1))
        z[2:]*=-1;p[:,:,:,2]=z[:,:,None];p[:,:,:,0]=np.sqrt(1-z*z)[:,:,None]
        return p,t,dict(sha256='synthetic')
    monkeypatch.setattr(a,'read',read)
    a.run()
    result=json.loads((tmp_path/'analysis.json').read_text())
    assert result['status']=='COMPLETE_EXPLORATORY_NOT_CERTIFIED'
    assert len(result['rows'])==12*6*3*3
    assert len(list((tmp_path/'figures').glob('*.png')))==14
    queue=json.loads((tmp_path/'augmentation_priority.json').read_text())
    assert all(r['score']==0 for r in queue['rows'])
