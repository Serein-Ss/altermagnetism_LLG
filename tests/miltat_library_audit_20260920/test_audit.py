import json
import h5py
import numpy as np
import pytest
from tests.miltat_library_audit_20260920.run import inspect_parent


def test_all_outcomes_retained_and_seed_mismatch_rejected(tmp_path):
    path=tmp_path/'parent.h5'
    row=dict(parent_id='p',split='test',cell_shape=[2,2],preparation_seed=1,noise_seeds=[2,3])
    cfg=dict(dt_fs=.3125,save_ps=.1,cold_duration_ps=50.,parent_preparation_ps=20.,future_duration_ps=.1,temperature_K=12.5,alpha=.5,branches_per_parent=2)
    with h5py.File(path,'w') as h:
        h.attrs['preparation_seed']=1;h['noise_seeds']=[2,3]
        h.attrs['config_json']=json.dumps(cfg)
        h['spins']=np.zeros((2,2,2,2,2,3));h['time_ps']=[0.,.1]
        h.attrs['initial_core_json']=json.dumps(dict(status='ISOLATED_POSITIVE_CORE'))
        h.attrs['final_core_json']=json.dumps([dict(status='NO_POSITIVE_CORE'),dict(status='ISOLATED_POSITIVE_CORE')])
    result=inspect_parent(path,row,cfg)
    assert result['final_status_counts']==dict(NO_POSITIVE_CORE=1,ISOLATED_POSITIVE_CORE=1)
    with h5py.File(path,'r+') as h:h['noise_seeds'][1]=4
    with pytest.raises(ValueError,match='Noise seeds'):inspect_parent(path,row,cfg)
