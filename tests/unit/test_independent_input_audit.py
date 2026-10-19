import copy
import numpy as np
import pytest
from tests.experiments.independent_input_audit import validate_generated_report,validate_spins


def test_incomplete_or_duplicate_reports_rejected():
    cfg=dict(theta=[.11,.13],n_per_half=128,generated_sampling_steps=1024)
    report=dict(status='COMPLETE_GENERATED_ENSEMBLE_NOT_CERTIFIED',training_seed=11,
        rows=[dict(family=i,theta=t,paths=128,sampling_steps=1024) for i in range(4) for t in cfg['theta']])
    validate_generated_report(report,cfg,11)
    for failure in ('running','missing','duplicate','count','steps'):
        bad=copy.deepcopy(report)
        if failure=='running':bad['status']='RUNNING'
        elif failure=='missing':bad['rows'].pop()
        elif failure=='duplicate':bad['rows'][-1]=bad['rows'][0]
        elif failure=='count':bad['rows'][0]['paths']=16
        else:bad['rows'][0]['sampling_steps']=128
        with pytest.raises(ValueError):validate_generated_report(bad,cfg,11)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_spin_shape_anchor_and_norm(dtype):
    initial=np.zeros((3,3));initial[:,2]=1
    a=np.broadcast_to(initial,(2,4,3,3)).astype(dtype).copy()
    validate_spins(a,initial,2,4)
    for failure in ('shape','initial','norm','nan'):
        b=a.copy()
        if failure=='shape':b=b[:1]
        elif failure=='initial':b[0,0]*=-1
        elif failure=='norm':b[0,1]*=2
        else:b[0,1,0,0]=np.nan
        with pytest.raises(ValueError):validate_spins(b,initial,2,4)
