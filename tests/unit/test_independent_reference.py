import numpy as np
import pytest
from tests.experiments.independent_reference import generate,validate_inputs


def setup():
    states=np.zeros((2,5,3));states[...,2]=1
    return states,np.array([810010001,810010002]),dict(dt=.01,duration=.1,save_dt=.05,anisotropy=.1,damping=.1,theta=.13,method='mt')


@pytest.mark.parametrize('method',['mt','heun'])
def test_reference_reproducible_and_batch_independent(method):
    s,seeds,cfg=setup();cfg['method']=method
    a,manifest=generate(s,seeds,**cfg)
    b,_=generate(s[::-1].copy(),seeds[::-1].copy(),**cfg)
    np.testing.assert_array_equal(a,b[::-1])
    assert a.shape==(2,3,5,3)
    np.testing.assert_array_equal(a[:,0],s)
    assert not np.array_equal(a[0,1:],a[1,1:])
    assert manifest['max_saved_norm_error']<1e-10
    assert manifest['noise_seeds']==seeds.tolist()


@pytest.mark.parametrize('case',['duplicate','forbidden','float_seed','negative_seed','nonunit','nan','grid','method'])
def test_invalid_reference_contract_rejected(case):
    s,seeds,cfg=setup()
    if case=='duplicate':seeds[1]=seeds[0]
    elif case=='forbidden':cfg['forbidden_seeds']=[seeds[0]]
    elif case=='float_seed':seeds=seeds.astype(float)
    elif case=='negative_seed':seeds[0]=-1
    elif case=='nonunit':s[0,0]*=2
    elif case=='nan':cfg['theta']=np.nan
    elif case=='grid':cfg['duration']=.103
    elif case=='method':cfg['method']='rk4'
    with pytest.raises(ValueError):validate_inputs(s,seeds,**cfg)
