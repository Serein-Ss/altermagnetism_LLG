import numpy as np
import pytest
import torch
from tests.experiments.stable_independent_sampling import sample_family


class ZeroField:
    def __call__(self,x,*args,**kwargs):return torch.zeros_like(x)


def test_replay_paired_sources_and_initial_anchor():
    initial=np.zeros((5,3));initial[:,2]=1
    cfg=dict(n_per_half=4,generated_latent_base=850000000,damping=.1,generated_sampling_steps=2)
    a,seeds=sample_family(ZeroField(),initial,.11,cfg,0,2,'cpu')
    b,other=sample_family(ZeroField(),initial,.13,cfg,0,2,'cpu')
    np.testing.assert_array_equal(a,b)
    assert a.shape==(4,201,5,3) and seeds==other==[850000000,850000001]
    np.testing.assert_array_equal(a[:,0],np.repeat(initial[None],4,axis=0))
    assert not np.array_equal(a[0,1:],a[1,1:])
    c,_=sample_family(ZeroField(),initial,.11,cfg,1,2,'cpu')
    assert not np.array_equal(a,c)
    with pytest.raises(ValueError):sample_family(ZeroField(),initial,.11,cfg,0,3,'cpu')
