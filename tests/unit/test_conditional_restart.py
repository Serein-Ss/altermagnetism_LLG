import json
import numpy as np
from tests.experiments.conditional_restart import CONFIG, selections, noise_seeds
from tests.experiments.independent_reference import generate


def test_frozen_selection_and_noise_disjointness():
    cfg=json.loads(CONFIG.read_text())
    a=selections(cfg,256)
    assert a==selections(cfg,256) and len(a)==12
    assert all(len(x)==5 and x[0]==(0,0) for x in a)
    for x in a:
        assert len({p for p,f in x[1:]})==2
        assert {f for p,f in x[1:]}=={20,40}
    seeds=np.concatenate([noise_seeds(cfg,c,s,60) for c in range(3) for s in range(60)])
    assert len(seeds)==184320 and len(np.unique(seeds))==len(seeds)
    assert seeds.min()>=1300000000 and seeds.max()<1400000000


def test_reference_short_step_all_cases():
    cfg=json.loads(CONFIG.read_text())
    initial=np.zeros((2,25,3)); initial[:,:,2]=1
    for case in cfg['cases']:
        paths,meta=generate(initial,np.array([101,102]),**case,duration=.01,save_dt=.01,
            anisotropy=.1,damping=.1,theta=.11)
        assert paths.shape==(2,2,25,3)
        np.testing.assert_array_equal(paths[:,0],initial)
        np.testing.assert_allclose(np.linalg.norm(paths,axis=-1),1,atol=1e-10)
