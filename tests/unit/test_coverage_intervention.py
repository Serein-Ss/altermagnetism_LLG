import json
from collections import Counter
import numpy as np
from tests.experiments.coverage_intervention_data import initial_bank,CONFIG
from tests.experiments.coverage_transition_dataset import replacement_index


def test_bank_independent_unit_states():
    cfg=json.loads(CONFIG.read_text()); a,rows=initial_bank(cfg); b,_=initial_bank(cfg)
    np.testing.assert_array_equal(a,b)
    assert a.shape==(12,25,3) and len(rows)==12
    np.testing.assert_allclose(np.linalg.norm(a,axis=-1),1,atol=1e-14)
    assert len({s.tobytes() for s in a})==12
    assert a[0,:,2].mean()>0 and a[1,:,2].mean()<0


def test_fixed_count_unique_replacement_mapping():
    mapped=[replacement_index(i) for i in range(12288)]
    kept=[i for i,x in enumerate(mapped) if x is None]
    replaced=[x for x in mapped if x is not None]
    assert len(kept)==len(replaced)==6144 and len(set(replaced))==6144
    assert Counter(x[0] for x in replaced)==Counter({a:256 for a in range(24)})
    for i,x in enumerate(mapped):
        if x is not None: assert (i//2048)//3==x[0]//12
