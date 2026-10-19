import numpy as np
import pytest
from tests.experiments.reflection_analysis import compare
from tests.experiments.verify_reflection_analysis import verify_paired_draws


def test_verifier_accepts_actual_reflection_analysis_archive(tmp_path):
    rng=np.random.default_rng(41)
    data=[rng.normal(size=(8,3,4,3)) for _ in range(3)]
    data=[p/np.linalg.norm(p,axis=-1,keepdims=True) for p in data]
    arrays,metrics=compare(data,rng,20)
    path=tmp_path/'reflection.npz'; np.savez(path,**arrays)
    with np.load(path) as z:
        assert not any(k.startswith('coverage_') for k in z.files)
        for key in metrics['paired_change']: verify_paired_draws(z,key)
    arrays['paired_change_mean_rmse']=arrays['paired_change_mean_rmse']+1
    with pytest.raises(AssertionError): verify_paired_draws(arrays,'mean_rmse')
