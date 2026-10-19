import numpy as np
import pytest
from tests.experiments.reflection_augmentation_analysis import compare
from tests.experiments.verify_reflection_augmentation_analysis import verify_paired_draws


def test_analysis_to_verifier_actual_archive(tmp_path):
    rng=np.random.default_rng(44)
    data=[rng.normal(size=(8,3,4,3)) for _ in range(3)]
    data=[p/np.linalg.norm(p,axis=-1,keepdims=True) for p in data]
    arrays,metrics=compare(data,rng,20)
    assert set(metrics)=={'control','augmentation','paired_change'}
    p=tmp_path/'draws.npz'; np.savez(p,**arrays)
    with np.load(p) as z:
        for key in metrics['paired_change']: verify_paired_draws(z,key)
    arrays['paired_change_mean_rmse']+=1
    with pytest.raises(AssertionError): verify_paired_draws(arrays,'mean_rmse')
