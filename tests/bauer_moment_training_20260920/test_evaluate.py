import pytest
from tests.bauer_moment_training_20260920.evaluate import identity


def test_all_paired_identities():
    assert [identity(i) for i in range(6)] == ['w0_11','w0_22','w0_33','w1_11','w1_22','w1_33']
    with pytest.raises(ValueError): identity(6)
