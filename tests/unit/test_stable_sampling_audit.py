import numpy as np
import pytest
from tests.experiments.stable_sampling_audit import paired_errors


def test_paired_errors():
    a=np.zeros((2,3,4,3));b=a.copy();b[:,:,:,2]=1
    assert paired_errors(a,a)==dict(path_rmse=0.,endpoint_mz_rmse=0.)
    assert paired_errors(a,b)['endpoint_mz_rmse']==1
    assert paired_errors(a,b)['path_rmse']==pytest.approx(np.sqrt(1/3))
    with pytest.raises(ValueError):paired_errors(a,b[:1])
    b[0,0,0,0]=np.nan
    with pytest.raises(ValueError):paired_errors(a,b)
