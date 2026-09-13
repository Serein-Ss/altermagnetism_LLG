import h5py
import numpy as np
import pytest
from scripts.analysis.bauer_revision4_audit import audit


def test_audit_missing_is_hold(tmp_path):
    result = audit(tmp_path/'missing.h5', tmp_path/'out')
    assert result['status'] == 'HOLD_missing_completed_reference'
    assert result['numerical_failure_is_censoring'] is False


def test_audit_no_invented_reversal(tmp_path):
    path = tmp_path/'trajectory.h5'
    x = np.zeros((12, 2, 5, 3)); x[..., 2] = 1
    x[6:, 0, :, 2] = -1
    with h5py.File(path, 'x') as h:
        h.attrs.update(complete=True, dt=.01)
        h['spins'] = x; h['time'] = np.arange(12.)
    report = audit(path, tmp_path/'out')
    assert report['zero_crossing_counts'] == [1, 0]
    assert report['committed_switch'] is None
    assert report['maximum_norm_error'] == 0
    with np.load(tmp_path/'out/observables.npz') as data:
        assert data['magnetization'].shape == (12, 2, 3)
        assert np.allclose(data['energy_per_spin'], -.9)


def test_incomplete_not_physical_censoring(tmp_path):
    path = tmp_path/'trajectory.h5'
    with h5py.File(path,'x') as h: h.attrs['complete'] = False
    with pytest.raises(ValueError): audit(path, tmp_path/'out')
