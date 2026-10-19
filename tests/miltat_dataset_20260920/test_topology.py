import numpy as np
import pytest
from tests.miltat_dataset_20260920.topology import charge, triangles


def test_octant_and_orientation():
    s = np.eye(3)
    assert charge(s, np.array([[0,1,2]]))['Q'] == pytest.approx(1/8)
    assert charge(s, np.array([[0,2,1]]))['Q'] == pytest.approx(-1/8)


def test_smooth_skyrmion_and_spin_rotation():
    x, y = np.meshgrid(np.arange(80), np.arange(80))
    p = np.stack((x+.5*(y%2), y*np.sqrt(3)/2, np.zeros_like(x)), -1).reshape(-1,3)*.251
    tri = triangles(p)
    delta = p-p.mean(0)
    radius = np.linalg.norm(delta[:, :2], axis=-1)
    angle = np.pi/2*(1+np.tanh((radius-2)/.5))
    phi = np.arctan2(delta[:,1], delta[:,0])
    s = np.stack((np.sin(angle)*np.cos(phi), np.sin(angle)*np.sin(phi), np.cos(angle)), -1)
    assert charge(s, tri)['Q'] == pytest.approx(1, abs=1e-6)
    assert charge(s[:, [1,2,0]], tri)['Q'] == pytest.approx(charge(s, tri)['Q'])
    s[:]=[0,0,1]
    assert charge(s, tri)['Q'] == 0


def test_invalid_norm_rejected():
    with pytest.raises(ValueError):
        charge(np.eye(3)*2, np.array([[0,1,2]]))
