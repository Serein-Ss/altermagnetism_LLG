import numpy as np
from tests.experiments.midpoint_mean_resolution import corrected_mean_mse


def test_mean_noise_correction_known_ensembles():
    a = np.zeros((2,2,1,3))
    b = np.zeros_like(a)
    a[:,1,0,0] = [0,2]
    b[:,1,0,0] = [2,4]
    raw, noise, corrected = corrected_mean_mse(a,b)
    np.testing.assert_allclose(raw,[4/3])
    np.testing.assert_allclose(noise,[2/3])
    np.testing.assert_allclose(corrected,[2/3])


def test_negative_correction_retained_and_initial_offsets_removed():
    a = np.zeros((2,2,1,3))
    a[:,1,0,0] = [-1,1]
    raw, noise, corrected = corrected_mean_mse(a,a.copy())
    assert raw[0] == 0 and noise[0] > 0 and corrected[0] < 0
    offset = np.array([2.,3.,4.])[None,None,None,:]
    shifted = corrected_mean_mse(a+offset,a-offset)
    for x,y in zip((raw,noise,corrected),shifted):
        np.testing.assert_allclose(x,y)
