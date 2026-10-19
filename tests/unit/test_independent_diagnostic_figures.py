import numpy as np
import matplotlib.pyplot as plt
from tests.experiments.independent_diagnostic_figures import figure


def test_curves_histogram_and_events():
    mz = np.array([[1, -.2, .1], [1, .2, -.1]])
    mag = np.zeros((2, 3, 3)); mag[..., 2] = mz
    p = dict(magnetization_per_path=mag, spatial_dot_lag1=np.array([1, .9, .8]))
    profiles = {name: p for name in ['R1', 'R2', 'Model 11', 'Model 22', 'Model 33']}
    fig, events = figure(profiles, np.array([0, 80, 160]), 'Fixture')
    np.testing.assert_array_equal(fig.axes[3].lines[0].get_ydata(), [0, .5, 1])
    np.testing.assert_array_equal(fig.axes[1].lines[2].get_ydata(), [0, 0, 0])
    assert events['R1']['negative_endpoint'].sum() == 1
    assert events['R1']['crossed_zero'].sum() == 2
    assert len(fig.axes) == 4
    plt.close(fig)
