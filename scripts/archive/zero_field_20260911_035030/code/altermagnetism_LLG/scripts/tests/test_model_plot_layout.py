"""Ensure extracted visualization functions remain independently executable."""
import numpy as np

from altermagnetism_LLG.scripts.visualization.model_plots import _animate_pair, _plot_histograms


def test_extracted_model_plots(tmp_path):
    rows = [{"lattice_size": 2, "drive_T": 0.7,
             "reference": {"paths": 2, "endpoints": [-0.8, 0.9]},
             "flow": {"paths": 2, "endpoints": [-0.7, 0.8]}}]
    _plot_histograms(rows, tmp_path / "histogram.png")
    spins = np.zeros((2, 2, 2, 2, 3))
    spins[:, 0, :, :, 2] = 1
    spins[:, 1, :, :, 2] = -1
    _animate_pair(spins, spins, np.array([0., 1e-12]), tmp_path / "trajectory.gif")
    assert (tmp_path / "histogram.png").stat().st_size > 0
    assert (tmp_path / "trajectory.gif").stat().st_size > 0
