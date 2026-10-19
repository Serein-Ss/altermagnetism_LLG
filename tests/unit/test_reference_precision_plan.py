import numpy as np
import pytest
from scipy.stats import binom
from tests.experiments.reference_precision_plan import exact_interval,planning_row


def test_exact_interval_coverage_and_symmetry():
    n=16
    for k in range(n+1):
        lo,hi=exact_interval(k,n);rlo,rhi=exact_interval(n-k,n)
        assert lo==pytest.approx(1-rhi)
        assert hi==pytest.approx(1-rlo)
    for p in np.linspace(0,1,101):
        cover=[exact_interval(k,n)[0]<=p<=exact_interval(k,n)[1] for k in range(n+1)]
        assert binom.pmf(np.arange(n+1),n,p)@cover>=.95-1e-12


def test_precision_improves_with_n_and_widens_with_multiplicity():
    a=planning_row(16);b=planning_row(128);c=planning_row(128,24)
    assert b['worst_single_probability_interval_half_width']<a['worst_single_probability_interval_half_width']
    assert c['worst_single_probability_interval_half_width']>b['worst_single_probability_interval_half_width']
    assert b['zero_events_one_sided_upper']==pytest.approx(1-.05**(1/128))
    assert b['real_paths_per_initial']==256
    for args in ((-1,3),(4,3),(1,0),(1.5,3)):
        with pytest.raises(ValueError):exact_interval(*args)
