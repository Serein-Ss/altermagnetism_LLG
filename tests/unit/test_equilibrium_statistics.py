import numpy as np

from scripts.core.equilibrium_statistics import diagnostics, integrated_time


def test_independent_equilibrium_chains_pass():
    x=np.random.default_rng(17).normal(size=(4,2000))
    result=diagnostics(x)
    assert result["passed"]
    assert result["rhat"]<1.05


def test_nonmixing_chains_fail():
    x=np.random.default_rng(18).normal(size=(4,2000))
    x[0]+=4
    assert not diagnostics(x)["passed"]


def test_autocorrelation_reduces_effective_samples():
    rng=np.random.default_rng(19)
    x=np.zeros(10000)
    noise=rng.normal(size=len(x))
    for i in range(1,len(x)):
        x[i]=.9*x[i-1]+noise[i]
    assert integrated_time(x)>8
