import torch
import pytest
from tests.experiments.stable_optimization import cosine_lr,update_ema


def test_schedule_endpoints_monotonic_and_ema():
    rates=[cosine_lr(i,100,.001,.00001) for i in range(1,101)]
    assert rates[0]==pytest.approx(.001) and rates[-1]==pytest.approx(.00001)
    assert all(a>=b for a,b in zip(rates,rates[1:]))
    average=dict(w=torch.tensor([0.,2.]),count=torch.tensor(0))
    current=dict(w=torch.tensor([2.,4.]),count=torch.tensor(5))
    update_ema(average,current,.5)
    assert torch.equal(average['w'],torch.tensor([1.,3.]))
    assert average['count']==5 and torch.equal(current['w'],torch.tensor([2.,4.]))
