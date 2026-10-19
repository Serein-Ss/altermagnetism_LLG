import copy
import json
import pytest
import torch
from tests.experiments.director_budget import config_for_budget, train_step
from tests.experiments.director_controlled import train_step as original_step
from tests.experiments.director_transition import DirectorTransition


def test_only_planned_config_changes():
    original = json.load(open('tests/conf/local_transition_training.json'))
    small = config_for_budget(6000)
    large = config_for_budget(24000)
    assert {k for k in original if original[k] != small[k]} == {'flow_steps'}
    assert {k for k in small if small[k] != large[k]} == {'updates'}
    assert small['flow_steps'] == large['flow_steps'] == 32
    with pytest.raises(ValueError):
        config_for_budget(1)


def test_training_step_unchanged():
    torch.manual_seed(19)
    model = DirectorTransition(.12, .01, 16)
    other = copy.deepcopy(model)
    initial = torch.nn.functional.normalize(torch.randn(4, 25, 3), dim=-1)
    target = torch.nn.functional.normalize(initial+.02*torch.randn_like(initial), dim=-1)
    theta = torch.full((4,), .11)
    lag = torch.full((4,), .01)
    axis = torch.tensor([0., 0., 1.]).expand(4, -1)
    states = []
    for m, step in [(model, train_step), (other, original_step)]:
        rng = torch.Generator().manual_seed(95)
        step(m, torch.optim.Adam(m.parameters(), lr=.001), initial, target, theta, lag, axis, rng, .02)
        states.append(rng.get_state())
    assert torch.equal(*states)
    for key, value in model.state_dict().items():
        assert torch.equal(value, other.state_dict()[key])
