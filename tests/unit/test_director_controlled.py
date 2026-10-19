import torch
from tests.experiments.local_equivariant_transition import LocalEquivariantTransition
from tests.experiments.director_transition import DirectorTransition
from tests.experiments.director_controlled import train_step


def test_paired_initialization_and_random_stream():
    models = []
    states = []
    for cls in [LocalEquivariantTransition, DirectorTransition]:
        torch.manual_seed(11)
        models.append(cls(.12, .01, 16))
        states.append(torch.get_rng_state())
    assert torch.equal(states[0], states[1])
    for key, value in models[0].state_dict().items():
        assert torch.equal(value, models[1].state_dict()[key])
    initial = torch.nn.functional.normalize(torch.randn(8, 25, 3), dim=-1)
    target = torch.nn.functional.normalize(initial+.02*torch.randn_like(initial), dim=-1)
    theta = torch.full((8,), .11); lag = theta*0+.01
    axis = torch.tensor([0., 0., 1.]).expand(8, -1)
    streams = []
    for model in models:
        rng = torch.Generator().manual_seed(950000011)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        train_step(model, optimizer, initial, target, theta, lag, axis, rng, .02)
        streams.append(rng.get_state())
    assert torch.equal(*streams)
