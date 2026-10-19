import torch
from tests.experiments.local_equivariant_transition import LocalEquivariantTransition
from tests.experiments.local_transition_training import train_step


def test_optimizer_updates_finitely():
    torch.manual_seed(10)
    model = LocalEquivariantTransition(.12, .01, 16)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    initial = torch.nn.functional.normalize(torch.randn(8, 25, 3), dim=-1)
    target = torch.nn.functional.normalize(initial+.02*torch.randn_like(initial), dim=-1)
    before = model.output.weight.detach().clone()
    theta = torch.full((8,), .11); lag = theta*0+.01
    axis = torch.tensor([0., 0., 1.]).expand(8, -1)
    value, grad, diagnostics = train_step(model, optimizer, initial, target, theta, lag, axis, torch.Generator().manual_seed(11), .02)
    assert value >= 0 and grad >= 0 and diagnostics['free_spin_count'] == 200
    assert not torch.equal(before, model.output.weight)
    assert all(torch.isfinite(p).all() for p in model.parameters())
