import pytest
import torch
from tests.experiments.local_equivariant_transition import LocalEquivariantTransition, neighbors, loss, sample


def state(n, length):
    return torch.nn.functional.normalize(torch.randn(n, length, 3, dtype=torch.float64), dim=-1)


@pytest.mark.parametrize('length', [16, 25, 32])
def test_rotation_tangency_and_open_boundaries(length):
    torch.manual_seed(1)
    model = LocalEquivariantTransition(.12, .01, width=16).double()
    x, initial = state(2, length), state(2, length)
    axis = state(2, 1)[:, 0]
    tau = torch.tensor([.2, .7], dtype=torch.float64)
    theta, lag = tau*0+.11, tau*0+.01
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    q[:, 0] *= torch.linalg.det(q)
    v = model(x, tau, initial, theta, lag, axis)
    vr = model(x@q, tau, initial@q, theta, lag, axis@q)
    torch.testing.assert_close(vr, v@q, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close((v*x).sum(-1), torch.zeros_like(tau[:, None].expand(-1, length)), atol=1e-12, rtol=0)
    left, right = neighbors(x)
    assert torch.count_nonzero(left[:, 0]) == torch.count_nonzero(right[:, -1]) == 0
    changed = x.clone(); changed[:, -1] *= -1
    torch.testing.assert_close(model(changed, tau, initial, theta, lag, axis)[:, 0], v[:, 0])


def test_loss_backward_and_stochastic_sphere_sampling():
    torch.manual_seed(2)
    model = LocalEquivariantTransition(.12, .01, width=16).double()
    initial = state(4, 8); target = state(4, 8)
    theta = torch.full((4,), .11, dtype=torch.float64)
    lag = theta*0+.01; axis = state(4, 1)[:, 0]
    value, diagnostics = loss(model, initial, target, theta, lag, axis, torch.Generator().manual_seed(3), .02)
    assert torch.isfinite(value) and diagnostics['free_spin_count'] == 32
    value.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    a = sample(model, initial, theta, lag, axis, torch.Generator().manual_seed(4), .02, 4)
    b = sample(model, initial, theta, lag, axis, torch.Generator().manual_seed(4), .02, 4)
    c = sample(model, initial, theta, lag, axis, torch.Generator().manual_seed(5), .02, 4)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert not torch.equal(a, c)
    torch.testing.assert_close(a.norm(dim=-1), torch.ones_like(a[..., 0]), atol=1e-12, rtol=0)
