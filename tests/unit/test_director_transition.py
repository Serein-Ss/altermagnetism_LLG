import numpy as np
import torch
from scripts.core.bauer_fast import drift
from tests.experiments.director_transition import DirectorTransition
from tests.experiments.local_equivariant_transition import LocalEquivariantTransition


def test_bauer_fixed_axis_half_turn_not_spin_inversion():
    rng = np.random.default_rng(1)
    s = rng.normal(size=(25, 3)); s /= np.linalg.norm(s, axis=-1, keepdims=True)
    rotation = np.diag([1., -1., -1.])
    f = drift.py_func(s, .1, .1)
    np.testing.assert_allclose(drift.py_func(s@rotation, .1, .1), f@rotation, atol=1e-14)
    assert not np.allclose(drift.py_func(-s, .1, .1), -f)


def test_director_symmetry_rotation_gradients_and_weight_compatibility():
    torch.manual_seed(1)
    base = LocalEquivariantTransition(.12, .01, 16).double()
    model = DirectorTransition(.12, .01, 16).double()
    model.load_state_dict(base.state_dict(), strict=True)
    x = torch.nn.functional.normalize(torch.randn(3, 25, 3, dtype=torch.float64), dim=-1)
    a = torch.nn.functional.normalize(torch.randn_like(x), dim=-1)
    t = x.new_tensor([.2, .5, .9]); theta = t*0+.11; lag = t*0+.01
    axis = x.new_tensor([0., 0., 1.]).expand(3, -1)
    v = model(x, t, a, theta, lag, axis)
    torch.testing.assert_close(v, model(x, t, a, theta, lag, -axis), atol=0, rtol=0)
    rotation = torch.diag(x.new_tensor([1., -1., -1.]))
    torch.testing.assert_close(model(x@rotation, t, a@rotation, theta, lag, axis), v@rotation, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close((v*x).sum(-1), torch.zeros_like(x[..., 0]), atol=1e-12, rtol=0)
    v.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # The unconstrained architecture does not impose this symmetry by construction.
    assert not torch.allclose(base(x, t, a, theta, lag, axis), base(x, t, a, theta, lag, -axis))
