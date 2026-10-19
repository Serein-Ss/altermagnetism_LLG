import io

import pytest
import torch
from torch.nn import functional as F

from scripts.core.cell_hamiltonian import CellHamiltonian
from scripts.model.sphere import geodesic_interpolate
from tests.path_smoke_20260917.model import generate
from .model import SpinGraphNet, cell_edges


def setup(shape, periodic, basis=1):
    torch.set_num_threads(1)
    torch.manual_seed(29)
    templates = [(0, 0, tuple(int(i == axis) for i in range(len(shape))), 1)
                 for axis in range(len(shape))]
    if basis == 2:
        templates.append((0, 1, (0,)*len(shape), 1))
    cell = CellHamiltonian(shape, basis, templates, periodic=periodic)
    net = SpinGraphNet(periodic, 16, 2)
    x = F.normalize(torch.randn(2, 3, *shape, basis, 3), dim=-1)
    return net, cell, x, torch.tensor([.2, .7]), torch.eye(3), torch.arange(3)*.1


@pytest.mark.parametrize('shape,periodic,basis', [((5,), (False,), 1), ((6, 4), (True, True), 2), ((3, 4, 3), (False, False, False), 1)])
def test_rotation_tangent_gradient(shape, periodic, basis):
    net, cell, x, tau, frame, time = setup(shape, periodic, basis)
    v = net(x, tau, x[:, 0], cell, frame, .1, time)
    for _ in range(3):
        q, _ = torch.linalg.qr(torch.randn(3, 3))
        q[:, 0] *= torch.linalg.det(q)
        rotated = net(x@q, tau, x[:, 0]@q, cell, frame@q, .1, time)
        torch.testing.assert_close(rotated, v@q, atol=3e-6, rtol=3e-5)
    assert (x*v).sum(-1).abs().max() < 2e-6
    assert torch.count_nonzero(v[:, 0]) == 0
    v.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_edges_and_boundary_contract():
    net, cell, x, tau, frame, time = setup((5,), (False,))
    edges = cell_edges(cell)
    assert edges.shape == (2, 8)
    assert set(map(tuple, edges.T.tolist())) == {(i, j) for i in range(5) for j in range(5) if abs(i-j) == 1}
    periodic = CellHamiltonian((5,), 1, cell.templates, periodic=(True,))
    assert cell_edges(periodic).shape == (2, 10)
    with pytest.raises(ValueError, match='boundary'):
        net(x, tau, x[:, 0], periodic, frame, .1, time)
    tiny = CellHamiltonian((2,), 1, cell.templates, periodic=(True,))
    assert cell_edges(tiny).shape == (2, 4)  # two distinct periodic images
    with pytest.raises(ValueError, match='contract'):
        SpinGraphNet((True,), 16, 2).load_state_dict(net.state_dict())


def test_translation_size_and_no_hamiltonian_input():
    net, cell, x, tau, frame, time = setup((5, 4), (True, True))
    v = net(x, tau, x[:, 0], cell, frame, .1, time)
    shifted = net(x.roll(1, 2), tau, x[:, 0].roll(1, 1), cell, frame, .1, time)
    torch.testing.assert_close(shifted, v.roll(1, 2), atol=2e-6, rtol=2e-5)
    changed = CellHamiltonian(cell.shape, 1, [(a, b, d, 999) for a, b, d, _ in cell.templates], periodic=cell.periodic, anisotropy=(7, 8, 9))
    torch.testing.assert_close(net(x, tau, x[:, 0], changed, frame, .1, time), v, atol=0, rtol=0)
    _, bigger, large, _, _, _ = setup((9, 6), (True, True))
    assert net(large, tau, large[:, 0], bigger, frame, .1, time).shape == large.shape


def test_train_sample_and_checkpoint():
    net, cell, truth, tau, frame, time = setup((5,), (False,))
    source = F.normalize(torch.randn_like(truth), dim=-1)
    source[:, 0] = truth[:, 0]
    state, target = geodesic_interpolate(source, truth, tau)
    optimizer = torch.optim.Adam(net.parameters(), lr=.003)
    losses = []
    for _ in range(25):
        optimizer.zero_grad()
        prediction = net(state, tau, truth[:, 0], cell, frame, .1, time)
        loss = (prediction[:, 1:]-target[:, 1:]).square().mean()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0]
    a = generate(net, truth[:, 0], cell, frame, .1, time, seed=7, steps=8)
    b = generate(net, truth[:, 0], cell, frame, .1, time, seed=7, steps=8)
    c = generate(net, truth[:, 0], cell, frame, .1, time, seed=8, steps=8)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert not torch.allclose(a[:, 1:], c[:, 1:])
    torch.testing.assert_close(a[:, 0], truth[:, 0], atol=0, rtol=0)
    assert (a.norm(dim=-1)-1).abs().max() < 3e-7
    buffer = io.BytesIO()
    torch.save(net.state_dict(), buffer)
    buffer.seek(0)
    restored = SpinGraphNet((False,), 16, 2)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(generate(restored, truth[:, 0], cell, frame, .1, time, seed=7, steps=8), a, atol=0, rtol=0)


def test_full_sampler_joint_rotation_and_open_reversal():
    net, cell, x, tau, frame, time = setup((7,), (False,))
    v = net(x, tau, x[:, 0], cell, frame, .1, time)
    reverse = net(x.flip(2), tau, x[:, 0].flip(1), cell, frame, .1, time)
    torch.testing.assert_close(reverse, v.flip(2), atol=2e-6, rtol=2e-5)
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    q[:, 0] *= torch.linalg.det(q)
    a = generate(net, x[:, 0], cell, frame, .1, time, source=x, steps=8)
    b = generate(net, x[:, 0]@q, cell, frame@q, .1, time, source=x@q, steps=8)
    torch.testing.assert_close(b, a@q, atol=4e-6, rtol=4e-5)
    # Same numeric RNG seed alone is not a rotated realization of vector noise.
    with pytest.raises(ValueError, match='times'):
        net(x, tau, x[:, 0], cell, frame, .1, time.flip(0))
