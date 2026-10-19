import torch
from torch.nn import functional as F
from scripts.core.cell_hamiltonian import CellHamiltonian
from .models import ReversalNet, SpaceTimeNet, observed_context, masked_flow_loss


def test_chain_reversal_and_gradient():
    torch.set_num_threads(1); torch.manual_seed(7)
    net = ReversalNet('C', 16, 1)
    for n in (16, 25, 50, 100):
        x = F.normalize(torch.randn(2, 3, n, 3), dim=-1)
        args = (torch.tensor([.3, .7]), x[:, 0], torch.arange(3)*.1, 63.82485)
        v = net(x, *args)
        reverse = net(x.flip(-2), args[0], args[1].flip(-2), *args[2:]).flip(-2)
        torch.testing.assert_close(v, reverse, atol=1e-6, rtol=1e-5)
        assert (v*x).sum(-1).abs().max() < 1e-5
    v.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_space_time_covariance_translation_and_gradient():
    torch.set_num_threads(1); torch.manual_seed(8)
    cell = CellHamiltonian((6, 5), 1, [(0, 0, (1, 0), 1), (0, 0, (0, 1), 1)], periodic=(True, True))
    net = SpaceTimeNet(16, 2)
    x = F.normalize(torch.randn(1, 3, 6, 5, 1, 3), dim=-1)
    tau = torch.tensor([.3]); times = torch.arange(3)*.1; frame = torch.eye(3)
    v = net(x, tau, x[:, 0], cell, frame, .1, times)
    moved = net(x.roll(1, 2), tau, x[:, 0].roll(1, 1), cell, frame, .1, times)
    torch.testing.assert_close(moved, v.roll(1, 2), atol=2e-6, rtol=2e-5)
    q, _ = torch.linalg.qr(torch.randn(3, 3)); q[:, 0] *= torch.linalg.det(q)
    rotated = net(x@q, tau, x[:, 0]@q, cell, frame@q, .1, times)
    torch.testing.assert_close(rotated, v@q, atol=2e-6, rtol=2e-5)
    assert (v*x).sum(-1).abs().max() < 1e-5
    v.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_mask_does_not_leak_unknown_frames():
    truth = torch.randn(2, 5, 4, 3)
    known = torch.zeros(2, 5, 4, dtype=torch.bool); known[:, [0, -1]] = True
    context = observed_context(truth, known)
    changed = truth.clone(); changed[~known] += 100
    torch.testing.assert_close(context, observed_context(changed, known))
    assert masked_flow_loss(torch.zeros_like(truth), torch.ones_like(truth), known) == 3
