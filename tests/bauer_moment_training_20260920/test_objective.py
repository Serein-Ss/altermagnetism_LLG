import itertools
import torch
from torch.nn import functional as F
from tests.bauer_closed_loop_20260919.model import Net
from tests.bauer_two_stage_20260919.models import sample_path
from tests.bauer_moment_training_20260920.objective import differentiable_sample, replica_mean_loss


def test_replica_expectation_has_no_variance_penalty():
    mean = torch.tensor(.2, requires_grad=True)
    scale = torch.tensor(.7, requires_grad=True)
    losses = []
    target = torch.zeros(1, 1, 1, 1)
    for a, b in itertools.product([-1., 1.], repeat=2):
        x = (mean+scale*a).reshape(1, 1, 1, 1)
        y = (mean+scale*b).reshape(1, 1, 1, 1)
        losses.append(replica_mean_loss(x, y, target, target))
    loss = torch.stack(losses).mean()
    loss.backward()
    torch.testing.assert_close(loss, mean.detach()**2)
    torch.testing.assert_close(mean.grad, 2*mean.detach())
    torch.testing.assert_close(scale.grad, torch.zeros_like(scale))


def test_sampler_matches_frozen_sampler_and_backpropagates():
    torch.manual_seed(4); torch.set_num_threads(1)
    net = Net('C', 8, 1)
    initial = F.normalize(torch.randn(2, 5, 3), dim=-1)
    times = torch.tensor([0., .1])
    actual = differentiable_sample(net, initial, times, 10, 4, 63.82484966960072)
    expected = sample_path(net, initial, times, 10, 4, 'anchored',
                           dict(anchored_sigma=.7, anchored_time_rho=.8))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[:, 0], initial)
    torch.testing.assert_close(actual.norm(dim=-1), torch.ones_like(actual[..., 0]))
    actual[:, -1, :, 2].mean().backward()
    gradients = [p.grad for p in net.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0
