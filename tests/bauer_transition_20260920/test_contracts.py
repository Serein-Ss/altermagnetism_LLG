import json
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.bauer_transition_20260920.run import config, rollout, temporal_errors, ROOT, old_config
from tests.bauer_closed_loop_20260919.model import Net
from tests.bauer_two_stage_20260919.models import base_path
from tests.bauer_publication_20260920.audit import phase_statistics
from tests.mdgen_path_refactor_20260918.data import PathWindows


@pytest.mark.parametrize('variant', ['path', 'transition'])
def test_composition_prefix_norm_and_anchor(variant):
    torch.set_num_threads(1); torch.manual_seed(2)
    net = Net('C', width=16, depth=1).eval()
    initial = F.normalize(torch.randn(2, 7, 3), dim=-1)
    c = config(); duration = c['duration_ps'][variant]
    a = rollout(net, initial, duration, 99, 2, variant, c)
    b = rollout(net, initial, duration*2, 99, 2, variant, c)
    assert torch.equal(a, b[:, :len(a[0])])
    assert torch.equal(b[:, 0], initial)
    torch.testing.assert_close(b.norm(dim=-1), torch.ones_like(b[..., 0]), atol=2e-6, rtol=0)
    assert not torch.equal(b[:, 1], b[:, -1])
    assert not torch.equal(a, rollout(net, initial, duration, 100, 2, variant, c))


def test_rotation_equivariance_and_train_gradient():
    torch.manual_seed(6); net = Net('C', width=16, depth=1)
    y = F.normalize(torch.randn(2, 2, 7, 3), dim=-1)
    rng = torch.Generator().manual_seed(71)
    z = base_path(y[:, 0], 2, rng, 'anchored'); tau = torch.tensor([.2, .7])
    x, target, _ = path_interpolate(z, y, tau, rng)
    times = torch.tensor([0., .1]); temperature = config()['temperature_K']
    v = net(x, tau, y[:, 0], times, temperature)
    rotation, _ = torch.linalg.qr(torch.randn(3,3))
    rotation[:, 0] *= torch.linalg.det(rotation)
    axis = torch.tensor([0., 0., 1.])@rotation
    rotated = net(x@rotation, tau, y[:, 0]@rotation, times, temperature, axis=axis)
    torch.testing.assert_close(rotated, v@rotation, atol=2e-5, rtol=2e-5)
    loss = (v[:, 1:]-target[:, 1:]).square().mean(); loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())


def test_short_phase_diagnostics_are_serializable():
    x = np.zeros((2, 11, 7, 3))
    values = phase_statistics(x)
    assert values['adjacent_increment_dot'][-1] is None
    json.dumps(values, allow_nan=False)
    assert temporal_errors(x, x) == dict(increment_power_relative=0., adjacent_dot_relative=0.)
    with pytest.raises(ValueError): temporal_errors(x[:, :2], x[:, :2])


def test_real_data_window_and_parent_split():
    c = old_config()
    root = ROOT/c['data']/'L16'
    manifest = json.loads((root/'manifest.json').read_text())
    groups = {split: {r['parent_id'] for r in manifest if r['split']==split} for split in ['train','dev','test']}
    assert not (groups['train'] & groups['dev'] or groups['train'] & groups['test'] or groups['dev'] & groups['test'])
    ds = PathWindows([root], 'train', .1, stride_frames=1)
    a, b = ds[0], ds[1]
    assert a['spins'].shape == (2,16,3)
    assert a['time_ps'][-1] == pytest.approx(.1)
    assert np.array_equal(a['spins'][1], b['spins'][0])
    assert a['parent_id'] in groups['train']


def test_invalid_duration():
    with pytest.raises(ValueError):
        rollout(None, torch.zeros(1,7,3), .15, 1, 2, 'transition', config())
