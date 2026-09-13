"""Synthetic open-chain software checks; no real training or P1/P2 certificate."""
import copy
import os
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from scripts.model.bauer_gate_f import open_chain_graph, build_model
from scripts.model.gate_f import reference_path, matching_loss, sample
from scripts.literature.bauer_2011.model import OpenChain
from scripts.analysis.bauer_features import primary_features
from scripts.validation.bauer_contract import validate_design, require_trainable

DEVICE = os.environ.get('GATE_F_TEST_DEVICE', 'cpu')


def fixture(dtype=torch.float64, length=7, frames=5, width=8, blocks=1):
    torch.manual_seed(20260913)
    g = torch.Generator(device=DEVICE).manual_seed(20260913)
    initial = torch.randn((1, 1, length, 1, 3), generator=g, device=DEVICE, dtype=dtype)
    initial /= initial.norm(dim=-1, keepdim=True)
    y = reference_path(initial, frames, g)
    graph = open_chain_graph(length).to(DEVICE)
    condition = y.new_tensor([[.11, .1]])
    time = torch.arange(frames, device=DEVICE, dtype=dtype)[None]*.1
    return g, y, graph, condition, time, y.new_tensor([[0., 0., 1.]])


def test_open_edges_energy_field_and_gradient():
    _, y, graph, _, _, axis = fixture()
    assert len(graph.edges) == 6
    assert not ((graph.edges[:, 0] == 0) & (graph.edges[:, 1] == 6)).any()
    flat = y.reshape(1, 5, 7, 3).requires_grad_()
    reference = OpenChain(dict(equation_convention='bauer_ll', exchange=1., anisotropy=.1))
    expected = reference.field(flat.reshape(5, 7, 3)).reshape_as(flat)
    assert torch.allclose(graph.field(flat, axis), expected, atol=1e-12)
    energy = graph.energy(flat, axis)
    assert torch.allclose(energy.flatten(), reference.energy(flat.reshape(5, 7, 3)), atol=1e-12)
    grad, = torch.autograd.grad(energy.sum(), flat)
    assert torch.allclose(-grad, expected, atol=1e-12)
    degree = graph.aggregate(torch.ones_like(flat[..., :1]))[0, 0, :, 0]
    assert degree.tolist() == [1., 2., 2., 2., 2., 2., 1.]


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_joint_so3_and_relabeling(dtype):
    g, y, graph, c, time, axis = fixture(dtype)
    net = build_model('rfm', width=8, blocks=1).to(device=DEVICE, dtype=dtype)
    tau = y.new_tensor([.4])
    v = net(y, tau, y[:, 0], c, time, graph, axis)
    tolerance = 1e-4 if dtype == torch.float32 else 1e-8
    for _ in range(20):
        q, _ = torch.linalg.qr(torch.randn((3, 3), generator=g, device=DEVICE, dtype=dtype))
        q[:, 0] *= torch.linalg.det(q)
        actual = net(y@q, tau, y[:, 0]@q, c, time, graph, axis@q)
        denominator = max(float(v.norm()), 1e-8*v.numel()**.5)
        assert float((actual-v@q).norm())/denominator < tolerance
    permutation = torch.randperm(7, generator=g, device=DEVICE)
    inverse = torch.argsort(permutation)
    shuffled = copy.deepcopy(graph)
    shuffled.edges = inverse[graph.edges]
    yy = y[:, :, :, permutation]
    actual = net(yy, tau, yy[:, 0], c, time, shuffled, axis)
    assert torch.allclose(actual, v[:, :, :, permutation], atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('kind', ['rfm', 'deterministic', 'euclidean', 'autoregressive'])
def test_all_backbones(kind):
    g, y, graph, c, time, axis = fixture()
    net = build_model(kind, width=8, blocks=1).to(device=DEVICE, dtype=y.dtype)
    loss, _ = matching_loss(net, y, c, time, graph, axis, g, kind=kind)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    prediction = sample(net, y[:, 0], c, time, graph, axis, g, kind=kind, steps=4)
    assert torch.isfinite(prediction).all()
    assert torch.equal(prediction[:, 0], y[:, 0])
    if kind != 'euclidean':
        assert float((prediction.norm(dim=-1)-1).abs().max()) < 1e-10


def test_lambda_and_theta_consumed():
    _, y, graph, c, time, axis = fixture()
    net = build_model('rfm', width=8, blocks=1).to(device=DEVICE, dtype=y.dtype)
    tau = y.new_tensor([.4])
    baseline = net(y, tau, y[:, 0], c, time, graph, axis)
    for column in (0, 1):
        changed = c.clone(); changed[:, column] *= 2
        assert not torch.allclose(baseline, net(y, tau, y[:, 0], changed, time, graph, axis))
    assert net.damping_parameter == 'lambda'


def test_44_features_and_contract_block():
    assert primary_features(np.zeros((4, 11)), np.zeros((4, 11, 3)), list(range(11))).shape == (4, 44)
    path = Path('conf/bauer_path/draft.yaml')
    c = yaml.safe_load(path.read_text())
    validate_design(c)
    with pytest.raises(ValueError): require_trainable(path)
    c['equation']['damping_parameter'] = 'alpha'
    with pytest.raises(ValueError): validate_design(c)


@pytest.mark.skipif(DEVICE != 'cuda', reason='requires Slurm GPU allocation')
def test_full_chain_candidate_architecture():
    torch.cuda.reset_peak_memory_stats()
    g, y, graph, c, time, axis = fixture(torch.float32, length=100, frames=201)
    net = build_model('rfm', width=32, blocks=3).to(DEVICE)
    loss, _ = matching_loss(net, y, c, time, graph, axis, g)
    loss.backward()
    assert torch.isfinite(loss)
    print('peak_cuda_bytes', torch.cuda.max_memory_allocated())
