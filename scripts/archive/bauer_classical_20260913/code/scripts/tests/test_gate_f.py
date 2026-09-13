"""Synthetic software checks only; these do not certify a thermal model."""
import os
import pytest
import torch
from scripts.literature.gomonay_2024.model import DoubleLayer
from scripts.model.gate_f import CellGraph, GateFNet, reference_path, path_interpolate, matching_loss, build_model, sample
from scripts.validation.gate_f_contract import load_contract, audit_records
from pathlib import Path

DEVICE = os.environ.get('GATE_F_TEST_DEVICE', 'cpu')
REDUCED = dict(J1=1., J2=1.88/11.1, J_tilde=.8/11.1, K_SW=0., K_DW=.047/11.1)


def fixture(orientation='100', dtype=torch.float64, shape=(3, 4)):
    torch.manual_seed(20260912)
    cell = DoubleLayer(REDUCED, shape, wall=True, orientation=orientation)
    graph = CellGraph(cell).to(DEVICE)
    generator = torch.Generator(device=DEVICE).manual_seed(314)
    initial = torch.randn((1, cell.basis, *shape, 3), dtype=dtype, device=DEVICE, generator=generator)
    initial = initial/initial.norm(dim=-1, keepdim=True)
    y = reference_path(initial, 4, generator)
    condition = y.new_tensor([[.3, .05]])
    time = y.new_tensor([[0., .1, .2, .3]])
    axis = y.new_tensor([[0., 0., 1.]])
    return cell, graph, generator, y, condition, time, axis


@pytest.mark.parametrize('orientation', ['100', '110'])
@pytest.mark.parametrize('periodic', [(True, True), (False, True), (False, False)])
def test_graph_matches_existing_hamiltonian(orientation, periodic):
    cell, graph, gen, y, c, t, axis = fixture(orientation)
    cell = DoubleLayer(REDUCED, (3, 4), wall=True, orientation=orientation, periodic=periodic)
    graph = CellGraph(cell).to(DEVICE)
    flat = y.reshape(1, 4, -1, 3).requires_grad_()
    native = flat.reshape_as(y).permute(0, 1, 3, 4, 2, 5).reshape(4, 3, 4, cell.basis, 3)
    expected = cell.field(native).reshape(1, 4, 3, 4, cell.basis, 3).permute(0, 1, 4, 2, 3, 5).reshape_as(flat)
    assert torch.allclose(graph.field(flat, axis), expected, atol=1e-12)
    energy = graph.energy(flat, axis)
    assert torch.allclose(energy.flatten(), cell.energy(native), atol=1e-12)
    grad, = torch.autograd.grad(energy.sum(), flat)
    assert torch.allclose(-grad, graph.field(flat, axis), atol=1e-12)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('orientation', ['100', '110'])
def test_twenty_joint_rotations_and_translations(dtype, orientation):
    _, graph, gen, y, c, time, axis = fixture(orientation, dtype)
    net = GateFNet(width=8, blocks=1).to(device=DEVICE, dtype=dtype)
    tau = y.new_tensor([.3])
    output = net(y, tau, y[:, 0], c, time, graph, axis)
    tolerance = 1e-4 if dtype == torch.float32 else 1e-8
    for _ in range(20):
        q, _ = torch.linalg.qr(torch.randn((3, 3), dtype=dtype, device=DEVICE, generator=gen))
        q[:, 0] *= torch.det(q)
        rotated = net(y@q, tau, y[:, 0]@q, c, time, graph, axis@q)
        error = (rotated-output@q).norm()/max(output.norm(), 1e-8*output.numel()**.5)
        assert error < tolerance
        dx, dy = torch.randint(0, 10, (2,), generator=gen, device=DEVICE).tolist()
        shifted = y.roll((dx, dy), (3, 4))
        prediction = net(shifted, tau, shifted[:, 0], c, time, graph, axis)
        assert (prediction-output.roll((dx, dy), (3, 4))).abs().max() < tolerance
    assert (output*y).sum(-1).abs().max() < tolerance
    assert torch.equal(output[:, 0], torch.zeros_like(output[:, 0]))


def test_source_and_cut_locus():
    _, graph, gen, y, c, t, axis = fixture()
    z = reference_path(y[:, 0], 4, gen)
    target = -z.clone()
    target[:, 0] = z[:, 0]
    for tau in (0., .5, 1.):
        x, v, info = path_interpolate(z, target, z.new_tensor([tau]), gen)
        assert info['unresolved_cut_count'] == info['free_spin_count']
        assert torch.allclose(x.norm(dim=-1), torch.ones_like(x[..., 0]), atol=1e-12)
        assert (x*v).sum(-1).abs().max() < 1e-12
        if tau == 1.:
            assert torch.allclose(x, target, atol=1e-12)
    x, v, info = path_interpolate(y, y, y.new_tensor([.2]), gen)
    assert torch.allclose(x, y, atol=1e-12)
    assert v.abs().max() < 1e-12
    assert info['unresolved_cut_count'] == 0


@pytest.mark.parametrize('kind', ['rfm', 'deterministic', 'euclidean', 'autoregressive'])
def test_backprop_sampling_and_checkpoint(kind):
    _, graph, gen, y, c, time, axis = fixture()
    net = build_model(kind, width=8, blocks=1).to(device=DEVICE, dtype=y.dtype)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        loss, info = matching_loss(net, y, c, time, graph, axis, gen, kind=kind)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
        optimizer.step()
    prediction = sample(net, y[:, 0], c, time, graph, axis, gen, steps=3, kind=kind)
    assert prediction.shape == y.shape and torch.isfinite(prediction).all()
    assert torch.equal(prediction[:, 0], y[:, 0])
    if kind != 'euclidean':
        assert (prediction.norm(dim=-1)-1).abs().max() < 1e-10
    assert 'condition_mean' in net.state_dict() and 'condition_std' in net.state_dict()


def test_conditions_and_physical_duration_are_consumed():
    _, graph, gen, y, c, t, axis = fixture()
    net = GateFNet(width=8, blocks=2).to(device=DEVICE, dtype=y.dtype)
    tau = y.new_tensor([.3])
    original = net(y, tau, y[:, 0], c, t, graph, axis)
    for changed in (c+y.new_tensor([[.1, 0.]]), c+y.new_tensor([[0., .1]])):
        assert not torch.allclose(original, net(y, tau, y[:, 0], changed, t, graph, axis))
    assert not torch.allclose(original, net(y, tau, y[:, 0], c, 2*t, graph, axis))


def test_reject_draft_and_leakage():
    draft = Path(__file__).resolve().parents[2]/'conf/gate_f/draft.yaml'
    with pytest.raises(ValueError, match='frozen'):
        load_contract(draft)
    rows = [dict(initial_id='a', source_family_id='family', parent_trajectory_id='p',
                 initial_type='random_sphere', split='train', noise_id=1),
            dict(initial_id='b', source_family_id='family', parent_trajectory_id='q',
                 initial_type='random_sphere', split='development', noise_id=2)]
    with pytest.raises(ValueError, match='leakage'):
        audit_records(rows)


def test_graph_relabeling_covariance():
    _, graph, gen, y, c, time, axis = fixture('110')
    net = GateFNet(width=8, blocks=1).to(device=DEVICE, dtype=y.dtype)
    tau = y.new_tensor([.4])
    original = net(y, tau, y[:, 0], c, time, graph, axis)
    n = y[0, 0, ..., 0].numel()
    permutation = torch.randperm(n, generator=gen, device=DEVICE)
    inverse = torch.argsort(permutation)
    graph.edges = inverse[graph.edges]
    relabeled = y.reshape(1, 4, n, 3)[:, :, permutation].reshape_as(y)
    prediction = net(relabeled, tau, relabeled[:, 0], c, time, graph, axis)
    expected = original.reshape(1, 4, n, 3)[:, :, permutation].reshape_as(y)
    assert torch.allclose(prediction, expected, atol=1e-10)


def test_ed_and_scales():
    import numpy as np
    from scripts.analysis.gate_f_metrics import energy_distance, fit_training_scale, simultaneous_intervals
    x = np.zeros((8, 2))
    y = np.ones_like(x)
    assert energy_distance(x, x) == 0
    assert np.isclose(energy_distance(x, y), 2)
    with pytest.raises(ValueError, match='training'):
        fit_training_scale(x, [.1, .1], split='development')
    mean, scale = fit_training_scale(x, [.1, .2], split='train')
    assert np.array_equal(scale, [.1, .2])
    intervals = simultaneous_intervals(np.zeros(2), np.ones((2000, 2)))
    assert np.array_equal(intervals, [[-1, 1], [-1, 1]])


@pytest.mark.skipif(DEVICE != 'cuda', reason='full Gate-F shape smoke requires Slurm GPU')
@pytest.mark.parametrize('size', [16, 32])
def test_full_gate_shape_gpu(size):
    _, graph, gen, y, c, time, axis = fixture(dtype=torch.float32, shape=(size, size))
    y = reference_path(y[:, 0], 101, gen)
    time = torch.linspace(0, 1, 101, device=DEVICE)[None]
    net = GateFNet(width=8, blocks=1).to(DEVICE)
    loss, info = matching_loss(net, y, c, time, graph, axis, gen)
    loss.backward()
    assert torch.isfinite(loss)
    source = reference_path(y[:, 0], 101, gen)
    for steps in (2, 4):
        result = sample(net, y[:, 0], c, time, graph, axis, gen, steps=steps, source=source)
        assert torch.equal(result[:, 0], y[:, 0])
        assert (result.norm(dim=-1)-1).abs().max() < 1e-5


def test_lazy_schema_roundtrip(tmp_path):
    import json
    import hashlib
    import h5py
    import numpy as np
    from scripts.datasets.gate_f_paths import GateFPaths
    from scripts.validation.gate_f_contract import digest
    # Deliberately tiny SOFTWARE fixture, not a frozen 1152-path contract.
    rows, initials = [], []
    kinds = ['ground_perturbed', 'random_sphere', 'equilibrium_pool']
    for theta in [.1, .3, .6]:
        for kind in kinds:
            for index in range(4):
                identity = f'{theta}_{kind}_{index}'
                split = 'train' if index < 2 else 'development'
                path = tmp_path/(identity+'.h5')
                initial = np.zeros((2, 2, 2, 3), dtype=np.float64)
                initial[..., 2] = 1
                row = dict(path=str(path), initial_id=identity, noise_id=0,
                           source_family_id=identity, source_chain_id=identity,
                           parent_trajectory_id=identity, split=split,
                           initial_type=kind, theta=theta, alpha=.05)
                with h5py.File(path, 'w') as h:
                    for key, value in row.items():
                        h.attrs[key] = value
                    h.attrs.update(schema='gate_f_path_v1', complete=True)
                    h['spins'] = np.stack([initial]*3)
                    h['initial'] = initial
                    h['time'] = [0., .1, .2]
                    h['energy'] = np.zeros(3)
                    h['neel'] = np.zeros((3, 3))
                row['sha256'] = digest(path)
                rows.append(row)
                initials.append(dict(initial_id=identity, shape=list(initial.shape), dtype=str(initial.dtype),
                                     spins_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                                     split=split, source_family_id=identity))
    manifest, catalog = tmp_path/'manifest.json', tmp_path/'initials.json'
    manifest.write_text(json.dumps(dict(paths=rows)))
    catalog.write_text(json.dumps(dict(initials=initials)))
    c = dict(data_manifest={'path': str(manifest)}, initial_manifest={'path': str(catalog)},
             frames=3, model={'sizes': [2], 'alpha': .05}, save_dt=.1,
             theta=[.1, .3, .6], initial_types=kinds, noise_per_initial=1)
    dataset = GateFPaths(c, 'train')
    assert len(dataset) == 18
    assert dataset[0]['spins'].shape == (3, 2, 2, 2, 3)
    with h5py.File(rows[0]['path'], 'a') as h:
        h.attrs['complete'] = False
    rows[0]['sha256'] = digest(rows[0]['path'])
    manifest.write_text(json.dumps(dict(paths=rows)))
    with pytest.raises(ValueError, match='complete'):
        GateFPaths(c, 'train')


def test_bootstrap_is_reproducible_and_finite():
    import numpy as np
    from scripts.analysis.gate_f_metrics import bootstrap_family
    a = np.arange(8, dtype=float).reshape(2, 2, 2)/10
    group = dict(group_id='synthetic', real_a=a, real_b=a+.1,
                 generated={'rfm:1': a+.2}, initial_ids=['a', 'b'], source_families=['a', 'b'])
    first = bootstrap_family([group], ['rfm:1'], repetitions=2000, seed=1)
    second = bootstrap_family([group], ['rfm:1'], repetitions=2000, seed=1)
    assert first == second
    assert np.isfinite(first['simultaneous_ci']).all()
    assert first['status'] != 'GO'
