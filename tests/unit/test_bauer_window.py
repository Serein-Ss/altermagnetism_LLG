import pytest
import torch
import json
from pathlib import Path
from tests.experiments.bauer_window import normalized_window, sample_chunks
from scripts.model.gate_f import build_model, sample
from scripts.workflow.bauer_campaign import graph


def test_time_scale():
    x = torch.zeros(1, 1, 5, 1, 3)
    assert normalized_window(x)[0, -1] == 1
    assert normalized_window(x, 401, 32000)[0, -1] == 2
    with pytest.raises(ValueError):
        normalized_window(x, 1)


def test_chunk_anchor_and_length():
    torch.manual_seed(42)
    x = torch.zeros(1, 1, 5, 1, 3); x[..., 2] = 1
    net = build_model('rfm', width=8, blocks=1)
    t = normalized_window(x, 5)
    c = torch.tensor([[.11, .1]]); a = torch.tensor([[0., 0., 1.]])
    g = graph(5, dtype=torch.float32)
    gen = lambda: torch.Generator().manual_seed(7)
    one = sample(net, x, c, t, g, a, gen(), steps=2)
    two = sample_chunks(net, x, c, t, g, a, gen(), chunks=2, steps=2)
    assert two.shape == (1, 9, 1, 5, 1, 3)
    assert torch.equal(two[:, :5], one)
    assert torch.equal(two[:, 0], x)
    assert (two.norm(dim=-1)-1).abs().max() < 1e-5
    with pytest.raises(ValueError):
        sample_chunks(net, x, c, t, g, a, gen(), chunks=0)


def test_diagnostic_pipeline_smoke(tmp_path):
    from tests.experiments.bauer_checkpoint_diagnostics import run
    cfg = json.loads(Path('tests/conf/bauer_diagnostics_20260913_v2.json').read_text())
    cfg.update(initials=2, latent_seeds=1, integration_steps=[1, 2], benchmark_repeats=1,
               interface_lengths=[5], interface_steps=1, rollout_steps=1,
               raw_output=str(tmp_path/'data'), analysis_output=str(tmp_path/'assets'))
    path = tmp_path/'config.json'
    path.write_text(json.dumps(cfg))
    run(path, 11, 'cpu')
    result = json.loads((tmp_path/'assets/11/summary.json').read_text())
    assert result['status'] == 'COMPLETE_DIAGNOSTICS_NOT_CERTIFIED'
    assert len(result['cases']) == 9
    assert result['condition_response']['paired_path_rmse'] > 0
    assert all(row['norm_error'] < 1e-5 for row in result['cases'])
    assert all(row['selected_pool_indices'] == [0, 16] for row in result['initial_families'])
    assert all(len(row['per_initial']) == 2 for row in result['development_reference'])
    assert len(result['sampling_benchmark']['rows']) == 4
    assert all(row['median_batch_seconds'] > 0 for row in result['sampling_benchmark']['rows'])


def test_initial_family_selection():
    import numpy as np
    from tests.experiments.bauer_checkpoint_diagnostics import initial_families
    pool = np.zeros((6, 2, 3, 3))
    pool[3:, 0, :, 2] = 1
    selected, groups = initial_families(pool, 2, 2)
    assert selected.tolist() == [0, 3]
    assert groups.tolist() == [[0, 1], [3, 4]]
    with pytest.raises(ValueError, match='distinct'):
        initial_families(pool, 3, 2)
    with pytest.raises(ValueError, match='real trajectories'):
        initial_families(pool, 2, 4)
