import json
from pathlib import Path
import numpy as np
import torch
from tests.experiments.bauer_budget_comparison import run


def test_historical_diagnostic_path():
    from scripts.core.project_paths import resolve_recorded_path
    path = resolve_recorded_path('assets/research/bauer_training_budget/20260913_v1/11/summary.json')
    assert path.exists()
    assert 'tests/results/bauer_training_budget' in str(path)


def test_single_path_overfit_pipeline(tmp_path):
    torch.set_num_threads(1)
    cfg = json.loads(Path('tests/conf/bauer_training_budget_20260913.json').read_text())
    cfg.update(train_indices=[97], evaluate_training_subset=True, compare_legacy=False,
        evaluation_theta=[.13], initials=1, reference_latents=1, latents=2,
        budgets=[1, 2], sampling_steps=1, loss_draws=1,
        **{k: str(tmp_path/k) for k in ('output', 'raw', 'analysis')})
    path = tmp_path/'config.json'; path.write_text(json.dumps(cfg))
    run(path, 11, 'cpu')
    report = json.loads((tmp_path/'analysis/11/summary.json').read_text())
    assert len(report['evaluation']) == 2
    for row in report['evaluation']:
        assert row['nearest_training_path_counts'] == [2]
        assert np.isfinite(row['nearest_training_path_rmse']).all()
        assert set(row['loss']) == {'train'}
