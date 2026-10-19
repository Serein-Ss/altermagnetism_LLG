import json
from pathlib import Path
import torch
from tests.experiments.bauer_budget_comparison import run


def test_budget_prefix_and_pipeline(tmp_path):
    torch.set_num_threads(1)
    cfg = json.loads(Path('tests/conf/bauer_training_budget_20260913.json').read_text())
    cfg.update(initials=2, latents=1, sampling_steps=1, loss_draws=1)
    for name, budgets in [('short', [1]), ('long', [1, 2])]:
        cfg.update(budgets=budgets, **{k: str(tmp_path/name/k) for k in ('output', 'raw', 'analysis')})
        path = tmp_path/f'{name}.json'; path.write_text(json.dumps(cfg))
        run(path, 11, 'cpu')
        report = json.loads((tmp_path/name/'analysis/11/summary.json').read_text())
        assert report['status'] == 'COMPLETE_EXPLORATORY_NOT_CERTIFIED'
        assert len(report['evaluation']) == 2*len(budgets)
        assert all(r['initial_indices'] == [0, 16] for r in report['evaluation'])
    a, b = [torch.load(tmp_path/name/'output/11/step_1.pt', weights_only=False) for name in ('short', 'long')]
    assert all(torch.equal(v, b['model'][k]) for k, v in a['model'].items())
    assert torch.equal(a['generator_state'], b['generator_state'])
    assert a['optimizer']['state']
