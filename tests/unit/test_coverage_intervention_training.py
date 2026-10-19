import json
import numpy as np
from tests.experiments.director_budget import train_step,config_for_budget
from tests.experiments.coverage_intervention_training import train_step as intervention_step
from tests.experiments.coverage_intervention_analysis import compare


def test_control_training_step_and_config_reused():
    assert intervention_step is train_step
    for seed in [11,22,33]:
        with open(f'tests/results/director_budget/24000/{seed}/summary.json') as f: report=json.load(f)
        assert config_for_budget(24000)==report['config']


def test_paired_statistic_labels_and_identical_models():
    rng=np.random.default_rng(7)
    a=rng.normal(size=(8,3,4,3)); a/=np.linalg.norm(a,axis=-1,keepdims=True)
    b=rng.normal(size=a.shape); b/=np.linalg.norm(b,axis=-1,keepdims=True)
    arrays,metrics=compare([a,a.copy(),b],rng,replicates=20)
    assert set(metrics)=={'control','coverage','paired_change'}
    for k in metrics['paired_change']:
        np.testing.assert_array_equal(arrays['paired_change_'+k],0)
    assert arrays['control_variance_ratio'].shape==(2,)
