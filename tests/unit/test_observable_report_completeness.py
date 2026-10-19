import json
import pytest
from tests.experiments.compare_overfit_observables import collect_reports, main


@pytest.mark.parametrize('variant',['dilation','global_context','weighted'])
def test_incomplete_and_duplicate_evaluations_rejected_before_output(tmp_path,monkeypatch,variant):
    monkeypatch.chdir(tmp_path)
    names=dict(dilation='temporal_dilation_overfit',global_context='global_context_overfit',weighted='weighted_flow_overfit')
    for experiment in ('cartesian_overfit',names[variant]):
        for seed in (11,22,33):
            path=tmp_path/f'tests/results/{experiment}/mixed_eight/{seed}/summary.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(status='COMPLETE_EXPLORATORY_NOT_CERTIFIED',
                config=dict(budgets=[600],evaluation_theta=[.13]),
                evaluation=[dict(step=600,theta=.13)])))
    kwargs={variant:True}
    assert len(collect_reports(**kwargs))==6
    d=json.loads(path.read_text());d['evaluation']=[];path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='evaluations'):main(**kwargs)
    assert not (tmp_path/'tests/results/temporal_dilation_observables').exists()
    assert not (tmp_path/'tests/results/global_context_observables').exists()
    assert not (tmp_path/'tests/results/weighted_flow_observables').exists()
    d['evaluation']=[dict(step=600,theta=.13)]*2;path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='evaluations'):collect_reports(**kwargs)
    d['status']='TRAINING';path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='not complete'):collect_reports(**kwargs)
