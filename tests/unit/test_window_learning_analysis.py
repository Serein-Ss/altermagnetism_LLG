import numpy as np
import pytest
from tests.experiments.window_learning_analysis import boundary_increments,main,prefix_errors


def test_boundary_is_first_increment_after_anchor():
    paths=np.zeros((2,21,3,3));paths[:,11:,:,0]=1
    result=boundary_increments(paths)
    assert result['after_boundary']==1
    assert result['elsewhere']==0
    paths=np.zeros((2,21,3,3));paths[:,10:,:,0]=1
    result=boundary_increments(paths)
    assert result['after_boundary']==0
    assert result['elsewhere']>0


@pytest.mark.parametrize('mode',['default','conditional','stable'])
def test_missing_results_do_not_create_analysis(tmp_path,monkeypatch,mode):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):main(conditional=mode=='conditional',stable=mode=='stable')
    assert not (tmp_path/'tests/results').exists()


def test_prefix_includes_initial_and_endpoint_with_correct_time():
    reference=np.zeros((201,3));prediction=np.ones((201,3))
    result=prefix_errors(prediction,reference,80.)
    assert [r['reduced_time'] for r in result.values()]==[800.,4000.,16000.]
    assert all(r['mean_magnetization_rmse']==1 and r['endpoint_mz_bias']==1 for r in result.values())
    prediction[:]=0;prediction[10,2]=1
    result=prefix_errors(prediction,reference,80.)
    assert result['10']['mean_magnetization_rmse']==pytest.approx(np.sqrt(1/33))
