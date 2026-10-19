import torch
from tests.experiments.flow_stage_audit import source_error_decomposition


def test_source_bias_variance_identity():
    gen=torch.Generator().manual_seed(93)
    targets=torch.randn(8,4,9,3,generator=gen,dtype=torch.float64)
    prediction=torch.randn(4,9,3,generator=gen,dtype=torch.float64)
    result=source_error_decomposition(prediction,targets)
    assert abs(result['total_mse']-result['target_variance']-result['model_bias_mse'])<1e-12
    ideal=source_error_decomposition(targets.mean(0),targets)
    assert ideal['model_bias_mse']==0
    assert ideal['total_mse']>0
