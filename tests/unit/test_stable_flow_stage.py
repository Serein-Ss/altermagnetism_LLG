import pytest
import torch
from tests.experiments.stable_flow_stage import free_frame_risk


def test_vector_loss_reduction_and_anchor_exclusion():
    target=torch.zeros(2,4,1,5,1,3)
    pred=torch.ones_like(target); pred[:,0]=100
    error,norm=free_frame_risk(pred,target)
    torch.testing.assert_close(error,torch.full((2,3),3.))
    torch.testing.assert_close(norm,torch.zeros((2,3)))
    pred[:,2]=2
    error,_=free_frame_risk(pred,target)
    torch.testing.assert_close(error[:,1],torch.full((2,),12.))
    with pytest.raises(ValueError): free_frame_risk(pred[:1],target)
    pred[0,1,0,0,0,0]=float('nan')
    with pytest.raises(ValueError): free_frame_risk(pred,target)
