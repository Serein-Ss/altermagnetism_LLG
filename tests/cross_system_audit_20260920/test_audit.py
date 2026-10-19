import numpy as np
import torch
from scripts.model.gate_f import path_interpolate
from tests.cross_system_audit_20260920.report import local_metrics
from tests.cross_system_audit_20260920.symmetry import symmetric_hierarchy


def test_identical_distribution_and_persistence():
    rng=np.random.default_rng(42)
    x=rng.normal(size=(16,2,4,3));x/=np.linalg.norm(x,axis=-1,keepdims=True)
    x[:,0]=x[0,0]
    result=local_metrics(x,x)
    assert result['raw_mse']==0 and result['generated_variance']==result['reference_variance']
    assert abs(result['mean_tangent_cosine']-1)<1e-12
    assert abs(result['tangent_amplitude_ratio']-1)<1e-12


def test_interpolation_velocity_is_time_derivative():
    torch.manual_seed(5)
    z=torch.nn.functional.normalize(torch.randn(2,2,16,3,dtype=torch.float64),dim=-1)
    y=torch.nn.functional.normalize(torch.randn_like(z),dim=-1);z[:,0]=y[:,0]
    tau=torch.full((2,),.4,dtype=torch.float64);eps=1e-6
    def at(t):return path_interpolate(z,y,t,torch.Generator().manual_seed(1))
    state,velocity,_=at(tau)
    fd=(at(tau+eps)[0]-at(tau-eps)[0])/(2*eps)
    assert torch.max(abs(fd-velocity))<1e-8
    assert torch.max(abs((state*velocity).sum(-1)))<1e-12


def test_candidate_pooling_reflection_and_rotation():
    torch.manual_seed(10)
    q,_=torch.linalg.qr(torch.randn(3,3))
    for n in [16,25,32,50,100]:
        x=torch.randn(1,2,n,3)
        torch.testing.assert_close(symmetric_hierarchy(x),symmetric_hierarchy(x.flip(-2)).flip(-2))
        torch.testing.assert_close(symmetric_hierarchy(x@q),symmetric_hierarchy(x)@q)
