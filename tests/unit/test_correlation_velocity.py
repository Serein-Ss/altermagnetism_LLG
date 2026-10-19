import pytest
import torch
from tests.experiments.correlation_velocity import correlation_rate


def test_tangent_finite_difference_and_linearity():
    g = torch.Generator().manual_seed(71)
    s = torch.randn(3, 4, 5, 3, generator=g, dtype=torch.float64)
    s = s/s.norm(dim=-1, keepdim=True)
    v = torch.randn(s.shape, generator=g, dtype=s.dtype)
    v = v-(v*s).sum(-1, keepdim=True)*s
    def corr(x): return (x[:, :, :-1]*x[:, :, 1:]).sum(-1).mean(-1)
    def move(eps):
        x = s+eps*v
        return x/x.norm(dim=-1, keepdim=True)
    eps = 1e-6
    derivative = (corr(move(eps))-corr(move(-eps)))/(2*eps)
    torch.testing.assert_close(correlation_rate(s,v), derivative, rtol=1e-7, atol=1e-8)
    torch.testing.assert_close(correlation_rate(s,2*v), 2*correlation_rate(s,v))
    torch.testing.assert_close(correlation_rate(s,torch.zeros_like(v)), torch.zeros(3,4,dtype=s.dtype))


def test_no_periodic_bond_and_global_rotation_invariance():
    s = torch.tensor([[[[1.,0,0],[0,1,0],[0,0,1]]]])
    v = torch.zeros_like(s); v[0,0,0,2] = 1
    # Only the absent closing bond would contribute here.
    assert correlation_rate(s,v).item() == 0
    q = torch.tensor([[0.,0,1],[1,0,0],[0,1,0]])
    torch.testing.assert_close(correlation_rate(s@q,v@q),correlation_rate(s,v))


def test_bad_shapes_rejected():
    s = torch.zeros(2,3,1,3)
    with pytest.raises(ValueError): correlation_rate(s,s)
