import unittest
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.empirical_transport_repair import test_oracle
from .run import mean_target


class TargetTests(unittest.TestCase):
    def test_existing_oracle(self):
        test_oracle.test_density_normalization_and_radial_quadrature()
        test_oracle.test_tau_zero_uniform_posterior_and_uniform_source_limit()
        test_oracle.test_single_target_velocity_rotation_and_tangency()

    def test_average_target_tau_zero(self):
        g=torch.Generator().manual_seed(93601)
        y=F.normalize(torch.randn(72,11,16,3,generator=g,dtype=torch.float64),dim=-1);y[:,0]=y[:1,0]
        z=F.normalize(torch.randn(1,11,16,3,generator=g,dtype=torch.float64)+2*y[:1,:1],dim=-1);z[:,0]=y[:1,0]
        xx,v,_=path_interpolate(z.expand_as(y),y,torch.zeros(72,dtype=y.dtype),g)
        truth,w=mean_target(z,torch.zeros(1,dtype=y.dtype),y,y[:1,0])
        torch.testing.assert_close(truth,v.mean(0,keepdim=True),atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(w,torch.full_like(w,1/72),atol=1e-10,rtol=1e-10)

    def test_float32_single_target_and_tangent(self):
        y,a,z=test_oracle.fixture();y=y[:1].float();a=a[:1].float();z=z[:1].float()
        for time in (.02,.25,.75,.9):
            t=z.new_full((1,),time);x,v,_=path_interpolate(z,y,t,torch.Generator().manual_seed(1))
            truth,_=mean_target(x,t,y,a)
            torch.testing.assert_close(truth,v,atol=2e-5,rtol=2e-5)
            self.assertLess(float((truth*x).sum(-1).abs().max()),1e-5)
            self.assertTrue(torch.equal(truth[:,0],torch.zeros_like(truth[:,0])))


if __name__=='__main__':unittest.main()
