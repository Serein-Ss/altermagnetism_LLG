import unittest
import h5py
import numpy as np
import torch
from numba import set_num_threads
from tests.bauer_active_data_20260918.run import simulate
from tests.transport_audit.run import integrate
from .run import parents,reference_inputs,seeds_for,ROOT


class Tests(unittest.TestCase):
    def test_reference_replay(self):
        set_num_threads(2);row=parents()[0];initial,config=reference_inputs(row)
        with h5py.File(ROOT/row['path']) as h:
            expected=h['spins'][:2,:11];seeds=h['seeds'][:2]
        x,_=simulate(np.repeat(initial[None],2,axis=0),seeds,config,.25,1.)
        np.testing.assert_array_equal(x.astype('f4'),expected)

    def test_new_seeds(self):
        ids=np.concatenate([seeds_for(i,fine) for i in range(18) for fine in (False,True)])
        self.assertEqual(len(ids),4608);self.assertEqual(len(set(ids)),len(ids))
        self.assertTrue((ids<2**32).all())

    def test_midpoint_rotation_convergence(self):
        x=torch.zeros(2,11,4,3,dtype=torch.float64);x[...,0]=1
        axis=x.new_tensor([0.,0.,1.]).expand_as(x)
        def field(s,t):
            v=torch.linalg.cross(axis,s);v[:,0]=0;return v
        truth=x.clone();truth[:,1:,:,0]=np.cos(1.);truth[:,1:,:,1]=np.sin(1.)
        errors=[]
        for steps in (16,32,64):
            y=integrate(field,x,0.,1.,steps,'midpoint')
            self.assertLess(float((y.norm(dim=-1)-1).abs().max()),1e-12)
            torch.testing.assert_close(y[:,0],x[:,0],rtol=0,atol=0)
            errors.append(float((y-truth).abs().max()))
        self.assertGreater(errors[0]/errors[1],3.8);self.assertGreater(errors[1]/errors[2],3.8)

    def test_joint_rotation(self):
        torch.manual_seed(9);x=torch.nn.functional.normalize(torch.randn(2,11,4,3,dtype=torch.float64),dim=-1)
        q,_=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype));q=q*torch.linalg.det(q)
        a=x.new_tensor([.2,.3,.7]).expand_as(x)
        def field(axis):
            def f(s,t):
                v=torch.linalg.cross(axis,s);v[:,0]=0;return v
            return f
        for method in ('euler','midpoint'):
            y=integrate(field(a),x,0.,1.,16,method)
            yr=integrate(field(a@q.T),x@q.T,0.,1.,16,method)
            torch.testing.assert_close(yr,y@q.T,atol=1e-12,rtol=1e-12)


if __name__=='__main__':unittest.main()
