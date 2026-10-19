import unittest
import numpy as np
import torch
from torch.nn import functional as F
from scripts.core.cell_hamiltonian import CellHamiltonian
from tests.spin_path_transformer.run import make_net
from .model import make_transition,pair_window,rollout
from .metrics import statistics,compare_first


class ContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);self.task=dict(domain='bauer',arm='hierarchical',seed=11)
        self.net=make_transition(self.task,'cpu').double().eval()
        self.a=F.normalize(torch.randn(3,16,3,dtype=torch.float64),dim=-1)

    def test_initialization_metadata_and_grid(self):
        base=make_net(self.task,'cpu').double()
        for a,b in zip(base.parameters(),self.net.parameters()):self.assertTrue(torch.equal(a,b))
        self.assertEqual(self.net.get_extra_state()['block_ps'],.1)
        with self.assertRaises(ValueError):self.net.load_state_dict(base.state_dict())
        x=self.a[:,None].repeat(1,2,1,1)
        for t in (torch.tensor([0.,1.],dtype=x.dtype),torch.arange(11,dtype=x.dtype)*.1):
            with self.assertRaises(ValueError):self.net.velocity(x,x.new_zeros(3),self.a,t)
        with self.assertRaises(ValueError):self.net.velocity(x,x.new_zeros(3),self.a,x.new_tensor([0.,.1]),temperature=1.)
        cell=CellHamiltonian((16,),1,[(0,0,(1,),1.)],periodic=(True,))
        with self.assertRaises(ValueError):self.net(x.unsqueeze(-2),x.new_zeros(3),self.a.unsqueeze(-2),cell,torch.eye(3,dtype=x.dtype),0.,x.new_tensor([0.,.1]))

    def test_nonzero_covariance_tangent_anchor(self):
        with torch.no_grad():self.net.output.weight.normal_(0,.02)
        x=F.normalize(torch.randn(3,2,16,3,dtype=torch.float64),dim=-1);x[:,0]=self.a
        tau=x.new_full((3,),.3);times=x.new_tensor([0.,.1])
        q,_=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype));q[:,0]*=torch.linalg.det(q)
        v=self.net.velocity(x,tau,self.a,times);vr=self.net.velocity(x@q,tau,self.a@q,times,frame=q)
        self.assertLess(float(abs(vr-v@q).max()),1e-9)
        self.assertLess(float(abs((v*x).sum(-1)).max()),1e-9)
        self.assertEqual(float(v[:,0].abs().max()),0.)

    def test_rollout_resume_and_randomness(self):
        gen=torch.Generator().manual_seed(9);full=rollout(self.net,self.a,gen,2,10);end=gen.get_state()
        gen=torch.Generator().manual_seed(9);part=rollout(self.net,self.a,gen,2,4);saved=gen.get_state()
        resumed=rollout(self.net,part[:,-1],torch.Generator().set_state(saved),2,6)
        self.assertTrue(torch.equal(full,torch.cat((part,resumed[:,1:]),1)))
        self.assertEqual(full.shape,(3,11,16,3));self.assertTrue(torch.equal(full[:,0],self.a))
        self.assertLess(float(abs(full.norm(dim=-1)-1).max()),1e-12)
        self.assertFalse(torch.equal(full[:,1],full[:,2]));self.assertFalse(torch.equal(saved,end))

    def test_pair_selection_and_metrics(self):
        x=torch.arange(3*11*16*3).reshape(3,11,16,3);starts=torch.tensor([0,4,9]);y=pair_window(x,starts)
        for i,j in enumerate(starts):self.assertTrue(torch.equal(y[i],x[i,j:j+2]))
        with self.assertRaises(ValueError):pair_window(x,torch.tensor([0,1,10]))
        g=F.normalize(torch.randn(20,2,16,3,dtype=torch.float64),dim=-1).numpy()
        s=statistics(g);d=g[:,1]-g[:,0]
        expected=np.stack([np.cov(d[:,i].T,ddof=1) for i in range(16)])
        np.testing.assert_allclose(s['increment_cov'],expected)
        values=compare_first(g,g,10)
        for v in values.values():self.assertEqual(v['rmse'],0.);self.assertTrue(np.isfinite(v['pointwise_ci95']).all())


if __name__=='__main__':unittest.main()
