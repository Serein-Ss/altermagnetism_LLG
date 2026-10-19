import unittest
import torch
from torch.nn import functional as F
from scripts.core.cell_hamiltonian import CellHamiltonian
from tests.transition_block_repair.model import TransitionNet
from tests.transport_audit.run import integrate
from .model import LocalReadout,initialize
from .run import source


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ck=torch.load(source(11)/'checkpoint.pt',map_location='cpu',weights_only=True)

    def sample(self,L=7):
        torch.manual_seed(4)
        x=F.normalize(torch.randn(2,2,L,3),dim=-1)
        return x,torch.tensor([.3,.7]),x[:,0],torch.tensor([0.,.1])

    def active(self,mode):
        net=LocalReadout(mode)
        torch.nn.init.normal_(net.readout[-1].weight,std=.03)
        torch.nn.init.normal_(net.readout[-1].bias,std=.03)
        return net

    def test_zero_identity_and_parameter_pair(self):
        x,t,a,time=self.sample();old=TransitionNet();old.load_state_dict(self.ck['model'])
        base=old.velocity(x,t,a,time)
        count=[]
        for mode in ('pointwise','neighbor'):
            net,opt=initialize(self.ck,mode,11,'cpu')
            torch.testing.assert_close(net.velocity(x,t,a,time),base,rtol=0,atol=0)
            count.append(sum(p.numel() for p in net.readout.parameters()))
            self.assertEqual(len(opt.param_groups),2)
            self.assertEqual({int(v['step']) for v in opt.state.values()}, {30000})
        self.assertEqual(count,[6403,6403])

    def test_joint_rotation_and_tangent(self):
        x,t,a,time=self.sample();q,_=torch.linalg.qr(torch.randn(3,3));q[:,0]*=torch.linalg.det(q)
        for mode in ('pointwise','neighbor'):
            net=self.active(mode)
            v=net.velocity(x,t,a,time)
            w=net.velocity(x@q,t,a@q,time,frame=q)
            torch.testing.assert_close(w,v@q,atol=3e-6,rtol=3e-5)
            self.assertLess(float((v*x).sum(-1).abs().max()),2e-6)
            self.assertTrue(torch.equal(v[:,0],torch.zeros_like(v[:,0])))

    def test_sampling_covariance_and_anchor(self):
        x,t,a,time=self.sample();q,_=torch.linalg.qr(torch.randn(3,3));q[:,0]*=torch.linalg.det(q)
        net=self.active('neighbor')
        y=integrate(lambda z,tau:net.velocity(z,tau,a,time),x,0.,1.,8,'midpoint')
        yr=integrate(lambda z,tau:net.velocity(z,tau,a@q,time,frame=q),x@q,0.,1.,8,'midpoint')
        torch.testing.assert_close(yr,y@q,rtol=3e-5,atol=3e-6)
        self.assertTrue(torch.equal(y[:,0],a));self.assertLess(float((y.norm(dim=-1)-1).abs().max()),2e-6)

    def test_local_features_and_no_wrap(self):
        x,t,a,time=self.sample();cell=CellHamiltonian((7,),1,[(0,0,(1,),1.)],periodic=(False,));frame=torch.eye(3)
        for mode in ('pointwise','neighbor'):
            net=LocalReadout(mode)
            before=net.features(x.unsqueeze(-2),t,a.unsqueeze(-2),cell,frame,time)
            b=a.clone();b[:,-1]=-b[:,-1]
            after=net.features(x.unsqueeze(-2),t,b.unsqueeze(-2),cell,frame,time)
            torch.testing.assert_close(before[:,:,0],after[:,:,0],atol=0,rtol=0)
            if mode=='pointwise':torch.testing.assert_close(before[:,:,5],after[:,:,5],atol=0,rtol=0)
            else:self.assertGreater(float((before[:,:,5]-after[:,:,5]).abs().max()),.01)

    def test_cancellation_and_edge_average(self):
        x,t,a,time=self.sample(3);a=a.clone();a[:,0]=torch.tensor([0.,0.,1.]);a[:,2]=torch.tensor([0.,0.,-1.])
        cell=CellHamiltonian((3,),1,[(0,0,(1,),1.)],periodic=(False,))
        f=LocalReadout('neighbor').features(x.unsqueeze(-2),t,a.unsqueeze(-2),cell,torch.eye(3),time)
        torch.testing.assert_close(f[:,:,1,6:9],torch.zeros_like(f[:,:,1,6:9]),atol=0,rtol=0)
        torch.testing.assert_close(f[:,:,0,6:9],a[:,None,1].expand(-1,2,-1),atol=0,rtol=0)

    def test_checkpoint_and_boundary_contract(self):
        x,t,a,time=self.sample();p=LocalReadout('pointwise');n=LocalReadout('neighbor')
        with self.assertRaises(ValueError):n.load_state_dict(p.state_dict())
        cell=CellHamiltonian((7,),1,[(0,0,(1,),1.)],periodic=(True,))
        with self.assertRaises(ValueError):n(x.unsqueeze(-2),t,a.unsqueeze(-2),cell,torch.eye(3),0.,time)
        with self.assertRaises(ValueError):n.velocity(x,t,a,time,temperature=1.)
        with self.assertRaises(ValueError):n.velocity(x,t,a,time*2)

    def test_added_head_receives_gradient(self):
        x,t,a,time=self.sample();net,_=initialize(self.ck,'neighbor',11,'cpu')
        net.velocity(x,t,a,time).square().mean().backward()
        self.assertGreater(float(net.readout[-1].weight.grad.norm()),0.)


if __name__=='__main__':unittest.main()
