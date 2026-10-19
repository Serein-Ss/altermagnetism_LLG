import copy
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.bauer_p2_20260923.common import Model
from .run import optimizer_for


class Contract(unittest.TestCase):
    def test_velocity_derivative_and_endpoints(self):
        torch.manual_seed(19)
        z=F.normalize(torch.randn(4,11,16,3,dtype=torch.float64),dim=-1)
        y=F.normalize(torch.randn_like(z),dim=-1);z[:,0]=y[:,0]
        tau=torch.full((4,),.37,dtype=torch.float64)
        def interpolate(t):return path_interpolate(z,y,t,torch.Generator().manual_seed(41))
        x,v,_=interpolate(tau);plus,_,_=interpolate(tau+1e-6);minus,_,_=interpolate(tau-1e-6)
        torch.testing.assert_close((plus-minus)/2e-6,v,atol=1e-8,rtol=1e-8)
        torch.testing.assert_close((x*v).sum(-1),torch.zeros_like(x[...,0]),atol=1e-12,rtol=0)
        torch.testing.assert_close(interpolate(tau*0)[0],z,atol=1e-12,rtol=0)
        torch.testing.assert_close(interpolate(tau*0+1)[0],y,atol=1e-12,rtol=0)

    def test_optimizer_only_lr_changes(self):
        a=torch.nn.Linear(2,1);opt=torch.optim.AdamW(a.parameters(),lr=.001)
        a(torch.ones(2,2)).sum().backward();opt.step()
        ck={'optimizer':copy.deepcopy(opt.state_dict())}
        b=copy.deepcopy(a)
        x=optimizer_for(a,ck,.001);y=optimizer_for(b,ck,.0001)
        for p,q in zip(a.parameters(),b.parameters()):
            for name in x.state[p]:torch.testing.assert_close(x.state[p][name],y.state[q][name])
        self.assertEqual(x.param_groups[0]['lr'],.001)
        self.assertEqual(y.param_groups[0]['lr'],.0001)
        self.assertEqual(ck['optimizer']['param_groups'][0]['lr'],.001)

    def test_real_checkpoint_forward_and_gradient(self):
        from .run import P2
        torch.set_num_threads(1)
        ck=torch.load(P2/'runs/seed_11/best.pt',map_location='cpu',weights_only=False)
        net=Model(ck['config']);net.load_state_dict(ck['model'])
        with np.load(P2/'runs/seed_11/generated/dev_128/L16_parent145.npz') as z:
            y=F.normalize(torch.tensor(z['reference'][:2]),dim=-1)
        gen=torch.Generator().manual_seed(41)
        z=F.normalize(torch.randn(y.shape,generator=gen),dim=-1);z[:,0]=y[:,0]
        tau=torch.full((2,),.5);x,target,_=path_interpolate(z,y,tau,gen)
        loss=(net.velocity(x,tau,y[:,0],torch.arange(11)*.1)[:,1:]-target[:,1:]).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))
