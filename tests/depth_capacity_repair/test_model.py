import copy,unittest
import torch
from torch.nn import functional as F
from .model import initialize,DeepNet,activation
from tests.transition_block_repair.model import TransitionNet

class Tests(unittest.TestCase):
    def checkpoint(self):
        net=TransitionNet()
        # Activate the real readout so identity tests cannot pass on a zero field.
        torch.nn.init.normal_(net.output.weight,std=.05)
        opt=torch.optim.AdamW(net.parameters(),lr=3e-5)
        for p in net.parameters():p.grad=torch.full_like(p,.01)
        opt.step()
        return {'model':copy.deepcopy(net.state_dict()),'optimizer':copy.deepcopy(opt.state_dict())}
    def inputs(self):
        g=torch.Generator().manual_seed(32)
        a=F.normalize(torch.randn(2,16,3,generator=g),dim=-1)
        x=F.normalize(torch.randn(2,2,16,3,generator=g),dim=-1);x[:,0]=a
        return x,a,torch.tensor([.2,.8]),torch.tensor([0.,.1])
    def test_function_and_optimizer_preserved(self):
        ck=self.checkpoint();base,old=initialize(ck,False,11);net,opt=initialize(ck,True,11)
        x,a,t,time=self.inputs()
        with torch.no_grad():self.assertTrue(torch.equal(base.velocity(x,t,a,time),net.velocity(x,t,a,time)))
        for p,q in zip(old.param_groups[0]['params'],opt.param_groups[0]['params']):
            self.assertTrue(torch.equal(p,q))
            for k in ('step','exp_avg','exp_avg_sq'):self.assertTrue(torch.equal(old.state[p][k],opt.state[q][k]))
        self.assertTrue(all(p not in opt.state for p in opt.param_groups[1]['params']))
        self.assertEqual(len(net.blocks),8)
    def test_covariance_and_constraints_after_activation(self):
        net,_=initialize(self.checkpoint(),True,11);net.eval()
        for block in net.blocks[4:]:torch.nn.init.normal_(block.modulation[-1].bias,std=.05)
        x,a,t,time=self.inputs();q,_=torch.linalg.qr(torch.randn(3,3));q[:,0]*=torch.linalg.det(q)
        with torch.no_grad():
            v=net.velocity(x,t,a,time)
            w=net.velocity(x@q.T,t,a@q.T,time,frame=torch.eye(3)@q.T)
        self.assertTrue(torch.allclose(w,v@q.T,atol=3e-6,rtol=3e-5))
        self.assertLess(float((v*x).sum(-1).abs().max()),2e-6)
        self.assertTrue(torch.equal(v[:,0],torch.zeros_like(v[:,0])))
        with self.assertRaises(ValueError):net.velocity(x,t,a,time,temperature=0.)
        with self.assertRaises(ValueError):DeepNet().load_state_dict(self.checkpoint()['model'])
    def test_added_gates_receive_gradient(self):
        net,_=initialize(self.checkpoint(),True,11);x,a,t,time=self.inputs()
        net.velocity(x,t,a,time).square().mean().backward()
        self.assertTrue(all(float(b.modulation[-1].bias.grad.abs().sum())>0 for b in net.blocks[4:]))
