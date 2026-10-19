import unittest
import torch
from torch.nn import functional as F
from .run import trace,generate,TAUS,Model,checkpoints


class TraceTests(unittest.TestCase):
    def test_matches_frozen_sampler(self):
        torch.set_num_threads(1);torch.manual_seed(17)
        ck=torch.load(checkpoints(11)['original'],map_location='cpu',weights_only=False)
        net=Model(ck['config']);net.load_state_dict(ck['model']);net.eval()
        initial=F.normalize(torch.randn(2,16,3),dim=-1)
        source=F.normalize(torch.randn(2,11,16,3),dim=-1);source[:,0]=initial
        times=torch.arange(11)*.1
        a=trace(net,source,initial,times,16)
        b=generate(net,initial,times,torch.Generator().manual_seed(1),16,source=source)
        torch.testing.assert_close(a[-1],b,atol=0,rtol=0)
        torch.testing.assert_close(a[0],source,atol=0,rtol=0)
        torch.testing.assert_close(a[:,:,0],initial.expand(len(TAUS),-1,-1,-1),atol=0,rtol=0)

    def test_static_transport(self):
        class Zero:
            def velocity(self,x,*args):return torch.zeros_like(x)
        source=F.normalize(torch.randn(2,11,4,3,dtype=torch.float64),dim=-1)
        result=trace(Zero(),source,source[:,0],torch.arange(11),16)
        torch.testing.assert_close(result,source.expand(len(TAUS),-1,-1,-1,-1),atol=1e-14,rtol=0)
