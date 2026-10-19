import unittest
import numpy as np
import torch
from torch.nn import functional as F
from .run import branch_stats, sensitivity


class ProbeTests(unittest.TestCase):
    def test_exact_and_zero(self):
        x=torch.randn(8,10,16,3)
        a=branch_stats(x,x);b=branch_stats(torch.zeros_like(x),x)
        torch.testing.assert_close(a['mse'],torch.zeros(8,10))
        torch.testing.assert_close(b['mse'],b['target_power'])
        torch.testing.assert_close(a['prediction_power'],a['cross'])

    def test_sensitivity_dependency(self):
        class Local:
            def velocity(self,x,*args):return x
        class Remote:
            def velocity(self,x,*args):return x[:,9:10].expand_as(x)
        torch.manual_seed(11);x=F.normalize(torch.randn(2,11,4,3,dtype=torch.float64),dim=-1)
        noise=torch.randn_like(x[:,0]);args=(x,torch.ones(2),x[:,0],torch.arange(11),1,9,noise,.001)
        np.testing.assert_allclose(sensitivity(Local(),*args),0,atol=1e-12)
        np.testing.assert_allclose(sensitivity(Remote(),*args),1,atol=1e-6)

    def test_real_inputs(self):
        import json,h5py
        from .run import HERE,ROOT,P2,Model,path_interpolate
        torch.set_num_threads(1)
        rows=json.loads((HERE/'protocol.json').read_text())['parents']
        self.assertEqual(len(rows),72)
        self.assertEqual(len({r['parent_id'] for r in rows}),72)
        with h5py.File(ROOT/rows[0]['path']) as h:
            y=F.normalize(torch.tensor(h['spins'][np.arange(8),:11]),dim=-1)
        ck=torch.load(P2/'runs/seed_11/best.pt',map_location='cpu',weights_only=False)
        net=Model(ck['config']);net.load_state_dict(ck['model']);net.eval()
        gen=torch.Generator().manual_seed(13)
        z=F.normalize(torch.randn(y.shape,generator=gen),dim=-1);z[:,0]=y[:,0]
        tau=torch.full((8,),.5);x,target,_=path_interpolate(z,y,tau,gen)
        with torch.no_grad():pred=net.velocity(x,tau,y[:,0],torch.arange(11)*.1)
        a=branch_stats(pred[:,1:],target[:,1:])
        self.assertEqual(a['mse'].shape,(8,10))
        self.assertTrue(all(torch.isfinite(v).all() for v in a.values()))
