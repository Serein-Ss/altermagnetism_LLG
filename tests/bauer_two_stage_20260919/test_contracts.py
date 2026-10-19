import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from .models import InitialNet,Net,base_path,sample_initial,sample_path
from .common import cfg,seed_stream


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_initial_geometry_and_gradients(self):
        net=InitialNet(16,1).double();x=F.normalize(torch.randn(2,9,3,dtype=torch.double),dim=-1)
        tau=x.new_tensor([.2,.7]);axis=x.new_tensor([0.,0.,1.])
        q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.double));q[:,0]*=torch.linalg.det(q)
        v=net(x,tau);w=net(x@q,tau,axis@q)
        self.assertLess(float(abs(w-v@q).max()),1e-9)
        self.assertLess(float(abs((x*v).sum(-1)).max()),1e-9)
        v.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))

    def test_initial_reproducibility_sizes(self):
        net=InitialNet(16,1)
        for n in [16,25,32,50,100]:
            x=sample_initial(net,n,2,12,2);y=sample_initial(net,n,2,12,2)
            self.assertTrue(torch.equal(x,y));self.assertLess(float(abs(x.norm(dim=-1)-1).max()),1e-6)

    def test_path_anchor_and_noise_streams(self):
        net=Net('C',16,1);s=sample_initial(InitialNet(16,1),9,2,seed_stream(12,0),2)
        for kind in ['uniform','anchored']:
            t=torch.arange(3)*.1
            a=sample_path(net,s,t,seed_stream(12,1),2,kind,cfg())
            b=sample_path(net,s,t,seed_stream(12,1),2,kind,cfg())
            c=sample_path(net,s,t,seed_stream(12,2),2,kind,cfg())
            self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(a[:,0],s))
            self.assertFalse(torch.equal(a[:,1:],c[:,1:]));self.assertLess(float(abs(a.norm(dim=-1)-1).max()),1e-6)

    def test_seed_namespace(self):
        seeds=[seed_stream(910000+n*1000+split,j) for n in [16,25,32,50,100] for split in [1,2,3] for j in range(256)]
        self.assertEqual(len(seeds),len(set(seeds)))
        extra=[seed_stream(root+n*100+j,k) for root in [920000,970000] for n in [16,25,32,50,100] for j in range(6) for k in range(32)]
        extra += [seed_stream(930000+n*100+j,k) for n in [16,25,32] for j in range(12) for k in range(64)]
        self.assertEqual(len(seeds+extra),len(set(seeds+extra)))
        from .common import latent_seed
        path_seeds=[latent_seed(latent_seed(980002,b),k) for b in range(100) for k in range(100)]
        self.assertEqual(len(path_seeds),len(set(path_seeds)))

    def test_initial_interpolation_training(self):
        from scripts.model.gate_f import path_interpolate
        torch.manual_seed(1);net=InitialNet(16,1);opt=torch.optim.Adam(net.parameters(),lr=.005)
        y=F.normalize(torch.randn(2,7,3),dim=-1)
        z=F.normalize(torch.randn_like(y),dim=-1);dummy=torch.zeros_like(y);dummy[...,2]=1
        x,v,_=path_interpolate(torch.stack((dummy,z),1),torch.stack((dummy,y),1),torch.tensor([.3,.7]),torch.Generator().manual_seed(3))
        def loss():return (net(x[:,1],torch.tensor([.3,.7]))-v[:,1]).square().mean()
        before=float(loss().detach())
        for _ in range(20):opt.zero_grad();value=loss();value.backward();opt.step()
        self.assertLess(float(loss().detach()),before)
        self.assertGreater(float(v[:,1,0].norm()),0.)

    def test_rollout_seam(self):
        from .evaluate import rollout
        c=cfg();c['flow_steps']=2;net=Net('C',16,1)
        s=F.normalize(torch.randn(2,7,3),dim=-1)
        x=rollout(net,s,2.,10,'anchored',c)
        self.assertEqual(x.shape,(2,21,7,3));np.testing.assert_array_equal(x[:,0],s.numpy())
        self.assertLess(abs(np.linalg.norm(x,axis=-1)-1).max(),2e-6)

    def test_reference_integrity(self):
        from tests.bauer_active_data_20260918.run import simulate
        c=cfg();c['numerics']['save_ps']=.001
        s=np.zeros((2,7,3));s[...,2]=1.
        x,_=simulate(s,np.array([111,222]),c,.25,.002)
        self.assertEqual(x.shape,(2,3,7,3));np.testing.assert_array_equal(x[:,0],s)
        self.assertFalse(np.array_equal(x[0,-1],x[1,-1]))

    def test_missing_report_is_not_success(self):
        from . import report
        from .common import HERE
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            with patch.object(report,'HERE',Path(d)):
                report.main()
            text=(Path(d)/'assets/index.html').read_text()
            self.assertIn('尚无完整评估结果',text)

    def test_prepare_small_full_workflow(self):
        import h5py
        from . import prepare
        from .common import HERE
        c=cfg();c.update(train_sizes=[50],transfer_sizes=[],initial_counts={'train':2,'dev':2,'test':2},evaluation_parents=1,reference_horizons_ps=[.002])
        c['numerics'].update(preparation_ps=.002,save_ps=.001)
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            with patch.object(prepare,'HERE',Path(d)),patch.object(prepare,'cfg',return_value=c),patch.object(prepare,'allocation'):
                prepare.run(50)
            with h5py.File(Path(d)/'data/L50/test_future_0.h5') as h:
                self.assertEqual(h['spins'].shape,(32,3,50,3));self.assertTrue(h.attrs['complete'])
                self.assertEqual(len(set(h['seeds'][:])),32)

    def test_four_way_assessment_and_failure_gate(self):
        import h5py
        import json
        from . import evaluate
        from .common import HERE
        c=cfg();c.update(train_sizes=[16],transfer_sizes=[],evaluation_parents=2,reference_horizons_ps=[1.])
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            root=Path(d);(root/'data/L16').mkdir(parents=True);(root/'generated').mkdir()
            initial=np.zeros((64,16,3));initial[...,2]=1
            paths=np.broadcast_to(initial[:32,None],(32,11,16,3)).copy()
            with h5py.File(root/'data/L16/test_initial.h5','w') as h:h['spins']=initial
            with h5py.File(root/'generated/L16.h5','w') as h,h5py.File(root/'generated/GR_L16.h5','w') as g:
                h.attrs['complete']=g.attrs['complete']=True;h['initial']=initial
                for j in range(2):
                    h[f'RG_{j}']=paths[:16];h[f'GG_{j}']=paths[:16];g[f'paths_{j}']=paths
                    with h5py.File(root/f'data/L16/test_future_{j}.h5','w') as r:r['spins']=paths
            with patch.object(evaluate,'HERE',root),patch.object(evaluate,'cfg',return_value=c):evaluate.assess()
            self.assertTrue(json.loads((root/'assessment.json').read_text())['permit_long_diagnostic'])
            with h5py.File(root/'generated/L16.h5','r+') as h:h['RG_0'][:,1:]*=-1
            with patch.object(evaluate,'HERE',root),patch.object(evaluate,'cfg',return_value=c):evaluate.assess()
            self.assertFalse(json.loads((root/'assessment.json').read_text())['permit_long_diagnostic'])


if __name__=='__main__':unittest.main(verbosity=2)
