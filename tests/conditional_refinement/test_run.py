import unittest
import numpy as np
import torch
from tests.bauer_closed_loop_20260919.run import batch as legacy_batch
from .run import indices, get_batch, lr_at, tasks


class FakeData:
    def __init__(self):
        self.ends=[24,48]
        self.records=[(None,None,range(3),10)]*2
    def __len__(self):return 48
    def __getitem__(self,i):
        rng=np.random.default_rng(i);s=rng.normal(size=(11,16,3)).astype('f4')
        s/=np.linalg.norm(s,axis=-1,keepdims=True)
        return dict(spins=s)


class Tests(unittest.TestCase):
    def test_legacy_exact(self):
        config=dict(batch=8,save_ps=.1);data=[FakeData(),FakeData()]
        a=np.random.default_rng(11);b=np.random.default_rng(11)
        for _ in range(12):
            y,t=legacy_batch(data,a,config,'cpu','B')
            z,u,_=get_batch(data,b,config,.5,'cpu')
            torch.testing.assert_close(y,z,rtol=0,atol=0)
            torch.testing.assert_close(t,u,rtol=0,atol=0)
        self.assertEqual(a.bit_generator.state,b.bit_generator.state)

    def test_anchor_preserves_branch_and_rng(self):
        data=[FakeData()];a=np.random.default_rng(7);b=np.random.default_rng(7)
        _,original=indices(data,a,200,0.)
        _,anchors=indices(data,b,200,1.)
        self.assertEqual(a.bit_generator.state,b.bit_generator.state)
        self.assertTrue(all(i//3==j//3 and j%3==0 for i,j in zip(original,anchors)))

    def test_probability(self):
        _,ids=indices([FakeData()],np.random.default_rng(2),20000,.9)
        self.assertLess(abs(np.mean(np.array(ids)%3==0)-(.9+.1/3)),.01)
        with self.assertRaises(ValueError):indices([FakeData()],np.random.default_rng(),1,2.)

    def test_accumulation_gradient(self):
        torch.manual_seed(1);model=torch.nn.Linear(4,3);x=torch.randn(32,4);y=torch.randn(32,3)
        ((model(x)-y)**2).mean().backward()
        expected=[p.grad.clone() for p in model.parameters()];model.zero_grad(set_to_none=True)
        for a,b in zip(x.chunk(4),y.chunk(4)):(((model(a)-b)**2).mean()/4).backward()
        for p,g in zip(model.parameters(),expected):torch.testing.assert_close(p.grad,g)

    def test_schedule_and_tasks(self):
        self.assertAlmostEqual(lr_at(1,6000),1e-4);self.assertAlmostEqual(lr_at(6000,6000),3e-6)
        self.assertEqual(len(tasks()),9)
        for seed in (11,22,33):
            self.assertEqual({r['repair'] for r in tasks() if r['seed']==seed},{'control','anchor90','accum4'})


if __name__=='__main__':unittest.main()
