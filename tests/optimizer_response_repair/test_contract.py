import copy
import unittest
import torch
from .diagnostic import optimizer,rate
from .run import tasks,source

class Contract(unittest.TestCase):
    def test_optimizer(self):
        net=torch.nn.Linear(2,1)
        old=torch.optim.AdamW(net.parameters(),lr=1e-3)
        net(torch.ones(3,2)).sum().backward();old.step()
        weights=copy.deepcopy(net.state_dict());state=copy.deepcopy(old.state_dict())
        for arm in ('restore_low','reset_low','restore_high','reset_high'):
            opt=optimizer(net,copy.deepcopy(state),arm)
            self.assertEqual(len(opt.state),2 if arm.startswith('restore') else 0)
            for k,v in weights.items():self.assertTrue(torch.equal(net.state_dict()[k],v))
            if opt.state:self.assertEqual({int(x['step']) for x in opt.state.values()},{1})
    def test_schedule(self):
        for step in (1,128,512):
            self.assertAlmostEqual(rate('reset_high',step,512)/rate('reset_low',step,512),10.)
        self.assertAlmostEqual(rate('restore_low',1,512),3e-5)
        self.assertAlmostEqual(rate('restore_low',512,512),3e-6)
    def test_sources(self):
        self.assertEqual(len(tasks()),12)
        for seed,index in ((11,'01'),(22,'03'),(33,'05')):
            self.assertEqual(source(seed).name,index)
        self.assertTrue(all(t['updates']==512 for t in tasks()))
