import unittest
import copy
import numpy as np
import torch
from .run import tasks,budget,selection,backward_mean


class Tests(unittest.TestCase):
    def test_budgets(self):
        t=tasks();self.assertEqual(len(t),9)
        self.assertEqual([budget(x) for x in t[:3]],[(6000,1,6000),(750,8,6000),(6000,8,48000)])
        self.assertEqual([budget(x,True) for x in t[:3]],[(16,1,16),(2,8,16),(16,8,128)])
    def test_mean_gradient(self):
        a=torch.tensor([.2,.8],requires_grad=True);b=a.detach().clone().requires_grad_()
        targets=[torch.arange(n,dtype=torch.float32) for n in (2,3,4,5,6,7,8,9)]
        losses=[]
        for target in targets:
            backward_mean(((a[0]*target-a[1])**2).mean(),8)
            backward_mean((a+target.mean()).square().mean(),8)
            losses.append(((b[0]*target-b[1])**2).mean()+(b+target.mean()).square().mean())
        torch.stack(losses).mean().backward();torch.testing.assert_close(a.grad,b.grad)
    def test_adam_once(self):
        a=torch.nn.Parameter(torch.tensor([.2,.8]));b=torch.nn.Parameter(a.detach().clone())
        oa=torch.optim.AdamW([a],lr=3e-5);ob=torch.optim.AdamW([b],lr=3e-5)
        for target in range(8):backward_mean((a-target).square().mean(),8)
        torch.stack([(b-target).square().mean() for target in range(8)]).mean().backward()
        torch.nn.utils.clip_grad_norm_([a],10.);torch.nn.utils.clip_grad_norm_([b],10.)
        oa.step();ob.step();torch.testing.assert_close(a,b)
        self.assertEqual(int(oa.state[a]['step']),1)
    def test_selection_pairing_and_restore(self):
        values=[]
        for task in tasks()[:3]:
            rng=np.random.default_rng(42);state=copy.deepcopy(rng.bit_generator.state)
            ids,branches,w=selection(task['repair'],1,rng);rng.bit_generator.state=state
            ids2,branches2,w2=selection(task['repair'],1,rng)
            np.testing.assert_array_equal(ids,ids2);np.testing.assert_array_equal(branches,branches2)
            np.testing.assert_array_equal(np.sort(ids),np.arange(48));self.assertEqual(w,1.)
            values.append((ids,branches))
        for v in values[1:]:
            for a,b in zip(v,values[0]):np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
