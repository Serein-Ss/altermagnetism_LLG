import unittest
import numpy as np
import torch
from tests.training_restart_repair.data import rows
from tests.intermediate_restart_repair.run import selected
from tests.transition_block_repair.model import TransitionNet
from .run import ARMS,mask,loss,source


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_partition_and_source_separation(self):
        meta=rows();parts=[mask(a,meta) for a in ARMS[1:]]
        self.assertEqual([int(p.sum()) for p in parts],[16,16,16])
        np.testing.assert_array_equal(np.sum(parts,axis=0),np.ones(48))
        self.assertFalse({r['parent']['parent_id'] for r in meta}&{r['parent']['parent_id'] for r in selected()})
        for a in ARMS[1:]:
            self.assertEqual(len({r['parent']['parent_id'] for r in meta if r['parent']['family']==a}),8)
        with self.assertRaises(ValueError):mask('unknown',meta)

    def test_loss_and_gradient_partition(self):
        torch.manual_seed(1);pred=torch.randn(48,2,7,3,requires_grad=True);target=torch.randn_like(pred)
        whole=loss(pred,target);gradient=torch.autograd.grad(whole,pred,retain_graph=True)[0]
        terms=[];grads=[]
        for a in ARMS[1:]:
            keep=torch.tensor(mask(a,rows()));term=loss(pred[keep],target[keep]);terms.append(term)
            g=torch.autograd.grad(term,pred,retain_graph=True)[0];grads.append(g)
            self.assertEqual(float(g[~keep].abs().max()),0.)
        torch.testing.assert_close(sum(terms),whole)
        torch.testing.assert_close(sum(grads),gradient)
        self.assertEqual(float(gradient[:,0].abs().max()),0.)

    def test_exposure_and_paired_future_selection(self):
        meta=rows();rng=np.random.default_rng(951011);counts=np.zeros((4,48,64),dtype=int)
        for _ in range(100):
            branch=rng.integers(64,size=48)
            for j,a in enumerate(ARMS):
                keep=mask(a,meta);np.add.at(counts[j],(np.flatnonzero(keep),branch[keep]),1)
        for j,a in enumerate(ARMS[1:],1):
            keep=mask(a,meta);np.testing.assert_array_equal(counts[j,keep],counts[0,keep])
            self.assertFalse(counts[j,~keep].any())

    def test_source_checkpoint_and_adam(self):
        for seed in (11,22,33):
            ck=torch.load(source(seed)/'checkpoint.pt',map_location='cpu',weights_only=True)
            self.assertEqual(ck['step'],36000);self.assertEqual(ck['task']['repair'],'baseline')
            net=TransitionNet();net.load_state_dict(ck['model']);opt=torch.optim.AdamW(net.parameters());opt.load_state_dict(ck['optimizer'])
            self.assertEqual({int(x['step']) for x in opt.state.values()},{36000})


if __name__=='__main__':unittest.main()
