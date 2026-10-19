import unittest
import torch
from tests.local_condition_repair.model import initialize
from tests.local_condition_repair.run import source
from .run import set_rates,ARMS
from .diagnostics import disabled,ablation_rows


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ck=torch.load(source(11)/'checkpoint.pt',map_location='cpu',weights_only=True)

    def test_rates_and_original_adam(self):
        for arm in ARMS:
            net,opt=initialize(self.ck,arm.split('_')[0],11,'cpu')
            for lr in (3e-5,3e-6):
                set_rates(opt,lr,arm)
                self.assertEqual(opt.param_groups[0]['lr'],lr)
                self.assertEqual(opt.param_groups[1]['lr'],lr*(10 if arm.endswith('fast') else 1))
            self.assertEqual({int(v['step']) for v in opt.state.values()},{30000})

    def test_head_intervention_restores_on_exception(self):
        net,_=initialize(self.ck,'neighbor',11,'cpu')
        torch.nn.init.normal_(net.readout[-1].weight,std=.03)
        x=torch.randn(2,31);before=net.readout(x).detach()
        with self.assertRaises(RuntimeError):
            with disabled(net):
                self.assertEqual(float(net.readout(x).abs().max()),0.)
                raise RuntimeError('test cleanup')
        torch.testing.assert_close(net.readout(x),before,atol=0,rtol=0)

    def test_ablation_independent_parents(self):
        rows=ablation_rows()
        self.assertEqual(len({r['parent']['parent_id'] for r in rows}),6)
        self.assertEqual(sorted([r['parent']['family'] for r in rows]),['interior_domain']*2+['single_wall']*2+['tilted']*2)
        self.assertTrue(all(r['origin']=='llg' for r in rows))


if __name__=='__main__':unittest.main()
