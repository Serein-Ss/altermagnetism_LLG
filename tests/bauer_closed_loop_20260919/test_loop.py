import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from .model import Net, generate, neighbor, coarse_mean
from .metrics import compare, summary
from .run import config, loss_for


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_geometry_and_gradients_all_models(self):
        for kind in 'ABCD':
            torch.manual_seed(51)
            net = Net(kind, 16, 2).double()
            x = F.normalize(torch.randn(2, 4, 7, 3, dtype=torch.float64), dim=-1)
            q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=x.dtype))
            q[:, 0] *= torch.linalg.det(q)
            axis = x.new_tensor([0., 0., 1.])
            tau, times = x.new_tensor([.2, .7]), x.new_tensor([0., .1, .2, .3])
            y = net(x, tau, x[:, 0], times, 63., axis)
            rotated = net(x@q, tau, x[:, 0]@q, times, 63., axis@q)
            self.assertLess(float((rotated-y@q).abs().max()), 1e-9)
            self.assertLess(float((y-net(x, tau, x[:, 0], times, 63., -axis)).abs().max()), 1e-9)
            self.assertLess(float((x*y).sum(-1).abs().max()), 1e-9)
            self.assertTrue(torch.equal(y[:, 0], torch.zeros_like(y[:, 0])))
            y.square().mean().backward()
            for p in net.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
            if kind == 'C':
                self.assertGreater(float(net.blocks[0].scale.weight.grad.abs().sum()), 0.)

    def test_open_neighbors_and_coarse_magnitude(self):
        x = torch.arange(5.).reshape(1, 1, 5, 1)
        self.assertTrue(torch.equal(neighbor(x).flatten(), torch.tensor([1., 1., 2., 3., 3.])))
        v = torch.tensor([[[[1., 0., 0.], [-1., 0., 0.], [0., 0., 1.]]]])
        c = coarse_mean(v)
        self.assertTrue(torch.equal(c[0, 0, 0], torch.zeros(3)))
        self.assertTrue(torch.equal(c[0, 0, 1], v[0, 0, 2]))

    def test_sampler_constraints_reproducibility(self):
        for kind in 'ABCD':
            torch.manual_seed(4)
            net = Net(kind, 16, 1)
            initial = F.normalize(torch.randn(2, 9, 3), dim=-1)
            times = torch.tensor([0., .1, .2])
            a = generate(net, initial, times, 63., 61, 3)
            b = generate(net, initial, times, 63., 61, 3)
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(a[:, 0], initial))
            self.assertLess(float(abs(a.norm(dim=-1)-1).max()), 5e-7)

    def test_loss_and_optimizer(self):
        cfg = config()
        cfg['temperature_K'] = 0.
        for kind in 'ABCD':
            torch.manual_seed(20)
            net = Net(kind, 16, 1)
            opt = torch.optim.Adam(net.parameters(), lr=.003)
            y = F.normalize(torch.randn(2, 3, 6, 3), dim=-1)
            times = torch.tensor([0., .1, .2])
            fixed = lambda: loss_for(net, y, times, cfg, torch.Generator().manual_seed(42))
            before = float(fixed())
            for _ in range(20):
                opt.zero_grad(); loss = fixed(); loss.backward(); opt.step()
            self.assertLess(float(fixed()), before)

    def test_metrics_and_screen(self):
        rng = np.random.default_rng(55)
        s = rng.normal(size=(16, 11, 8, 3))
        s /= np.linalg.norm(s, axis=-1, keepdims=True)
        values = compare(s, s)
        self.assertLess(max(values.values()), 1e-12)
        rows = [dict(values, L=16, family='test') for _ in range(2)]
        stats = summary(rows, config()['margins'], 20)
        self.assertTrue(all(v['status'] == 'WITHIN_PILOT_MARGIN' for v in stats.values()))

    def test_report_with_missing_runs_and_synthetic_data(self):
        from . import report
        from .run import HERE
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root = Path(tmp)
            audit = dict(integrity_pass=True, files=1, parents=1, max_stored_norm_error=0.,
                         reference_status='SYNTHETIC_TEST', random_baseline_sigma=.1)
            (root/'audit.json').write_text(json.dumps(audit))
            row = dict(parent_id='synthetic', L=16, family='synthetic')
            s = np.zeros((16, 11, 16, 3)); s[..., 2] = 1.
            with patch.object(report, 'HERE', root), patch.object(report, 'records', return_value=[row]), \
                 patch.object(report, 'reference', return_value=(s, np.arange(11)*.1, s[0, 0])):
                report.main()
            result = json.loads((root/'assets/report.json').read_text())
            self.assertEqual(result['status'], 'INCOMPLETE_RUNS')
            page = (root/'assets/index.html').read_text()
            self.assertIn('data:image/png;base64,', page)
            self.assertIn('data:application/json;base64,', page)
            with Image.open(root/'assets/metrics.png') as im:
                self.assertEqual(im.format, 'PNG')

    def test_streaming_digest(self):
        from .audit import digest
        from .run import HERE
        import hashlib
        source = HERE/'config.yaml'
        self.assertEqual(digest(source), hashlib.sha256(source.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main(verbosity=2)
