import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from data import PathWindows


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.tmp.name)
        self.rows = []
        for split in ('train', 'dev', 'test'):
            for arm in ('base', 'fine'):
                name = f'{split}_{arm}.h5'
                parent = f'L4_{split}'
                with h5py.File(self.root / name, 'w') as h:
                    h.attrs.update(complete=True, arm=arm,
                                   parent_json=json.dumps(dict(parent_id=parent, split=split)))
                    x = np.zeros((2, 11, 4, 3), dtype='float32')
                    x[..., 2] = 1
                    h['spins'] = x
                    h['time_ps'] = np.arange(11)*.1
                    h['seeds'] = [31, 32]
                self.rows.append(dict(file=name, parent_id=parent, split=split,
                                      arm=arm, branches=2, frames=11, family='positive'))
        self.write_manifest()
        (self.root / 'complete.json').write_text('{}')

    def tearDown(self):
        self.tmp.cleanup()

    def write_manifest(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.rows))

    def test_lazy_windows_and_time(self):
        data = PathWindows([self.root], 'train', .2, 2)
        self.assertEqual(len(data), 10)
        sample = data[1]
        self.assertEqual(sample['spins'].shape, (3, 4, 3))
        self.assertAlmostEqual(sample['origin_ps'], .2)
        np.testing.assert_allclose(sample['time_ps'], [0, .1, .2])
        np.testing.assert_array_equal(sample['initial'], sample['spins'][0])
        self.assertEqual({data[i]['seed'] for i in range(len(data))}, {31, 32})
        self.assertEqual(data[-1]['parent_id'], 'L4_train')
        with self.assertRaises(IndexError):
            data[len(data)]

    def test_split_and_fine_exclusion(self):
        for split in ('train', 'dev', 'test'):
            data = PathWindows([self.root], split, 1)
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]['split'], split)
            self.assertTrue(data.records[0][0].name.endswith('_base.h5'))

    def test_parent_leakage_rejected(self):
        self.rows[2]['parent_id'] = 'L4_train'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'leakage'):
            PathWindows([self.root], 'train', .2)

    def test_invalid_duration(self):
        for duration in (0, .15, 2):
            with self.assertRaises(ValueError):
                PathWindows([self.root], 'train', duration)

    def test_incomplete(self):
        with h5py.File(self.root / 'train_base.h5', 'r+') as h:
            h.attrs['complete'] = False
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            PathWindows([self.root], 'train', .2)

    def test_bad_spin_read_rejected(self):
        data = PathWindows([self.root], 'train', .2)
        with h5py.File(self.root / 'train_base.h5', 'r+') as h:
            h['spins'][0, 0, 0] = [0, 0, 0]
        with self.assertRaisesRegex(ValueError, 'invalid spin'):
            data[0]


if __name__ == '__main__':
    unittest.main()
