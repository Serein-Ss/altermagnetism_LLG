import unittest
from .run import tasks,ARMS,banks

class Tests(unittest.TestCase):
    def test_task_matrix_and_train_banks(self):
        tt=tasks();self.assertEqual(len(tt),9)
        self.assertEqual({t['repair'] for t in tt},set(ARMS))
        self.assertEqual({t['seed'] for t in tt},{11,22,33})
        bb,hh=banks(True)
        self.assertEqual([len(x) for x in bb],[2,2,2])
        self.assertEqual(len(hh),6)
        self.assertTrue(all(name.endswith('_train.h5') for name in hh))
        self.assertTrue(all(x.shape==(64,2,16,3) for bank in bb for x in bank))
