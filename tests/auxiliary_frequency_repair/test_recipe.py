import unittest
from .recipe import call_due,weight_at,role_at,expected_calls


class ScheduleTests(unittest.TestCase):
    def test_counts(self):
        for arm,count in [('sparse64',32),('dense8_same_mass',256),('dense8_strong',256)]:
            self.assertEqual(sum(call_due(s,arm) for s in range(1,2049)),count)
            self.assertEqual(expected_calls(2048,arm),count)
            self.assertEqual(expected_calls(16,arm,True),2)

    def test_equal_block_budget_including_ramp(self):
        for start in range(0,2048,64):
            values={a:sum(weight_at(s,a) for s in range(start+1,start+65) if call_due(s,a))
                    for a in ('sparse64','dense8_same_mass','dense8_strong')}
            self.assertAlmostEqual(values['sparse64'],values['dense8_same_mass'],14)
            self.assertAlmostEqual(values['sparse64']*8,values['dense8_strong'],14)

    def test_shared_roles_and_dense_input_plan(self):
        self.assertEqual([role_at(s) for s in range(64,257,64)],[0,2,0,2])
        for end in range(64,2049,64):
            self.assertEqual({role_at(s) for s in range(end-56,end+1,8)},{role_at(end)})
            self.assertTrue(all(call_due(end,a) for a in ('sparse64','dense8_same_mass','dense8_strong')))
        self.assertEqual([role_at(s,True) for s in (8,16)],[0,2])


if __name__=='__main__':unittest.main()
