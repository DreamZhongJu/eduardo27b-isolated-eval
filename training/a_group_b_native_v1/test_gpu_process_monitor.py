from gpu_process_monitor import sample
import unittest
class Tests(unittest.TestCase):
 def test_verified_descendant(self):
  rows={10:{'ppid':1,'hostpid':1010,'start':17},11:{'ppid':10,'hostpid':1011,'start':18},20:{'ppid':1,'hostpid':1020,'start':19}}
  x=sample({'trainer':[{'pid':10,'start':17}]},{0:{'apps':[(1011,500)],'used_mib':1370}},rows)['0']
  self.assertTrue(x['verified']); self.assertEqual(x['mib'],500)
 def test_unknown_external_pid(self):
  rows={10:{'ppid':1,'hostpid':1010,'start':17}}
  x=sample({'trainer':[{'pid':10,'start':17}]},{0:{'apps':[(1010,500),(3000,200)],'used_mib':1570}},rows)['0']
  self.assertFalse(x['verified']); self.assertIsNone(x['mib']); self.assertEqual(x['known_lower_bound_mib'],500)
 def test_missing_roots(self):
  x=sample({'trainer':[{'pid':10,'start':17}]},{0:{'apps':[],'used_mib':870}},{})['0']; self.assertFalse(x['verified'])
 def test_ambiguous_pid(self):
  rows={10:{'ppid':1,'hostpid':1010,'start':17},11:{'ppid':10,'hostpid':1010,'start':17}}
  self.assertFalse(sample({'trainer':[{'pid':10,'start':17}]},{0:{'apps':[(1010,500)],'used_mib':1370}},rows)['0']['verified'])
if __name__=='__main__': unittest.main()
