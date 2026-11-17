import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('stage_a', HERE / 'stage_a.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class Tests(unittest.TestCase):
    def test_eligibility(self):
        clean = {'uuid':'GPU-a','name':'NVIDIA A100-SXM4-80GB','total':81920,'free':81050,'util':0,'apps':[]}
        self.assertTrue(m.eligible(clean))
        self.assertTrue(m.same_empty(clean,dict(clean)))
        for change in ({'util':1},{'free':79000},{'apps':[(123,5)]},{'total':32000}):
            value = dict(clean, **change)
            self.assertFalse(m.eligible(value))
        self.assertFalse(m.same_empty(clean, dict(clean, uuid='GPU-b')))

    def test_snapshot_process_uuid_not_pid_namespace(self):
        responses = [
            subprocess.CompletedProcess([],0,'0, GPU-a, NVIDIA A100-SXM4-80GB, 81920, 81050, 0\n1, GPU-b, NVIDIA A100-SXM4-80GB, 81920, 50000, 2\n',''),
            subprocess.CompletedProcess([],0,'GPU-b, 900001, 12000\n','')
        ]
        with mock.patch.object(m.subprocess,'run',side_effect=responses):
            state = m.snapshot()
        self.assertTrue(m.eligible(state[0]))
        self.assertFalse(m.eligible(state[1]))
        self.assertEqual(state[1]['apps'],[(900001,12000)])

    def test_true_file_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            m.LOCK = Path(tmp) / 'pilot.lock'
            child = subprocess.Popen([sys.executable,'-u','-c',
                'import fcntl,sys,time;f=open(sys.argv[1],"a");fcntl.flock(f,fcntl.LOCK_EX);print("READY",flush=True);time.sleep(30)',
                str(m.LOCK)], stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(),'READY')
                self.assertEqual(m.lock_probe(),2)
            finally:
                child.kill(); child.wait(timeout=5); child.stdout.close()
            self.assertEqual(m.lock_probe(),0)

    def test_second_controller_rejection(self):
        done = subprocess.CompletedProcess([],2,b'{"error":"\xe5\xb7\xb2\xe6\x9c\x89\xe4\xbb\xbb\xe5\x8a\xa1\xe6\x8c\x81\xe6\x9c\x89\xe6\x8e\xa7\xe5\x88\xb6\xe5\x99\xa8\xe9\x94\x81"}',b'')
        with mock.patch.object(m.subprocess,'run',return_value=done) as f:
            self.assertTrue(m.second_controller_blocked(os.getpid(),0,'GPU-x'))
            self.assertIn('--internal-controller',f.call_args.args[0])
        with mock.patch.object(m.subprocess,'run',return_value=subprocess.CompletedProcess([],0,b'{}',b'')):
            self.assertFalse(m.second_controller_blocked(os.getpid(),0,'GPU-x'))

    def test_real_pdeathsig(self):
        # Real kernel parent-death signal, no CUDA allocation.
        child_code = ('import sys,time;sys.path.insert(0,sys.argv[1]);import stage_a;'
                      'stage_a.parent_death(int(sys.argv[2]));print("READY",flush=True);time.sleep(60)')
        parent_code = ('import os,subprocess,sys,time;'
                       'p=subprocess.Popen([sys.executable,"-u","-c",sys.argv[2],sys.argv[1],str(os.getpid())],stdout=subprocess.PIPE,text=True);'
                       'assert p.stdout.readline().strip()=="READY";print(p.pid,flush=True);time.sleep(60)')
        parent = subprocess.Popen([sys.executable,'-u','-c',parent_code,str(HERE),child_code],stdout=subprocess.PIPE,text=True)
        child_pid = None
        try:
            child_pid = int(parent.stdout.readline().strip())
            parent.kill(); parent.wait(timeout=5)
            parent.stdout.close()
            for _ in range(50):
                if m.exited_or_zombie(child_pid): break
                time.sleep(.1)
            self.assertTrue(m.exited_or_zombie(child_pid))
        finally:
            if parent.poll() is None: parent.kill(); parent.wait(timeout=5)
            # Do not signal a PID not held as a Popen object.

if __name__ == '__main__': unittest.main()
