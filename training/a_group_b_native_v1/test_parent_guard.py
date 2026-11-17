import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import parent_guard

HERE=Path(__file__).resolve().parent

def wait_until(fn, seconds=12):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        if fn(): return True
        time.sleep(.05)
    return False

def alive(pid):
    try: return parent_guard.stat_of(pid)['state'] != 'Z'
    except (FileNotFoundError,ProcessLookupError): return False

class Test(unittest.TestCase):
    def test_sigkill_controller_kills_owned_detached_descendants_only(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); ready=root/'ready.json'; service=root/'service.py'; control=root/'controller.py'
            service.write_text('''import os,signal,subprocess,sys,time,json\nfrom pathlib import Path\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ngrand=subprocess.Popen([sys.executable,"-c","import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"],start_new_session=True)\nPath(sys.argv[1]).write_text(json.dumps({"service":os.getpid(),"grandchild":grand.pid,"guard":os.getppid()}))\nwhile True: time.sleep(1)\n''')
            control.write_text('''import os,subprocess,sys,time\np=subprocess.Popen([sys.executable,sys.argv[1],"--parent",str(os.getpid()),"--",sys.executable,sys.argv[2],sys.argv[3]],start_new_session=True)\nwhile True: time.sleep(1)\n''')
            owner=subprocess.Popen([sys.executable,str(control),str(HERE/'parent_guard.py'),str(service),str(ready)])
            outsider=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
            identities={}
            try:
                self.assertTrue(wait_until(ready.is_file,5),'task tree did not start')
                ids=json.loads(ready.read_text())
                identities={pid:parent_guard.stat_of(pid)['start'] for pid in ids.values()}
                owner.kill(); owner.wait(timeout=3)
                self.assertTrue(wait_until(lambda: not any(alive(pid) for pid in ids.values()),12),'owned descendants did not all exit')
                self.assertIsNone(outsider.poll())
            finally:
                if owner.poll() is None: owner.kill(); owner.wait(timeout=3)
                for pid,start in identities.items():
                    if alive(pid): parent_guard.send_owned(pid,start,signal.SIGKILL)
                outsider.terminate(); outsider.wait(timeout=3)

if __name__=='__main__': unittest.main(verbosity=2)
