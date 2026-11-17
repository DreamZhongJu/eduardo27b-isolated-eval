import importlib.util, os, tempfile, unittest
from pathlib import Path
p = Path(__file__).with_name('resources.py')
spec=importlib.util.spec_from_file_location('resources_b',p)
r=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=r
spec.loader.exec_module(r)
class Clock:
 def __init__(self): self.t=0
 def mono(self): return self.t
 def wall(self): return self.t+1_000_000
 def sleep(self,n): self.t+=n
class Proc:
 counter=100
 def __init__(self): Proc.counter+=1; self.pid=Proc.counter; self.dead=False; self.stdout=None
 def poll(self): return 0 if self.dead else None
 def terminate(self): self.dead=True
 def wait(self,timeout=None): return 0
 def kill(self): self.dead=True
class A:
 @staticmethod
 def eligible(g): return g['free']>=80900 and g['util']==0 and not g['apps']
 @staticmethod
 def own_lock(): return os.open('/dev/null',os.O_RDONLY)
def snap(n=4):
 return {i:{'uuid':f'GPU-{i}','name':'A100','total':81920,'free':81050,'util':0,'apps':[]} for i in range(n)}
class Test(unittest.TestCase):
 def manager(self,**kw):
  d=tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); c=Clock()
  def spawn(b, timeout): return r.Holder(Proc(),b,30000,c.wall(),c.mono())
  obj=r.GPUResources(Path(d.name),A(),snapshot=lambda:snap(),spawn=spawn,monotonic=c.mono,wall=c.wall,sleep=c.sleep,jitter=lambda a,b: (self.assertEqual((a,b),(300,900)) or 300),**kw)
  obj.acquire_lock(); self.addCleanup(obj.close); return obj,c
 def test_multigpu_and_handoff(self):
  obj,c=self.manager(); log=[]
  class CB:
   def launch_service(self,role,b): log.append('start_'+role); return Proc()
   def service_ready(self,role,b,p): log.append('healthy_'+role); return True
   def launch_tutor(self,b): log.append('start_tutor'); return Proc()
   def stop_started(self): log.append('stop')
  obj.acquire_and_handoff(CB()); self.assertEqual(len(obj.holders),0)
  self.assertEqual([x['action'] for x in obj.events if x['action']=='已预留'],['已预留']*4)
  self.assertLess(log.index('healthy_student'),log.index('start_tutor'))
  self.assertLess(log.index('healthy_judge'),log.index('start_tutor'))
  self.assertEqual(len({x['gpu'] for x in obj.events if x['action']=='已预留'}),4)
  handoff=[(x['role'],x['action']) for x in obj.events if x['action'] != '已预留']
  self.assertEqual(handoff,[('judge','释放占位'),('judge','启动服务'),('judge','服务健康'),('student','释放占位'),('student','启动服务'),('student','服务健康'),('tutor_1','释放占位'),('tutor_2','释放占位'),('tutor','启动训练')])
 def test_timeout_release_and_jitter(self):
  obj,c=self.manager(deadline_seconds=2700,poll_seconds=30)
  rounds=[]
  def snapshot():
   if c.t>=3000: return snap(4)
   return snap(2)
  obj.snapshot=snapshot
  result=obj.acquire(); self.assertEqual(len(result),4); self.assertEqual(obj.round_no,2)
  self.assertTrue(any(x['action']=='释放占位' for x in obj.events)); self.assertGreaterEqual(c.t,3000)
 def test_three_handoff_rechecks_fail(self):
  obj,c=self.manager(); obj._after_release_ready=lambda b:False
  class CB:
   def launch_service(self,*a): self.fail('must not start')
   def stop_started(self): pass
  with self.assertRaises(r.HandoffError): obj.acquire_and_handoff(CB())
  self.assertEqual(obj.handoff_failures,3); self.assertFalse(obj.holders)
 def test_service_failure_is_not_retried(self):
  obj,c=self.manager(); calls=[]
  class CB:
   def launch_service(self,*args): calls.append('launch'); return None
   def stop_started(self): pass
  with self.assertRaises(r.ServiceError): obj.acquire_and_handoff(CB())
  self.assertEqual(calls,['launch']); self.assertEqual(obj.handoff_failures,0); self.assertFalse(obj.holders)
if __name__=='__main__': unittest.main()
