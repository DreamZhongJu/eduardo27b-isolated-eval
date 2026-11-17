import json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
import b_stage as b
import cuda_map

class Tests(unittest.TestCase):
 def test_native_b_config(self):
  args=b.training_overrides(Path('/tmp/w'),Path('/tmp/r'),'student',Path('/tmp/i'))
  for s in ('data.train_batch_size=2','actor_rollout_ref.actor.ppo_mini_batch_size=2','actor_rollout_ref.rollout.n=4',
            'custom_reward_function.reward_kwargs.num_student_attempts=4','actor_rollout_ref.actor.optim.lr=1e-5',
            'trainer.total_training_steps=2','trainer.test_freq=0','trainer.save_freq=1',
            'actor_rollout_ref.rollout.agent.num_workers=2','reward.num_workers=2','reward_model.num_workers=null',
            'custom_reward_function.reward_kwargs.judge_concurrency=16','custom_reward_function.reward_kwargs.student_concurrency=16',
            'actor_rollout_ref.rollout.gpu_memory_utilization=0.58','actor_rollout_ref.rollout.calculate_log_probs=true'):
   self.assertIn(s,args)
  self.assertFalse(any('b_guard' in s or 'TELEMETRY' in s or 'TRACE_DIR' in s for s in args))
  self.assertFalse(any('temperature' in s or 'max_tokens_per_turn' in s for s in args))
 def test_cuda_reorder_verified(self):
  calls=[]
  def runner(cmd,env,**kw):
   calls.append((cmd,env))
   if cmd[-1]=='--show':out={'0':'GPU-B','1':'GPU-A','2':'GPU-C'}
   else:out=['GPU-A','GPU-B']
   return SimpleNamespace(stdout=json.dumps(out))
  self.assertEqual(cuda_map.verified('/python',['GPU-A','GPU-B'],{},runner),[1,0])
  self.assertEqual(calls[-1][1]['CUDA_VISIBLE_DEVICES'],'1,0')
 def test_cuda_existing_limit_fails_closed(self):
  with self.assertRaises(RuntimeError):cuda_map.verified('/python',['GPU-A'],{'CUDA_VISIBLE_DEVICES':'0'},lambda *a,**k:None)
 def test_cuda_runtime_mismatch_fails_closed(self):
  def runner(cmd,env,**kw):return SimpleNamespace(stdout=json.dumps({'0':'GPU-A'} if cmd[-1]=='--show' else ['GPU-WRONG']))
  with self.assertRaises(RuntimeError):cuda_map.verified('/python',['GPU-A'],{},runner)
 def test_log_fallback_categories_and_split(self):
  for message in (b'gave up after 3 retries, using fallback',b'non-retriable 400 using fallback',b'EduardoInteraction: student call failed hard'):
   with tempfile.TemporaryDirectory() as d:
    root=Path(d);(root/'trainer.log').write_bytes(message[:9]);sc=b.LogScanner(root,root/'ray')
    self.assertFalse(sc.scan())
    with (root/'trainer.log').open('ab') as f:f.write(message[9:])
    self.assertTrue(sc.scan())
 def test_old_log_not_scanned(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'trainer.log').write_text('ok');(root/'old.log').write_text('gave up after x using fallback')
   self.assertFalse(b.LogScanner(root,root/'ray').scan())
 def test_rotated_ray_worker_and_compressed_detection(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); logs=root/'ray/session_latest/logs';logs.mkdir(parents=True)
   (logs/'worker-aaa.err.1').write_text('gave up after 3 retries using fallback')
   self.assertTrue(b.LogScanner(root,root/'ray').scan())
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); logs=root/'ray/session_latest/logs';logs.mkdir(parents=True)
   (logs/'worker-aaa.err.1.gz').write_bytes(b'x')
   with self.assertRaises(RuntimeError):b.LogScanner(root,root/'ray').scan()
 def test_two_services(self):
  for role,maxs,mem in [('student','64','0.4'),('judge','256','0.9')]:
   cmd=b.service_command(role,'student');self.assertEqual(cmd[cmd.index('--max-num-seqs')+1],maxs)
   self.assertEqual(cmd[cmd.index('--gpu-memory-utilization')+1],mem)
if __name__=='__main__':unittest.main(verbosity=2)
