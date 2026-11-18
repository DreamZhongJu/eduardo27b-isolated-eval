import json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import b_stage as b
import cuda_map

class Tests(unittest.TestCase):
 def test_interaction_works_without_worker_user(self):
  import yaml
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); source=root/'eduardo'; source.mkdir()
   original={'interaction':[{'name':'eduardo','class_name':'eduardo.eduardo_interaction.EduardoInteraction',
     'config':{'student_model':'Llama-3.1-8B-Instruct-${oc.env:USER}',
       'student_api_base':'${oc.env:STUDENT_API_BASE}', 'student_api_key_env':'SERVING_API_KEY',
       'student_concurrency':256}}]}
   path=source/'eduardo_interaction_config.yaml'; path.write_text(yaml.safe_dump(original))
   original_bytes=path.read_bytes(); work=root/'work'; work.mkdir()
   with patch.object(b,'REPO',root), patch.dict(b.os.environ,{},clear=True):
    copied=b.make_interaction(work,'Llama-3.1-8B-Instruct-root')
    cfg=yaml.safe_load(copied.read_text())['interaction'][0]['config']
    self.assertEqual(cfg['student_model'],'Llama-3.1-8B-Instruct-root')
    self.assertEqual(cfg['student_api_base'],'http://127.0.0.1:18861/v1')
    self.assertEqual(cfg['student_api_key_env'],'SERVING_API_KEY')
    self.assertEqual(cfg['student_concurrency'],16)
    self.assertNotIn('${oc.env:',copied.read_text())
    self.assertEqual(b.Callbacks(work,'student','local-token',root/'r',copied,lambda:{}).base['USER'],'root')
    self.assertEqual(path.read_bytes(),original_bytes)
 def test_interaction_preflight_isolated_from_secrets(self):
  import yaml
  config={'custom_reward_function':{'reward_kwargs':{'judge_concurrency':16,'student_concurrency':16}},
          'actor_rollout_ref':{'rollout':{'agent':{'num_workers':2}}},
          'reward':{'num_workers':2},'reward_model':{'num_workers':None}}
  calls=[]
  def fake_run(cmd,**kwargs):
   calls.append((cmd,kwargs))
   return SimpleNamespace(returncode=0,stdout=yaml.safe_dump(config))
  with patch.dict(b.os.environ,{'SERVING_API_KEY':'sensitive-placeholder'},clear=True),patch.object(b.subprocess,'run',fake_run):
   b.preflight_config(Path('/tmp/work'),Path('/tmp/r'),'student',Path('/tmp/interaction'))
  self.assertEqual(len(calls),2)
  worker_env=calls[1][1]['env']
  self.assertFalse(any(k in worker_env for k in ('USER','STUDENT_API_BASE','SERVING_API_KEY')))
  self.assertNotIn('sensitive-placeholder',str(calls[1][0]))
 def test_valid_symlinks_and_missing_paths(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); target=root/'target'; target.write_text('data'); target.chmod(0o700)
   link=root/'python'; link.symlink_to(target)
   b.assert_file(link,'Python解释器')
   link.unlink(); link.symlink_to(root/'missing')
   with self.assertRaisesRegex(RuntimeError,'Python解释器'):b.assert_file(link,'Python解释器')
   with self.assertRaisesRegex(RuntimeError,'训练集'):b.assert_file(root/'missing','训练集')
   target.chmod(0o600)
   with self.assertRaisesRegex(RuntimeError,'Python解释器'):b.assert_file(target,'Python解释器')
   self.assertEqual(b.safe_reason(RuntimeError('必要文件缺失：训练集')),'必要文件缺失：训练集')
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
  a='a1b2c3d4-1234-5678-9abc-1234567890ab'; b_uuid='b1b2c3d4-1234-5678-9abc-1234567890ab'; c='c1b2c3d4-1234-5678-9abc-1234567890ab'
  calls=[]
  def runner(cmd,env,**kw):
   calls.append((cmd,env))
   if cmd[-1]=='--show':out={'0':'GPU-'+b_uuid,'1':'GPU-'+a,'2':'GPU-'+c}
   else:out=[a,b_uuid]
   return SimpleNamespace(stdout=json.dumps(out))
  self.assertEqual(cuda_map.verified('/python',['GPU-'+a,'GPU-'+b_uuid],{},runner),[1,0])
  self.assertEqual(calls[-1][1]['CUDA_VISIBLE_DEVICES'],'1,0')
 def test_cuda_existing_limit_fails_closed(self):
  with self.assertRaises(RuntimeError):cuda_map.verified('/python',['GPU-A'],{'CUDA_VISIBLE_DEVICES':'0'},lambda *a,**k:None)
 def test_cuda_runtime_mismatch_fails_closed(self):
  a='a1b2c3d4-1234-5678-9abc-1234567890ab'; b_uuid='b1b2c3d4-1234-5678-9abc-1234567890ab'
  def runner(cmd,env,**kw):return SimpleNamespace(stdout=json.dumps({'0':'GPU-'+a} if cmd[-1]=='--show' else [b_uuid]))
  with self.assertRaises(RuntimeError):cuda_map.verified('/python',['GPU-'+a],{},runner)
 def test_cuda_uuid_invalid_and_mig_fail_closed(self):
  valid='a1b2c3d4-1234-5678-9abc-1234567890ab'
  self.assertEqual(cuda_map.parse_gpu_uuid('GPU-'+valid.upper()),cuda_map.parse_gpu_uuid(valid))
  for invalid in ('', 'GPU-'+valid[:-1], 'MIG-'+valid, 'GPU-A', None):
   with self.subTest(value=invalid):
    with self.assertRaises(ValueError):cuda_map.parse_gpu_uuid(invalid)
  def runner(cmd,env,**kw):return SimpleNamespace(stdout=json.dumps({'0':'GPU-'+valid} if cmd[-1]=='--show' else ['MIG-'+valid]))
  with self.assertRaises(RuntimeError):cuda_map.verified('/python',['GPU-'+valid],{},runner)
 def test_cuda_uuid_order_count_and_missing_fail_closed(self):
  a='a1b2c3d4-1234-5678-9abc-1234567890ab'; b_uuid='b1b2c3d4-1234-5678-9abc-1234567890ab'
  for selected in ([b_uuid,a],[a],[],None,[''],[a,b_uuid,a]):
   def runner(cmd,env,**kw):return SimpleNamespace(stdout=json.dumps({'0':'GPU-'+a,'1':'GPU-'+b_uuid} if cmd[-1]=='--show' else selected))
   with self.subTest(selected=selected):
    with self.assertRaises(RuntimeError):cuda_map.verified('/python',['GPU-'+a,'GPU-'+b_uuid],{},runner)
  with self.assertRaises(RuntimeError):cuda_map.verified('/python',[],{},lambda *a,**kw:None)
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
