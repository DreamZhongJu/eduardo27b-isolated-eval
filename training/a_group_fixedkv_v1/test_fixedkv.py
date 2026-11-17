"""CPU-only integration check: verify launch wiring without real GPU or training."""
import importlib.util
import tempfile
from pathlib import Path

p=Path(__file__).with_name('pilot.py')
spec=importlib.util.spec_from_file_location('fixedkv_pilot', p)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
commands=[]
class Service:
    def poll(self): return None
class Trainer:
    returncode=0
    def poll(self): return 0

def fake_popen(cmd, **kwargs):
    commands.append((cmd, kwargs))
    return Trainer()

with tempfile.TemporaryDirectory(prefix='a4b-', dir='/tmp') as tmp:
    m.ROOT=Path(tmp)
    (m.ROOT/'pilots').mkdir()
    m.DATA=m.ROOT/'data'
    m.REPO=m.ROOT/'source'
    m.TUTOR=m.ROOT/'tutor'
    m.PYTHON=m.ROOT/'python'
    m.TRAINER_ENTRY=m.ROOT/'trainer_entry.py'
    m.validate=lambda:None
    m.check_ports=lambda:None
    m.check_cards=lambda _:None
    m.service=lambda *args: Service()
    m.ready=lambda *args:None
    m.smoke=lambda *args:None
    m.stop_children=lambda:None
    m.subprocess.Popen=fake_popen
    work=m.ROOT/'pilots'/'test1'
    m.run([0,1,2,3], 1, work)
    assert len(commands)==1
    args=commands[0][0]
    assert '+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=4294967296' in args
    assert 'actor_rollout_ref.rollout.n=16' in args
    assert 'actor_rollout_ref.rollout.max_num_seqs=64' in args
    assert 'custom_reward_function.reward_kwargs.num_student_attempts=16' in args
    assert 'trainer.total_training_steps=1' in args
    assert (work/'deviations.md').is_file()
    print('固定 KV 参数、16 次采样及独立记录检查通过（CPU 模拟）')
