#!/usr/bin/env python3
"""B stage, native Eduardo configuration; aggregate telemetry only.

Single controller per A lock. No author/veRL runtime patches. Requires target
preflight and user review; GPU allocation is cooperative, never exclusive.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, secrets, signal, socket, subprocess, sys, tempfile, time
from pathlib import Path
import resources as gpu_resources
import cuda_map
import gpu_process_monitor

ROOT = Path('/workspace/A_group_repro_20261117')
PYTHON = ROOT/'envs/env-candidate-vllm0191/bin/python'
REPO = ROOT/'source/EduardoRL'
DATA = ROOT/'data/author_prepared'
TUTOR = ROOT/'models/Qwen3.5-4B'
STUDENT = Path('/workspace/models/Meta-Llama-3.1-8B-Instruct')
JUDGE = ROOT/'models/Qwen3.6-27B'
COMMIT='e8c89b7293e65e632149571b16a1c2d4e33c77fe'
PORTS={'student':18861,'judge':18862}
STUDENT_API_BASE=f"http://127.0.0.1:{PORTS['student']}/v1"
WORKERS=2
STUDENT_LIMIT=16
JUDGE_LIMIT=16
FALLBACK=re.compile(rb'(?:gave up after .*?using fallback|non-retriable .*?using fallback|EduardoInteraction: student call failed hard)',re.I)
# Sources can contain user data. Never display input, exception text or raw logs.
SAFE_REASONS=frozenset(('必要文件不存在或为链接','已有CUDA可见性限制，拒绝猜测物理卡号','作者版本不一致',
 '作者受版本控制文件存在修改','训练库版本不符','模型结构不符','模型分片缺失','模型权重缺失','作者数据文件校验不一致',
 '进程清理保护不可用','interaction 配置结构异常','学生交互配置预检失败','Hydra 配置预检失败','Hydra 配置字段核对失败',
 '本次 worker 日志压缩不可检查','本任务日志不可读','学生共享卡显存保护触发或状态未知','导师卡错误','运行名称格式不符',
 'Ray socket 路径过长','旧任务守护进程尚未退出','裁判服务提前退出','学生服务提前退出','检测到 fallback，B 无效','训练进程退出失败',
 '第1步检查点缺失或输出目录已存在','作者 CPU 合并工具失败','合并输出缺少 safetensors','合并索引分片不完整','合并输出不可读取',
 '已设置 CUDA_VISIBLE_DEVICES，无法安全确认卡映射','所选卡不在继承的CUDA可见范围','CUDA 映射变化','CUDA 映射非唯一',
 '卡身份不在CUDA可见列表','CUDA 卡身份重复','选中卡的CUDA UUID复核不一致'))
FILE_LABELS=frozenset(('Python解释器','作者主配置','学生交互配置','训练集','验证集','数据清单','Tutor配置','学生配置','裁判配置'))

def safe_reason(exc):
    message=str(exc)
    return message if message in SAFE_REASONS or message in {'必要文件缺失：'+x for x in FILE_LABELS} else type(exc).__name__

def assert_file(p, label):
    # Virtualenv interpreters and model snapshots may be valid symlinks.
    # is_file() follows the link while rejecting dangling links/directories.
    if label not in FILE_LABELS: raise ValueError('文件标签不合法')
    if not p.is_file() or (label == 'Python解释器' and not os.access(p,os.X_OK)):
        raise RuntimeError('必要文件缺失：'+label)

def validate():
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        raise RuntimeError('已有CUDA可见性限制，拒绝猜测物理卡号')
    for label,p in (('Python解释器',PYTHON),('作者主配置',REPO/'eduardo/configs/eduardo_base.yaml'),
              ('学生交互配置',REPO/'eduardo/eduardo_interaction_config.yaml'),
              ('训练集',DATA/'train.parquet'),('验证集',DATA/'val.parquet'),
              ('数据清单',DATA/'materialization_manifest.json'),('Tutor配置',TUTOR/'config.json'),
              ('学生配置',STUDENT/'config.json'),('裁判配置',JUDGE/'config.json')):
        assert_file(p,label)
    result=subprocess.run(['git','-C',str(REPO),'rev-parse','HEAD'],capture_output=True,text=True,check=True,timeout=30)
    if result.stdout.strip()!=COMMIT: raise RuntimeError('作者版本不一致')
    result=subprocess.run(['git','-C',str(REPO),'status','--porcelain','--untracked-files=no'],capture_output=True,text=True,check=True,timeout=30)
    if result.stdout.strip(): raise RuntimeError('作者受版本控制文件存在修改')
    from importlib import metadata
    if not metadata.version('verl').startswith('0.7.1') or not metadata.version('vllm').startswith('0.19.1'):
        raise RuntimeError('训练库版本不符')
    def model_ok(path,shape,kind):
        config=json.loads((path/'config.json').read_text())
        sub=config.get('text_config') or config
        if (sub.get('hidden_size'),sub.get('num_hidden_layers'))!=shape or kind not in str(config.get('model_type','')):
            raise RuntimeError('模型结构不符')
        indexes=list(path.glob('*.safetensors.index.json'))
        if indexes:
            names=set(json.loads(indexes[0].read_text()).get('weight_map',{}).values())
            if not names or not all(isinstance(n,str) and Path(n).name==n and (path/n).is_file() and (path/n).stat().st_size>0 for n in names):
                raise RuntimeError('模型分片缺失')
        elif not any(x.is_file() and x.stat().st_size>0 for x in path.glob('*.safetensors')):
            raise RuntimeError('模型权重缺失')
    model_ok(TUTOR,(2560,32),'qwen'); model_ok(JUDGE,(5120,64),'qwen');model_ok(STUDENT,(4096,32),'llama')
    manifest=json.loads((DATA/'materialization_manifest.json').read_text())
    for n in ('train.parquet','val.parquet'):
        h=hashlib.sha256()
        with (DATA/n).open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        if h.hexdigest()!=manifest['output'][n]['sha256']:raise RuntimeError('作者数据文件校验不一致')
    # verify A implementation and controller lock before any GPU action
    gpu_resources.load_stage_a()
    # Fail before reserving GPUs if platform cannot perform controlled cleanup.
    # Probe support in a child, so the controller itself is not modified into
    # a process subreaper before the owned guards start.
    probe=subprocess.run([str(PYTHON),'-B','-c','import parent_guard; parent_guard.subreaper()'],
                         cwd=Path(__file__).parent,capture_output=True,timeout=30)
    if probe.returncode:raise RuntimeError('进程清理保护不可用')
    socks=[]
    try:
        for port in PORTS.values():
            s=socket.socket(); socks.append(s); s.bind(('127.0.0.1',port))
    finally:
        for s in socks:s.close()


def training_overrides(work, raytmp, student_name, interaction):
    return [
        'trainer.nnodes=1','trainer.n_gpus_per_node=2',
        'actor_rollout_ref.actor.fsdp_config.fsdp_size=2',
        'data.train_batch_size=2','actor_rollout_ref.actor.ppo_mini_batch_size=2',
        'actor_rollout_ref.rollout.n=4',
        'custom_reward_function.reward_kwargs.num_student_attempts=4',
        'actor_rollout_ref.actor.optim.lr=1e-5',
        'actor_rollout_ref.rollout.gpu_memory_utilization=0.58',
        'actor_rollout_ref.rollout.calculate_log_probs=true',
        'actor_rollout_ref.rollout.agent.num_workers=2',
        'reward.num_workers=2','reward_model.num_workers=null',
        'custom_reward_function.reward_kwargs.judge_concurrency=16',
        'custom_reward_function.reward_kwargs.student_concurrency=16',
        f'actor_rollout_ref.rollout.multi_turn.interaction_config_path={interaction}',
        f'vars.student_model_name={student_name}','vars.judge_model_name=judge-local',
        f'data.train_files={DATA}/train.parquet',f'data.val_files={DATA}/val.parquet',
        'trainer.total_training_steps=2','trainer.save_freq=1','trainer.test_freq=0',
        'trainer.resume_mode=disable','trainer.logger=[console]',
        'custom_reward_function.reward_kwargs.eval_judge_model=',
        '+ray_kwargs.ray_init.address=local',f'+ray_kwargs.ray_init._temp_dir={raytmp}',
    ]


def service_command(role,student_name):
    if role not in PORTS:raise ValueError('unknown role')
    model=STUDENT if role=='student' else JUDGE
    served=student_name if role=='student' else 'judge-local'
    mem='0.4' if role=='student' else '0.9'
    seqs='64' if role=='student' else '256'
    return [str(PYTHON),'-m','vllm.entrypoints.openai.api_server','--model',str(model),
       '--served-model-name',served,'--port',str(PORTS[role]),'--host','127.0.0.1',
       '--dtype','bfloat16','--gpu-memory-utilization',mem,'--max-model-len','32768',
       '--max-num-seqs',seqs,'--tensor-parallel-size','1','--data-parallel-size','1',
       '--enable-chunked-prefill','--enable-prefix-caching']


def make_interaction(work,student_name):
    import yaml
    source=REPO/'eduardo/eduardo_interaction_config.yaml'
    o=yaml.safe_load(source.read_text(encoding='utf-8'))
    rows=o.get('interaction')
    if not isinstance(rows,list) or len(rows)!=1 or rows[0].get('name')!='eduardo':
        raise RuntimeError('interaction 配置结构异常')
    rows[0]['config']['student_concurrency']=STUDENT_LIMIT
    # Resolve only non-secret values in this private copy: workers must use
    # the same model and loopback endpoint even if USER is not inherited.
    rows[0]['config']['student_model']=student_name
    rows[0]['config']['student_api_base']=STUDENT_API_BASE
    target=work/'interaction_b.yaml'
    with target.open('x',encoding='utf-8') as f:yaml.safe_dump(o,f,allow_unicode=True,sort_keys=False)
    return target


def preflight_config(work,raytmp,student_name,interaction):
    """Compose the original Hydra config in a CPU-only subprocess before GPUs."""
    import yaml
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUDA_DEVICE_ORDER='PCI_BUS_ID',
      PYTHONPATH=str(REPO),WORK_DIR=str(work),EDUARDO_RECIPE_DIR=str(REPO/'eduardo'),
      MODEL_PATH=str(TUTOR),EXPERIMENT='b-functional',USER=os.environ.get('USER') or 'root',
      SERVING_API_KEY='preflight-placeholder',STUDENT_API_BASE=STUDENT_API_BASE,
      JUDGE_API_BASE='http://127.0.0.1:18862/v1',WANDB_MODE='disabled',
      HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    cmd=[str(PYTHON),str(Path(__file__).with_name('trainer_entry.py')),'--config-path',str(REPO/'eduardo/configs'),
         '--config-name','eduardo_base','--cfg','job','--resolve',*training_overrides(work,raytmp,student_name,interaction)]
    proc=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=180)
    if proc.returncode:raise RuntimeError('Hydra 配置预检失败')
    try:
        c=yaml.safe_load(proc.stdout)
        r=c['custom_reward_function']['reward_kwargs']; agent=c['actor_rollout_ref']['rollout']['agent']['num_workers']
        if (agent,c['reward']['num_workers'],c['reward_model']['num_workers'],r['judge_concurrency'],r['student_concurrency'])!=(2,2,None,16,16):
            raise ValueError('worker config mismatch')
    except (TypeError,KeyError,ValueError) as exc:raise RuntimeError('Hydra 配置字段核对失败') from exc
    # Hydra --cfg does not load this separate file. Resolve it explicitly in
    # a worker-like environment with the original two variables absent.
    worker_env=dict(env)
    for key in ('USER','STUDENT_API_BASE','SERVING_API_KEY'):
        worker_env.pop(key,None)
    check=('import sys; from omegaconf import OmegaConf; '
           'c=OmegaConf.to_container(OmegaConf.load(sys.argv[1]).interaction[0].config,resolve=True); '
           'actual=(c["student_model"], c["student_api_base"], c["student_api_key_env"], c["student_concurrency"]); '
           'sys.exit(actual != (sys.argv[2], sys.argv[3], "SERVING_API_KEY", int(sys.argv[4])))')
    worker_check=subprocess.run([str(PYTHON),'-B','-c',check,str(interaction),student_name,STUDENT_API_BASE,str(STUDENT_LIMIT)],
                                env=worker_env,capture_output=True,timeout=60)
    if worker_check.returncode:raise RuntimeError('学生交互配置预检失败')


def atomic(work,state):
    fd,p=tempfile.mkstemp(prefix='.controller-',dir=work)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(state,f,ensure_ascii=False);f.write('\n')
        os.replace(p,work/'orchestration.json')
    finally:
        if os.path.exists(p):os.unlink(p)


class LogScanner:
    """Incremental scan of this run's trainer/Ray worker logs only."""
    def __init__(self,work,raytmp):self.work=Path(work);self.raytmp=Path(raytmp);self.offsets={};self.tails={}
    def files(self):
        yield self.work/'trainer.log'
        logs=self.raytmp/'session_latest'/'logs'
        if logs.is_dir():
            for pat in ('worker-*.err*','worker-*.out*'):
                for path in logs.glob(pat):
                    if path.name.endswith(('.gz','.bz2','.xz','.zst','.zip')):
                        raise RuntimeError('本次 worker 日志压缩不可检查')
                    yield path
    def scan(self):
        for path in self.files():
            try:
                stat=path.stat()
                if not path.is_file():continue
                key=(stat.st_dev,stat.st_ino); off=self.offsets.get(key,0)
                if stat.st_size<off:off=0;self.tails.pop(key,None)
                with path.open('rb') as f:f.seek(off);chunk=f.read();self.offsets[key]=f.tell()
                if not chunk:continue
                data=self.tails.get(key,b'')+chunk
                if FALLBACK.search(data):return True
                self.tails[key]=data[-1024:]
            except FileNotFoundError:continue
            except OSError as e:raise RuntimeError('本任务日志不可读') from e
        return False


class Callbacks:
    def __init__(self,work,student_name,token,raytmp,interaction,snapshot):
        self.work,self.student_name,self.token,self.raytmp,self.interaction=work,student_name,token,raytmp,interaction
        self.snapshot=snapshot;self.children=[];self.process_by_role={};self.bindings={};self.protected_gpu=None
        self.base={k:v for k,v in os.environ.items() if not (k.upper().endswith(('_TOKEN','_API_KEY','_SECRET')) or k.upper() in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'))}
        self.base.update({'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','CUDA_DEVICE_ORDER':'PCI_BUS_ID',
           'USER':os.environ.get('USER') or 'root',
           'VLLM_API_KEY':token,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
           'HF_HUB_DISABLE_TELEMETRY':'1','VLLM_NO_USAGE_STATS':'1','WANDB_MODE':'disabled','RAY_DEDUP_LOGS':'0',
           'HF_HOME':str(work/'cache/hf'),'TMPDIR':str(work/'tmp'),'XDG_CACHE_HOME':str(work/'cache')})
    def spawn(self,name,args,env):
        log=(self.work/(name+'.log')).open('xb')
        try:p=subprocess.Popen([str(PYTHON),str(Path(__file__).with_name('parent_guard.py')),'--parent',str(os.getpid()),'--',*args],
            env=env,cwd=self.work,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        finally:log.close()
        self.children.append(p);self.process_by_role[name]=p;return p
    def launch_service(self,role,binding):
        self.bindings[role]=[binding.index]
        ordinal=cuda_map.verified(PYTHON,[binding.uuid],self.base)[0]
        return self.spawn(role,service_command(role,self.student_name),dict(self.base,CUDA_VISIBLE_DEVICES=str(ordinal)))
    def student_guard(self):
        cards=self.bindings.get('student')
        if cards is None:return
        row=self.snapshot().get(cards[0]) if len(cards)==1 else None
        if not row or not isinstance(row.get('free'),int) or row['free']<8192:
            self.protected_gpu=cards[0] if len(cards)==1 else None
            raise RuntimeError('学生共享卡显存保护触发或状态未知')
    def service_ready(self,role,binding,proc):
        self.student_guard()
        try:
            import urllib.request
            with urllib.request.urlopen('http://127.0.0.1:%d/health'%PORTS[role],timeout=3) as res:return proc.poll() is None and res.status==200
        except OSError:return False
    def launch_tutor(self,tutors):
        self.student_guard()
        if len(tutors)!=2 or tutors[0].index==tutors[1].index:raise RuntimeError('导师卡错误')
        self.bindings['tutor']=[x.index for x in tutors]
        ordinals=cuda_map.verified(PYTHON,[x.uuid for x in tutors],self.base)
        env=dict(self.base,CUDA_VISIBLE_DEVICES=','.join(map(str,ordinals)),
           PYTHONPATH=str(REPO),WORK_DIR=str(self.work),EDUARDO_RECIPE_DIR=str(REPO/'eduardo'),
           MODEL_PATH=str(TUTOR),EXPERIMENT='b-functional',SERVING_API_KEY=self.token,
           STUDENT_API_BASE=STUDENT_API_BASE,JUDGE_API_BASE='http://127.0.0.1:18862/v1',
           RAY_ADDRESS='local',RAY_TMPDIR=str(self.raytmp))
        cmd=[str(PYTHON),str(Path(__file__).with_name('trainer_entry.py')),'--config-path',str(REPO/'eduardo/configs'),'--config-name','eduardo_base']
        return self.spawn('trainer',cmd+training_overrides(self.work,self.raytmp,self.student_name,self.interaction),env)
    def exit_codes(self):
        return {name: proc.poll() for name, proc in self.process_by_role.items()}
    def stop_started(self):
        for p in reversed(self.children):
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        until=time.monotonic()+30
        for p in reversed(self.children):
            try:p.wait(timeout=max(0,until-time.monotonic()))
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                p.wait(timeout=10)


def wait_for_training(callbacks, scanner):
    """Check the two serving processes separately from normal trainer completion."""
    while True:
        codes = callbacks.exit_codes()
        if scanner.scan():
            raise RuntimeError('检测到 fallback，B 无效')
        if codes['trainer'] not in (None, 0):
            raise RuntimeError('训练进程退出失败')
        if codes['judge'] is not None:
            raise RuntimeError('裁判服务提前退出')
        if codes['student'] is not None:
            raise RuntimeError('学生服务提前退出')
        if codes['trainer'] == 0:
            return
        callbacks.student_guard()
        time.sleep(10)


def run(name):
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,39}',name):raise ValueError('运行名称格式不符')
    validate()
    work=ROOT/'pilots'/name
    work.mkdir(mode=0o700,parents=True,exist_ok=False)
    (work/'tmp').mkdir();(work/'cache').mkdir()
    raytmp=ROOT/('r'+secrets.token_hex(3))
    raytmp.mkdir(mode=0o700,exist_ok=False)
    if len((str(raytmp)+'/session_2099-12-31_23-59-59_999999_4194304/sockets/plasma_store').encode())>107:raise RuntimeError('Ray socket 路径过长')
    (work/'ray_tmp_path.txt').write_text(str(raytmp)+'\n')
    student_name='Llama-3.1-8B-Instruct-'+(os.environ.get('USER') or 'root')
    interaction=make_interaction(work,student_name)
    (work/'deviations.md').write_text('''# B 阶段偏离\n\n- 功能测试：2 步，每步 2 题，每题 4 对话，后测 4 次；mini-batch 2。\n- 学习率 1e-5；末次验证禁用；每步保存检查点。仅 B 生效。\n- 固定 agent worker=2、reward worker=2；judge 每进程 16，student 每组每进程 16；student max-num-seqs 64。实际 PID/并发需从日志核对。\n- 单机 A100；裁判 1 卡、学生 1 卡、Tutor 2 卡。导师 gpu_memory_utilization=0.58。\n- 代码 e8c89b7，veRL 0.7.1，候选 vLLM 0.19.1 与其声明依赖范围不一致。\n- 外部每 10 秒扫描技术 fallback 并停止；不修改原作者重试及 reward。\n- 论文教师生成设置不变；loss 函数与 judge REJECT 规则不变。\n- logprob/probability 差异只能辅助诊断，不证明权重完全同步。\n''',encoding='utf-8')
    with (work/'deviations.md').open('a',encoding='utf-8') as f:
        f.write('- B 的交互配置副本固定已选学生模型名称和本机地址，避免 Ray worker 缺少 USER；模型与地址不变，未写入密钥。\n')
    token=secrets.token_urlsafe(32)
    resources=gpu_resources.GPUResources(work)
    callbacks=Callbacks(work,student_name,token,raytmp,interaction,resources.snapshot)
    state={'phase':'准备','reason':'','protected_gpu':None,'events':[],'bindings':{},'role_processes':{}}
    def update(phase,reason=''):
        codes = callbacks.exit_codes()
        state.update(phase=phase,reason=reason,bindings=callbacks.bindings,
                     role_exit_codes=codes,role_processes={role: rc is None for role, rc in codes.items()},
                     protected_gpu=callbacks.protected_gpu,updated_at=time.time())
        state['events'].append({'phase':phase,'time':time.time()});atomic(work,state)
    lock=False; sampler=None
    def interrupt(sig,_):raise KeyboardInterrupt
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupt)
    try:
        preflight_config(work,raytmp,student_name,interaction)
        resources.acquire_lock();lock=True
        # Lock can be released milliseconds before an old guard finishes.
        # Refuse to proceed while any known A-group guard or endpoint remains;
        # never signal a process discovered here.
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try: args=(proc/'cmdline').read_bytes().split(b'\0')
            except OSError: continue
            if any(b'/A_group_repro_20261117/' in a and any(a.endswith(x) for x in (b'/parent_guard.py',b'/pilot.py',b'/reserve_worker.py')) for a in args):
                raise RuntimeError('旧任务守护进程尚未退出')
        sockets=[]
        try:
            for port in PORTS.values():
                s=socket.socket();sockets.append(s);s.bind(('127.0.0.1',port))
        finally:
            for s in sockets:s.close()
        def task_roots():
            found={}
            for role,h in list(resources.holders.items()):
                if h.alive():found.setdefault(role,[]).append(gpu_process_monitor.root_identity(h.process.pid))
            for role,p in list(callbacks.process_by_role.items()):
                if p.poll() is None:found.setdefault(role,[]).append(gpu_process_monitor.root_identity(p.pid))
            return found
        def task_cards():
            ids={h.binding.index for h in list(resources.holders.values())}
            ids.update(i for values in list(callbacks.bindings.values()) for i in values)
            rows=resources.snapshot()
            result={}
            for i in ids:
                row=rows.get(i)
                if row is None:raise RuntimeError('显卡状态不可读')
                result[i]={'apps':row['apps'],'used_mib':row['total']-row['free']}
            return result
        sampler=gpu_process_monitor.Sampler(work/'gpu_samples.jsonl',task_roots,task_cards,interval=10)
        sampler.start();update('凑卡')
        resources.acquire_and_handoff(callbacks)
        update('训练中')
        wait_for_training(callbacks, LogScanner(work,raytmp))
        # Release GPU services before the CPU-only author checkpoint merge.
        callbacks.stop_started()
        update('CPU合并','执行作者 FSDP 合并工具')
        actor=work/'verl_ckpts/b-functional/global_step_1/actor'
        merged=work/'eval_checkpoints/b-functional/step_1'
        if not actor.is_dir() or not any(actor.iterdir()) or merged.exists():
            raise RuntimeError('第1步检查点缺失或输出目录已存在')
        env=dict(callbacks.base,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(REPO))
        merge=callbacks.spawn('cpu_merge',[str(PYTHON),'-m','verl.model_merger','merge','--backend','fsdp',
                   '--local_dir',str(actor),'--target_dir',str(merged)],env)
        if merge.wait()!=0:raise RuntimeError('作者 CPU 合并工具失败')
        shards=list(merged.glob('*.safetensors')) if merged.is_dir() else []
        if not shards:raise RuntimeError('合并输出缺少 safetensors')
        from safetensors import safe_open
        keys=0; readable=False
        for shard in shards:
            with safe_open(str(shard),framework='pt',device='cpu') as f:
                names=list(f.keys());keys+=len(names)
                for name in names:
                    shape=f.get_slice(name).get_shape();size=1
                    for n in shape:size*=n
                    if 0<size<=1000000:
                        t=f.get_tensor(name)
                        if t.numel()==size:readable=True;break
        indexes=list(merged.glob('*.safetensors.index.json'))
        if indexes:
            vals=set(json.loads(indexes[0].read_text()).get('weight_map',{}).values())
            if not vals or any(not (merged/n).is_file() for n in vals):raise RuntimeError('合并索引分片不完整')
        if not keys or not readable or not (merged/'config.json').is_file():raise RuntimeError('合并输出不可读取')
        update('待核验','训练退出且CPU合并可读；仍需核查步骤、指标与失败率')
    except BaseException as e:
        update('失败',safe_reason(e))
        raise
    finally:
        try:
            if sampler is not None:
                sampler.stop()
                (work/'gpu_peaks.json').write_text(json.dumps(sampler.report(),ensure_ascii=False)+'\n')
        finally:
            try:callbacks.stop_started()
            finally:
                if lock:resources.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--name',default='b_native_v1');a=p.parse_args()
    if not a.run:p.error('只允许明确 --run')
    try:run(a.name)
    except BaseException as e:
        print('B 未完成：'+safe_reason(e),file=sys.stderr);sys.exit(1)
