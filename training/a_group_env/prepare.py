#!/usr/bin/env python3
"""Prepare one isolated, unvalidated A100 training environment without using GPUs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

ALIBABA = 'https://mirrors.aliyun.com/pypi/simple'
VERL_REMOTE = 'https://gh-proxy.com/https://github.com/volcengine/verl.git'
VERL_COMMIT = 'bec9ef74768dd201881cd4e54cd0385e87caae27'
ENV_NAME = 'env-candidate-vllm0191'
FLASH_URL = ('https://mirrors.aliyun.com/pypi/packages/3b/b2/'
             '8d76c41ad7974ee264754709c22963447f7f8134613fd9ce80984ed0dab7/'
             'flash_attn-2.8.3.tar.gz')
FLASH_SHA256 = '1e71dd64a9e0280e0447b8a0c2541bad4bf6ac65bdeaa2f90e51a9e57de0370d'
VERSIONS = {'torch':'2.10.0','vllm':'0.19.1','transformers':'5.7.0',
            'numpy':'2.1.0','ray':'2.55.1','verl':'0.7.1','math-verify':'0.8.0'}
PACKAGES = [
 'torch==2.10.0','vllm==0.19.1','transformers==5.7.0','numpy==2.1.0',
 'ray[default]==2.55.1','math-verify==0.8.0','datasets==4.8.5',
 'accelerate==1.13.0','codetiming==1.4.0','dill==0.4.0','hydra-core==1.3.2',
 'pandas==2.3.3','peft==0.19.1','pyarrow==22.0.0','pybind11==3.0.1',
 'pylatexenc==2.10','tensordict==0.10.0','torchdata==0.11.0','wandb==0.26.1',
 'packaging==25.0','tensorboard==2.20.0','setuptools==80.9.0','wheel',
 'openai==2.33.0','ninja==1.13.0'
]
KNOWN_NUMPY = re.compile(r'^verl\s+0\.7\.1\s+has requirement\s+numpy\s*<\s*2(?:\.0(?:\.0)?)?,\s+but you have numpy\s+2\.1\.0\.?$',re.I)

class Failure(Exception):
    def __init__(self, label): self.label=label


def safe_env(root):
    env=os.environ.copy()
    forbidden={'HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_HUB_TOKEN','HF_TOKEN_PATH',
      'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','PIP_EXTRA_INDEX_URL','UV_EXTRA_INDEX_URL',
      'PIP_FIND_LINKS','UV_FIND_LINKS','UV_INDEX','UV_CONFIG_FILE',
      'GIT_ASKPASS','SSH_ASKPASS','ROCR_VISIBLE_DEVICES','CONDA_PREFIX','VIRTUAL_ENV',
      'PYTHONPATH','PYTHONHOME','LD_PRELOAD'}
    for key in list(env):
        if key.upper() in forbidden:env.pop(key,None)
    env.update({'CUDA_VISIBLE_DEVICES':'','HF_HOME':str(root/'cache/hf-env'),
      'HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','UV_CACHE_DIR':str(root/'cache/uv-env'),
      'XDG_CACHE_HOME':str(root/'cache/xdg-env'),'TMPDIR':str(root/'tmp'),
      'PIP_INDEX_URL':ALIBABA,'UV_INDEX_URL':ALIBABA,'UV_DEFAULT_INDEX':ALIBABA,
      'UV_NO_CONFIG':'1','PIP_CONFIG_FILE':os.devnull,
      'UV_PYTHON_DOWNLOADS':'never','UV_NO_PROGRESS':'1','PYTHONDONTWRITEBYTECODE':'1',
      'PIP_DISABLE_PIP_VERSION_CHECK':'1','GIT_TERMINAL_PROMPT':'0',
      'GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':os.devnull,'NO_COLOR':'1',
      'A_GROUP_FAILURE_LOG':str(root/'env-candidate-errors.log')})
    return env


def classify(raw,default):
    raw=raw.lower()
    for needle,name in [('no space left','磁盘空间不足'),('timed out','网络超时'),
      ('timeout','网络超时'),('name or service not known','域名解析失败'),
      ('temporary failure in name resolution','域名解析失败'),('certificate verify failed','证书错误'),
      ('http error 403','镜像拒绝请求'),('http error 404','镜像资源不存在'),
      ('no matching distribution','镜像缺少依赖'),('could not find a version','镜像缺少依赖'),
      ('no solution found','依赖冲突'),('connection refused','连接被拒绝')]:
        if needle in raw:return name
    return default


def run(args,env,label,timeout=43200):
    try:
        result=subprocess.run(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                              timeout=timeout,check=False)
    except FileNotFoundError:raise Failure('所需程序缺失')
    except subprocess.TimeoutExpired:raise Failure('操作超时')
    if result.returncode:
        text=(result.stdout[-16000:]+result.stderr[-16000:]).decode('utf-8','replace')
        logfile=env.get('A_GROUP_FAILURE_LOG')
        if logfile:
            with open(logfile,'a',encoding='utf-8') as out:
                out.write('\n--- '+label+' ---\n'+text+'\n')
        raise Failure(classify(text,label))
    return result.stdout.decode('utf-8','replace').strip()


def clone_verl(src,env):
    if src.is_symlink():raise Failure('源码目录不能是链接')
    src.mkdir(parents=True,exist_ok=True)
    if not (src/'.git').exists():
        if any(src.iterdir()):raise Failure('veRL源码目录非空')
        run(['git','init','--quiet',str(src)],env,'初始化源码失败',timeout=120)
        run(['git','-C',str(src),'remote','add','origin',VERL_REMOTE],env,'源码远端配置失败',timeout=60)
    if not (src/'.git').is_dir():raise Failure('veRL源码目录异常')
    if run(['git','-C',str(src),'remote','get-url','origin'],env,'源码远端异常',timeout=60)!=VERL_REMOTE:
        raise Failure('源码远端不符')
    if run(['git','-C',str(src),'status','--porcelain','--untracked-files=all'],env,'源码状态异常',timeout=60):
        raise Failure('veRL源码目录存在改动')
    try:
        present=subprocess.run(['git','-C',str(src),'cat-file','-e',VERL_COMMIT+'^{commit}'],
             env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60)
    except (OSError,subprocess.TimeoutExpired):raise Failure('源码版本检查失败')
    if present.returncode:
        run(['git','-C',str(src),'fetch','--no-tags','--depth','1','origin',VERL_COMMIT],env,'源码下载失败')
    run(['git','-C',str(src),'checkout','--detach',VERL_COMMIT],env,'源码版本检出失败',timeout=120)
    if run(['git','-C',str(src),'rev-parse','HEAD'],env,'源码版本检查失败',timeout=60)!=VERL_COMMIT:
        raise Failure('源码版本不符')


def check_pip_output(returncode,stdout):
    if returncode==0:return '无依赖冲突（仍需真实训练验证）'
    lines=[line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines)==1 and KNOWN_NUMPY.fullmatch(lines[0]):
        return '仅有已知 veRL/numpy 版本声明冲突'
    raise Failure('存在未解释的依赖冲突')


def get_flash_source(root):
    """Download only the fixed public sdist; do not replace an existing file."""
    dest=root/'cache'/'flash_attn-2.8.3.tar.gz'
    if dest.is_symlink():raise Failure('Flash Attention 源码路径不能是链接')
    if dest.exists():
        if not dest.is_file() or hashlib.sha256(dest.read_bytes()).hexdigest()!=FLASH_SHA256:
            raise Failure('已有 Flash Attention 源码校验失败，保留供检查')
        return dest
    fd,tmp=tempfile.mkstemp(prefix='.flash-source-',suffix='.part',dir=str(root/'cache'))
    try:
        digest=hashlib.sha256()
        # No inherited proxy or mirror credentials; HTTPS certificate verification remains enabled.
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with os.fdopen(fd,'wb') as out, opener.open(FLASH_URL,timeout=120) as src:
            while True:
                block=src.read(1024*1024)
                if not block:break
                out.write(block);digest.update(block)
            out.flush();os.fsync(out.fileno())
        if digest.hexdigest()!=FLASH_SHA256:raise Failure('Flash Attention 源码校验失败')
        # Link fails instead of replacing if another process creates dest.
        os.link(tmp,dest)
        return dest
    except Failure:raise
    except Exception:raise Failure('Flash Attention 阿里源下载失败')
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def ensure_flash(root,env,py,uv):
    flash_code='import flash_attn; print(flash_attn.__version__)'
    probe=subprocess.run([str(py),'-c',flash_code],env=env,stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL,timeout=120)
    if probe.returncode==0:return
    write_status(root,'进行中','Flash Attention 下载','下载并校验阿里源固定源码')
    src=get_flash_source(root)
    build_env=env.copy()
    build_env.update({'FLASH_ATTENTION_FORCE_BUILD':'TRUE','FLASH_ATTN_CUDA_ARCHS':'80',
                      'MAX_JOBS':'4','NVCC_THREADS':'2','CUDA_HOME':'/usr/local/cuda',
                      'PATH':str(py.parent)+os.pathsep+env.get('PATH','')})
    write_status(root,'进行中','Flash Attention 编译','使用 CUDA 12.8 为 A100 编译；可能需要较长时间')
    command=[uv,'pip','install','--python',str(py),'--no-build-isolation','--no-deps',
             '--reinstall-package','flash-attn',str(src)]
    try:
        with (root/'flash-build.log').open('ab') as log:
            result=subprocess.run(command,env=build_env,stdout=log,stderr=subprocess.STDOUT,
                                  timeout=43200,check=False)
        if result.returncode:raise Failure('Flash Attention 编译失败，日志保存在任务目录')
    except subprocess.TimeoutExpired:raise Failure('Flash Attention 编译超时')
    except FileNotFoundError:raise Failure('Flash Attention 编译工具缺失')
    run([str(py),'-c',flash_code],env,'Flash Attention 导入失败',timeout=120)


def write_status(root,status,stage,detail):
    content={'status':status,'stage':stage,'detail':detail,'updated':int(time.time()),
             'note':'候选环境：尚未进行 GPU 加载或训练验证'}
    fd,path=tempfile.mkstemp(prefix='.env-candidate-',suffix='.tmp',dir=str(root))
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(content,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(path,root/'env-candidate-status.json')
    finally:
        if os.path.exists(path):os.unlink(path)


def validate_root(root):
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():raise Failure('工作目录无效')
    marker=root/'A_GROUP_OWNED.json'
    if marker.is_symlink() or not marker.is_file():raise Failure('缺少任务目录标记')
    try:value=json.loads(marker.read_text())
    except (OSError,ValueError):raise Failure('任务目录标记无效')
    if value.get('schema')!='eduardo-a-group-prep-v1':raise Failure('任务目录标记不符')


def setup(root,env):
    uv=shutil.which('uv')
    if not uv:raise Failure('uv 未安装')
    if sys.version_info < (3,10):raise Failure('Python 版本低于 3.10')
    if shutil.disk_usage(root).free < 20*1024**3:raise Failure('磁盘空间不足')
    for part in ('tmp','cache','source','envs'):
        d=root/part
        if d.is_symlink():raise Failure('工作子目录不能是链接')
        d.mkdir(exist_ok=True)
    deviation=root/'env-candidate-deviations.md'
    report='''# A 组候选环境偏离（待运行验证）\n\n作者镜像 NGC 26.03.post1 包含 NVIDIA 定制 vLLM 0.17.1+bd67d66a、CUDA 13.2 和 flash-attn 2.7.4.post1。为适配本服务器 CUDA 12.8，候选环境使用公共 vLLM 0.19.1、CUDA 12.8 及从源码构建的 flash-attn 2.8.3。\n\nveRL 0.7.1 声明 numpy <2、vLLM 额外依赖最高 0.12；作者镜像本身亦使用 numpy 2.1 和定制 vLLM，不能仅凭版本声明推定作者环境无冲突。候选环境也使用 numpy 2.1，并以 --no-deps 安装作者固定 veRL 提交。\n\n本文件不代表兼容性已验证：需要 A100 上完成学生与导师模型加载、前向反向、DPPO 更新、rollout 权重同步和检查点读写。\n'''
    if deviation.is_symlink():raise Failure('偏离记录不能是链接')
    if not deviation.exists():
        with deviation.open('x',encoding='utf-8') as f:f.write(report)
    elif deviation.read_text(encoding='utf-8')!=report:
        raise Failure('已有偏离记录不同，保留供检查')
    src=root/'source/verl-v0.7.1'
    write_status(root,'进行中','源码','获取并校验作者版本 veRL 源码')
    clone_verl(src,env)
    envdir=root/'envs'/ENV_NAME
    owner=root/'envs'/(ENV_NAME+'.owner.json')
    if envdir.is_symlink() or owner.is_symlink():raise Failure('环境路径不能是链接')
    if owner.exists():
        try: item=json.loads(owner.read_text())
        except (OSError,ValueError):raise Failure('环境标记无效')
        if item != {'schema':'a-group-candidate-v1','versions':VERSIONS}:
            raise Failure('环境标记不符')
    else:
        if envdir.exists():raise Failure('已有环境，禁止覆盖')
        with owner.open('x') as f:json.dump({'schema':'a-group-candidate-v1','versions':VERSIONS},f,sort_keys=True)
    py=envdir/'bin/python'
    write_status(root,'进行中','虚拟环境','创建或复用本任务拥有的独立环境')
    if not envdir.exists():
        run([uv,'venv','--seed','--python',sys.executable,str(envdir)],env,'创建虚拟环境失败',timeout=1200)
    if not py.is_file():raise Failure('虚拟环境创建不完整，请使用新目录')
    write_status(root,'进行中','基础依赖','从阿里源安装固定候选版本')
    run([uv,'pip','install','--python',str(py),'--index-url',ALIBABA,*PACKAGES],env,'候选依赖安装失败')
    write_status(root,'进行中','veRL','安装固定源码，不自动替换依赖')
    # Build outside the pinned checkout: setuptools may generate egg-info files.
    with tempfile.TemporaryDirectory(prefix='verl-build-',dir=str(root/'tmp')) as temp:
        build=Path(temp)/'verl'
        shutil.copytree(src,build,ignore=shutil.ignore_patterns('.git'),symlinks=True)
        run([uv,'pip','install','--python',str(py),'--index-url',ALIBABA,
             '--no-build-isolation','--no-deps',str(build)],env,'veRL 安装失败')
    write_status(root,'进行中','依赖检查','核对版本声明及允许的已知冲突')
    try:check=subprocess.run([str(py),'-m','pip','check'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
    except (OSError,subprocess.TimeoutExpired):raise Failure('依赖检查失败')
    explanation=check_pip_output(check.returncode,check.stdout.decode('utf-8','replace'))
    write_status(root,'进行中','只读导入检查',explanation)
    check_code='''import importlib.metadata as m\nversions=%r\nfor name,want in versions.items():\n got=m.version(name)\n if name=="torch": got=got.split("+",1)[0]\n assert got==want,(name,got,want)\nfrom transformers import Qwen3_5Config\nfrom verl.trainer.ppo.core_algos import compute_policy_loss_dppo_tv\nfrom vllm import ModelRegistry\nassert "Qwen3_5ForConditionalGeneration" in ModelRegistry.get_supported_archs()\nprint("OK")\n''' % VERSIONS
    run([str(py),'-c',check_code],env,'依赖导入或架构识别失败',timeout=300)
    write_status(root,'进行中','Flash Attention 检查','检查训练默认使用的独立 flash_attn')
    ensure_flash(root,env,py,uv)
    return explanation


def show(root):
    path=root/'env-candidate-status.json'
    try:st=json.loads(path.read_text())
    except (OSError,ValueError):st={'status':'尚未运行','stage':'—','detail':'等待启动'}
    if st.get('status') == '进行中':
        lock=root/'.env-candidate.lock'
        active=False
        try:
            fd=os.open(str(lock),os.O_RDWR|getattr(os,'O_NOFOLLOW',0))
            try:
                try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:active=True
            finally:os.close(fd)
        except OSError:pass
        if not active:st.update({'status':'中断','detail':'进程已退出；保留已有文件，可重试'})
    print('候选训练环境（不使用GPU）')
    for key,label in (('status','状态'),('stage','阶段'),('detail','说明')):
        print(f'{label}：{st.get(key,"未知")}')
    print('说明：veRL 声明 vLLM 上限 0.12；本候选为 0.19.1，须核验兼容性。')
    print('导入通过也不代表 A100 训练、模型反向或权重同步已验证。')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True)
    parser.add_argument('--status',action='store_true');args=parser.parse_args()
    root=Path(args.run_dir);validate_root(root)
    if args.status:show(root);return 0
    os.umask(0o077)
    fd=os.open(str(root/'.env-candidate.lock'),os.O_CREAT|os.O_RDWR|getattr(os,'O_NOFOLLOW',0),0o600)
    try:
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('环境准备已在运行，请勿重复启动。');return 0
        env=safe_env(root)
        try:
            detail=setup(root,env)
            write_status(root,'完成','候选安装',detail+'；尚未 GPU 验证')
            print('候选环境准备完成；尚未运行 GPU 或训练。')
            return 0
        except Failure as e:
            try:stage=json.loads((root/'env-candidate-status.json').read_text()).get('stage','准备中断')
            except (OSError,ValueError):stage='准备中断'
            write_status(root,'失败',stage,e.label+'；详见任务目录中的日志')
            print('候选环境准备失败：'+e.label)
            return 1
        except Exception as e:
            label='异常类别：'+type(e).__name__
            write_status(root,'失败','准备中断',label)
            print('候选环境准备失败：'+label)
            return 1
    finally:os.close(fd)

if __name__=='__main__':
    try:sys.exit(main())
    except Failure as exc:print('未启动：'+exc.label);sys.exit(1)
