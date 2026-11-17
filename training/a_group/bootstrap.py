#!/usr/bin/env python3
"""Prepare public A-group assets in an owned directory; never use GPUs."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
LOCK=json.loads((HERE/'assets.lock.json').read_text())
TASKS=('作者代码','作者数据','Tutor模型','门槛模型')
MIRROR='https://hf-mirror.com'
INDEX='https://mirrors.aliyun.com/pypi/simple'
GIT_PROXY='https://gh-proxy.com/'

class Failure(Exception):
    def __init__(self,category): self.category=category

def safe_env(root):
    env=os.environ.copy()
    for k in list(env):
        if k.upper() in {'HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_HUB_TOKEN',
            'HF_TOKEN_PATH','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','PIP_EXTRA_INDEX_URL',
            'UV_EXTRA_INDEX_URL','GIT_ASKPASS','SSH_ASKPASS'}: env.pop(k,None)
    env.update({'HF_ENDPOINT':MIRROR,'HF_HOME':str(root/'cache/hf'),
        'HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','HF_HUB_DISABLE_XET':'1',
        'HF_HUB_DISABLE_PROGRESS_BARS':'1','UV_CACHE_DIR':str(root/'cache/uv'),
        'XDG_CACHE_HOME':str(root/'cache/xdg'),'TMPDIR':str(root/'tmp'),'PIP_INDEX_URL':INDEX,'UV_INDEX_URL':INDEX,
        'GIT_TERMINAL_PROMPT':'0','GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null',
        'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','ROCR_VISIBLE_DEVICES':'',
        'UV_NO_PROGRESS':'1','NO_COLOR':'1','PIP_DISABLE_PIP_VERSION_CHECK':'1'})
    return env

def classify(text,default):
    text=text.lower()
    for needle,label in [('no space left','磁盘空间不足'),('timed out','网络超时'),
       ('timeout','网络超时'),('name or service not known','域名解析失败'),
       ('temporary failure in name resolution','域名解析失败'),('certificate verify failed','证书错误'),
       ('http error 401','认证失败'),('http error 403','服务拒绝'),('http error 404','资源不存在'),
       ('could not find a version','镜像缺少依赖'),('no solution found','依赖冲突'),
       ('network is unreachable','网络不可达'),('connection refused','连接被拒绝')]:
        if needle in text:return label
    return default

def execute(args,env,default,timeout=43200):
    try:
        r=subprocess.run(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                         timeout=timeout,check=False)
    except FileNotFoundError:raise Failure('所需程序缺失')
    except subprocess.TimeoutExpired:raise Failure('网络或安装超时')
    if r.returncode:
        msg=(r.stdout[-15000:]+r.stderr[-15000:]).decode('utf-8','replace')
        raise Failure(classify(msg,default))
    return r.stdout.decode('utf-8','replace').strip()

def digest(path,algo='sha256'):
    h=hashlib.new(algo)
    with path.open('rb') as f:
        if algo=='sha1':h.update(('blob %d\0'%path.stat().st_size).encode())
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def clone(spec,dest,env):
    dest.mkdir(parents=True,exist_ok=True)
    if not (dest/'.git').exists():
        if any(dest.iterdir()):raise Failure('已有目录非空')
        execute(['git','init','--quiet',str(dest)],env,'代码目录初始化失败',timeout=120)
        execute(['git','-C',str(dest),'remote','add','origin',GIT_PROXY+spec['repo']],env,'代码远端配置失败',timeout=60)
    if not (dest/'.git').is_dir():raise Failure('已有目录非有效仓库')
    remote=execute(['git','-C',str(dest),'remote','get-url','origin'],env,'代码远端异常',timeout=60)
    if remote!=GIT_PROXY+spec['repo']:raise Failure('代码远端不符')
    status=execute(['git','-C',str(dest),'status','--porcelain','--untracked-files=all'],env,'代码状态异常',timeout=60)
    if status:raise Failure('代码目录有改动')
    present=subprocess.run(['git','-C',str(dest),'cat-file','-e',spec['revision']+'^{commit}'],
                           env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60)
    if present.returncode:
        execute(['git','-C',str(dest),'fetch','--no-tags','--depth','1','origin',spec['revision']],env,'代码下载失败')
    execute(['git','-C',str(dest),'checkout','--detach',spec['revision']],env,'固定版本不存在',timeout=120)
    current=execute(['git','-C',str(dest),'rev-parse','HEAD'],env,'代码身份异常',timeout=60)
    if current!=spec['revision']:raise Failure('代码版本不符')

def snapshot(asset,dest,env,python):
    dest.mkdir(parents=True,exist_ok=True)
    missing=sum(int(x.get('size') or 0) for name,x in asset['files'].items()
                if not (dest/name).is_file() or (dest/name).stat().st_size!=(x.get('size') or 0))
    if shutil.disk_usage(dest).free < missing+5*1024**3:raise Failure('磁盘空间不足')
    code=("import json,sys; from huggingface_hub import snapshot_download; "
          "v=json.loads(sys.argv[1]); snapshot_download(repo_id=v['repo'],revision=v['revision'],"
          "repo_type=v['repo_type'],local_dir=sys.argv[2],allow_patterns=list(v['files']),"
          "token=False,max_workers=4)")
    execute([python,'-c',code,json.dumps(asset,separators=(',',':')),str(dest)],env,'镜像下载失败')
    for name,meta in asset['files'].items():
        path=dest/name
        if not path.is_file() or (meta.get('size') is not None and path.stat().st_size!=meta['size']):
            raise Failure('资源文件不完整')
        if meta.get('sha256') and digest(path)!=meta['sha256']:raise Failure('资源哈希不符')
        if meta.get('git_sha1') and digest(path,'sha1')!=meta['git_sha1']:raise Failure('资源哈希不符')


class State:
    def __init__(self,root):
        self.path=root/'status.json';self.lock=threading.Lock()
        try:self.data=json.loads(self.path.read_text())
        except (OSError,ValueError):self.data={'tasks':{}}
        for name in TASKS:
            if self.data['tasks'].get(name,{}).get('status')!='完成':
                self.data['tasks'][name]={'status':'等待','detail':'待执行'}
        self.data['controller']={'status':'运行中','pid':os.getpid()};self.save()
    def save(self):
        tmp=self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.data,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,self.path)
    def set(self,name,status,detail):
        with self.lock:self.data['tasks'][name]={'status':status,'detail':detail};self.save()
    def finish(self):
        with self.lock:self.data['controller']={'status':'已结束','pid':os.getpid()};self.save()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);ap.add_argument('--lock-fd',required=True,type=int);args=ap.parse_args()
    root=Path(args.run_dir)
    if not root.is_absolute() or root.is_symlink() or not (root/'A_GROUP_OWNED.json').is_file():return 2
    fd=args.lock_fd
    try:
        held=os.fstat(fd);expected=(root/'.bootstrap.lock').stat()
        if (held.st_dev,held.st_ino)!=(expected.st_dev,expected.st_ino):return 2
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return 3
        env=safe_env(root);state=State(root)
        for name in ('cache','source','models','data','tmp'):(root/name).mkdir(exist_ok=True)
        def task(name,func):
            if state.data['tasks'].get(name,{}).get('status')=='完成':return
            state.set(name,'进行中','准备或校验中')
            try: detail=func() or '版本和哈希已核验';state.set(name,'完成',detail)
            except Failure as e:state.set(name,'失败',e.category)
            except Exception as e:state.set(name,'失败','异常类别：'+type(e).__name__)
        def code():clone(LOCK['code'],root/'source/EduardoRL',env)
        def dataset():snapshot(LOCK['assets']['dataset'],root/'data/official',env,sys.executable)
        def tutor():snapshot(LOCK['assets']['tutor'],root/'models/Qwen3.5-4B',env,sys.executable)
        def judge():snapshot(LOCK['assets']['judge'],root/'models/Qwen3.6-27B',env,sys.executable)
        # Model downloads are serial to keep storage and network pressure bounded.
        def models():task('Tutor模型',tutor);task('门槛模型',judge)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(task,'作者代码',code),pool.submit(task,'作者数据',dataset),pool.submit(models)]
            for f in futures:f.result()
        state.finish()
        return 0 if all(state.data['tasks'][x]['status']=='完成' for x in TASKS) else 1
    finally:os.close(fd)

if __name__=='__main__':sys.exit(main())
