#!/usr/bin/env python3
"""Initialize or resume a private, isolated A-group preparation job."""
import argparse, fcntl, json, os
from pathlib import Path
import subprocess, sys

HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--init',action='store_true');a=p.parse_args()
 root=Path(a.run_dir)
 if not root.is_absolute() or root.is_symlink() or not root.is_dir():raise ValueError('无效工作目录')
 marker=root/'A_GROUP_OWNED.json'
 if a.init:
  if marker.exists() or marker.is_symlink():raise ValueError('已有标记，禁止覆盖')
  # Installer is placed under runner before --init; all other existing entries stop init.
  if {x.name for x in root.iterdir()}!={'runner'} or not (root/'runner').is_dir():
   raise ValueError('工作目录非全新目录')
  with marker.open('x') as f:json.dump({'schema':'eduardo-a-group-prep-v1'},f)
 else:
  if marker.is_symlink() or not marker.is_file() or json.loads(marker.read_text()).get('schema')!='eduardo-a-group-prep-v1':
   raise ValueError('未找到属于本任务的目录标记')
 fd=os.open(str(root/'.bootstrap.lock'),os.O_RDWR|os.O_CREAT|getattr(os,'O_NOFOLLOW',0),0o600)
 try:
  try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:print('准备任务已经运行，无需重复启动。');return 0
  log=(root/'bootstrap-launch.log').open('a');os.chmod(log.name,0o600)
  try:
   subprocess.Popen([sys.executable,str(HERE/'bootstrap.py'),'--run-dir',str(root),'--lock-fd',str(fd)],
      stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(fd,))
  finally:log.close()
  print('已提交后台准备任务；不会使用 GPU 或启动训练。')
  print('断线不影响进程；容器本身停止会使进程停止，可在容器恢复后续传。')
  return 0
 finally:os.close(fd)

if __name__=='__main__':
 try:sys.exit(main())
 except Exception as e:print('未启动，原因类别：'+type(e).__name__,file=sys.stderr);sys.exit(1)
