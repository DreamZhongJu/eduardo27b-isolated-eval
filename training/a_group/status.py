#!/usr/bin/env python3
"""Display read-only preparation status without logs, question text or secrets."""
import argparse, fcntl, json, os, unicodedata
from pathlib import Path

def width(x):return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in str(x))
def pad(x,n):x=str(x);return x+' '*(n-width(x))
def table(rows):
 widths=[max(width(r[i]) for r in rows) for i in range(3)]
 line='+'+'+'.join('-'*(n+2) for n in widths)+'+'
 print(line)
 for i,row in enumerate(rows):
  print('| '+' | '.join(pad(x,n) for x,n in zip(row,widths))+' |')
  if i==0:print(line)
 print(line)

def main():
 a=argparse.ArgumentParser();a.add_argument('--run-dir',required=True);root=Path(a.parse_args().run_dir)
 if not root.is_absolute() or root.is_symlink() or not (root/'A_GROUP_OWNED.json').is_file():
  print('未发现本任务工作目录。');return 1
 active=False
 try:
  fd=os.open(str(root/'.bootstrap.lock'),os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
  try:
   try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(fd,fcntl.LOCK_UN)
   except BlockingIOError:active=True
  finally:os.close(fd)
 except FileNotFoundError:pass
 try:data=json.loads((root/'status.json').read_text())
 except (OSError,ValueError):data={'tasks':{}}
 print('A组准备进程：','运行中' if active else '未运行')
 rows=[('项目','状态','说明')]
 for key in ('作者代码','作者数据','Tutor模型','门槛模型'):
  x=data.get('tasks',{}).get(key,{});state=x.get('status','尚未开始');detail=x.get('detail','等待状态')
  if state=='进行中' and not active:state='中断';detail='进程已停止，可续传'
  rows.append((key,state,detail))
 table(rows)
 print('说明：只准备公开资产；未占 GPU、未启动训练。')
 print('训练环境、学生模型版本和可用显卡仍需单独验证。')
 return 0

if __name__=='__main__':raise SystemExit(main())
