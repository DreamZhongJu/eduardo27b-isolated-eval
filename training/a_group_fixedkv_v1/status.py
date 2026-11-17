#!/usr/bin/env python3
import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('run'); a=p.parse_args()
root=Path('/workspace/A_group_repro_20261117/pilots')
if not a.run.isidentifier() or len(a.run)>50: raise SystemExit('试点名称无效')
try: info=json.loads((root/a.run/'status.json').read_text())
except (OSError,ValueError): raise SystemExit('未找到试点状态；可能尚未通过启动检查')
pid=info.get('pid'); alive=False
if isinstance(pid,int) and pid>0:
    try:
        args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        alive=any(x.endswith(b'/pilot.py') for x in args) and a.run.encode() in args
    except OSError: pass
print('+----------+------------------------------------------+')
print(f'| 试点     | {a.run:<40} |')
names={'waiting_gpu':'等待空闲显卡','starting':'正在加载模型','training':'训练中',
       'finished':'试点结束，待核查','failed':'失败，查看日志'}
phase=names.get(info.get('phase'),info.get('phase','未知'))
print(f'| 阶段     | {phase:<40} |')
print(f'| 控制进程 | {"运行中" if alive else "未发现" :<40} |')
print('+----------+------------------------------------------+')
print('说明：',info.get('message','未知'))
print('日志位置：',root/a.run)
