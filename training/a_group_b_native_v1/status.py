"""Aggregate-only B stage snapshot draft; does not start or modify work."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import time


def read_obj(path):
    try:
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        return data if isinstance(data,dict) else {}
    except (OSError,ValueError,TypeError): return {}


def timestamp(value):
    try: return datetime.fromtimestamp(float(value)).strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError,TypeError,OverflowError,OSError): return '未知'


def table(headers, rows):
    print('| ' + ' | '.join(headers) + ' |')
    print('| ' + ' | '.join('---' for _ in headers) + ' |')
    for row in rows: print('| ' + ' | '.join(str(x) for x in row) + ' |')


def get_conversation_counts(root):
    d=Path(root)/'eduardo_conversations'
    files=sorted(d.glob('conversations_*.jsonl')) if d.is_dir() else []
    if not files: return None
    values={}; invalid=0
    for p in files:
        try:
            with p.open(encoding='utf-8') as f:
                for line in f:
                    try:
                        row=json.loads(line)
                        uid=row.get('uuid')
                        if not isinstance(uid,str) or not uid: invalid+=1; continue
                        values[uid]=(row.get('note'),row.get('judge_pass'))
                    except (ValueError,TypeError,AttributeError): invalid+=1
        except OSError: invalid+=1
    total=Counter()
    for note,decision in values.values():
        total['条数']+=1
        if note=='ok':
            total['ok']+=1
            if decision==0: total['门槛拦截']+=1
        elif note=='empty_transcript': total['empty_transcript']+=1
        else: total['其他']+=1
    total['异常行']=invalid
    return total


def get_call_counts(path):
    """HTTP access-log line count; does not expose request content."""
    counts=Counter(); p=Path(path)
    if not p.is_file(): return None
    try:
        with p.open(errors='replace') as f:
            for line in f:
                if 'POST /v1/chat/completions' in line:
                    counts['http']+=1
    except OSError: return None
    return counts


def worker_counts(work):
    p=Path(work)/'ray_tmp_path.txt'
    try: d=Path(p.read_text().strip())/'session_latest'/'logs'
    except OSError: return None
    if not d.is_dir(): return None
    found={'judge':{},'student':{}}
    pattern=re.compile(r'created concurrency semaphore label=(judge|student) limit=(\d+) pid=(\d+)')
    seen=set()
    for fpath in list(d.glob('worker-*.err'))+list(d.glob('worker-*.out')):
        try:
            st=fpath.stat(); key=(st.st_dev,st.st_ino)
            if key in seen: continue
            seen.add(key)
            with fpath.open(errors='replace') as f:
                for line in f:
                    m=pattern.search(line)
                    if m: found[m.group(1)][m.group(3)]=int(m.group(2))
        except OSError: return None
    return found


def get_checkpoints(root):
    base=Path(root)/'verl_ckpts'
    if not base.is_dir(): return []
    rows=[]
    for d in sorted(base.glob('*/global_step_*')):
        if not d.is_dir(): continue
        size=0; errors=0
        for p in d.rglob('*'):
            try:
                if p.is_file() and not p.is_symlink(): size+=p.stat().st_size
            except OSError: errors+=1
        rows.append((str(d),round(size/(1024**2),2),'大小不完整' if errors else '未验证完整性'))
    return rows


def step_markers(root):
    p=Path(root)/'trainer.log'
    if not p.is_file(): return None
    try:
        pattern=re.compile(r'\bstep[:=]\s*(\d+)\b')
        with p.open(errors='replace') as f: return sorted({int(m.group(1)) for line in f for m in pattern.finditer(line)})
    except OSError: return None


def report(work):
    work=Path(work)
    state=read_obj(work/'orchestration.json')
    res=read_obj(work/'resources.json')
    print('B 阶段只读状态（草案；进程存活和数据完整性未实时确认）')
    table(('项目','状态'),[
        ('阶段',(res.get('phase') if state.get('phase') == '凑卡' else state.get('phase','未开始'))),('原因类别',state.get('reason','—') or '—'),
        ('保护触发显卡',state.get('protected_gpu') if state.get('protected_gpu') is not None else '无记录'),
        ('凑卡轮次',res.get('round','未记录')),('已交接角色',str(state.get('bindings',{}))),
        ('step 标记',str(step_markers(work)) if step_markers(work) is not None else '未发现日志'),
        ('显存峰值','见下表；进程归属无法证明时显示未知')])
    peaks=read_obj(work/'gpu_peaks.json')
    print('\n本任务进程显存峰值（MiB）')
    table(('显卡','本任务峰值'),[(gpu,v if isinstance(v,int) else '未知') for gpu,v in sorted(peaks.items())] or [('—','运行中或未知')])
    print('\n占位（存活状态为上次采样，非实时核验）')
    held=[]
    for item in res.get('held',[]):
        age=max(0,int(time.time()-item['acquired_epoch'])) if isinstance(item.get('acquired_epoch'),(float,int)) else '未知'
        held.append((item.get('role','?'),item.get('gpu','?'),item.get('acquired_at','?'),age,
                     '上次存活' if item.get('alive') else '上次退出'))
    table(('角色','显卡','占位开始','已占秒数','占位状态'),held or [('—','—','—','—','—')])
    print('\n交接时间')
    events=[]
    for item in res.get('events',[]):
        if item.get('action') in ('已预留','释放占位','启动服务','服务健康','启动训练'):
            events.append((item.get('role','?'),item.get('gpu','?'),item.get('action','?'),item.get('time','?')))
    table(('角色','显卡','动作','时间'),events or [('—','—','—','—')])
    print('\n阶段时间')
    table(('阶段','发生时间'),[(e.get('phase','?'),timestamp(e.get('time'))) for e in state.get('events',[])] or [('—','—')])
    conv=get_conversation_counts(work)
    judge=get_call_counts(work/'judge.log'); student=get_call_counts(work/'student.log')
    workers=worker_counts(work)
    print('\n调用和对话计数（运行中仅为快照）')
    rows=[('裁判服务 HTTP 请求',judge.get('http',0) if judge is not None else '未发现'),
          ('学生服务 HTTP 请求',student.get('http',0) if student is not None else '未发现')]
    for label in ('judge','student'):
        if workers and workers[label]:
            rows.append((label+'信号量进程数',len(workers[label])))
            rows.append((label+'已观测限额',','.join(map(str,sorted(set(workers[label].values()))))))
        else:
            rows.append((label+'信号量进程数','未知（未观察到日志）'))
    for name in ('条数','ok','empty_transcript','门槛拦截','其他','异常行'):
        rows.append((name,conv.get(name,0) if conv is not None else '未发现'))
    table(('项目','数量'),rows)
    print('\n检查点（仅文件与大小，不等于已完成 CPU 合并验证）')
    table(('目录','大小 MiB','验证状态'),get_checkpoints(work) or [('—','—','未发现')])

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--work',required=True); args=p.parse_args()
    report(args.work)
