"""Run one API producer and one or more vLLM student consumers on one host."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlsplit

from . import pipeline as p
from .store import Store, encode


class AdaptiveAPIWindow:
    """In-process AIMD request window; account-wide coordination is external."""
    def __init__(self, *, maximum=5, cooldown_base=1.0, cooldown_cap=60.0):
        if not 1 <= maximum <= 5 or cooldown_base < 0 or cooldown_cap < cooldown_base:
            raise ValueError('invalid API window settings')
        self.maximum = maximum
        self.limit = 1
        self.inflight = 0
        self.failures = 0
        self.blocked_until = 0.0
        self.epoch = 0
        self.auth_failed = False
        self.protocol_failures = 0
        self.stop_reason = None
        self.cooldown_base = cooldown_base
        self.cooldown_cap = cooldown_cap
        self.condition = threading.Condition()

    def acquire(self):
        """Get a permit before claiming a job; None means the circuit stopped."""
        with self.condition:
            while True:
                if self.stop_reason is not None:
                    return None
                delay = self.blocked_until - time.monotonic()
                if delay > 0:
                    self.condition.wait(timeout=delay)
                elif self.inflight < self.limit:
                    self.inflight += 1
                    return self.epoch
                else:
                    self.condition.wait()

    def report(self, lease, outcome):
        with self.condition:
            # A fatal circuit is latched. In-flight successes cannot reopen it.
            if self.stop_reason is not None:
                self.condition.notify_all()
                return
            if outcome == 'auth':
                self.auth_failed = True
                self.stop_reason = 'auth'
                self.epoch += 1
            elif outcome == 'model':
                self.stop_reason = 'model'
                self.epoch += 1
            elif outcome == 'protocol':
                self.protocol_failures += 1
                self.epoch += 1
                if self.protocol_failures >= 3:
                    self.stop_reason = 'protocol'
            elif outcome == 'congestion':
                self.limit = max(1, self.limit // 2)
                self.failures += 1
                # Limit exponent growth even for a prolonged outage.
                delay = min(self.cooldown_cap,
                            self.cooldown_base * (2 ** min(self.failures - 1, 16)))
                self.blocked_until = max(self.blocked_until, time.monotonic() + delay)
                self.epoch += 1
            elif outcome == 'success':
                # Before the circuit trips, any valid response interrupts a
                # consecutive protocol-error streak, even if it was in flight.
                self.protocol_failures = 0
                if (lease == self.epoch and not self.auth_failed and
                        time.monotonic() >= self.blocked_until):
                    # An older success must not cancel a newer backoff.
                    self.limit = min(self.maximum, self.limit + 1)
                    self.failures = 0
            self.condition.notify_all()

    def release(self):
        with self.condition:
            if self.inflight < 1:
                raise RuntimeError('API window permit underflow')
            self.inflight -= 1
            self.condition.notify_all()


def api_outcome(exc):
    """Classify fixed error categories without logging provider responses."""
    if not isinstance(exc, p.UnknownOutcome):
        # Repeated malformed response structures can raise these before a
        # provider-specific error is constructed. Do not sweep the queue.
        return 'protocol' if isinstance(exc, (AttributeError, TypeError, KeyError, ValueError)) else 'neutral'
    reason = str(exc)
    if reason == 'unexpected_provider_model':
        return 'model'
    if reason in ('invalid_json_response', 'response_schema', 'unparseable_content',
                  'nonobject_response', 'generation_schema', 'review_schema',
                  'review_enum', 'equivalence_schema', 'grade_schema',
                  'missing_content', 'incomplete_response', 'empty_candidate'):
        return 'protocol'
    status = None
    for prefix in ('http_', 'http_retries_exhausted_'):
        if reason.startswith(prefix) and reason[len(prefix):].isdigit():
            status = int(reason[len(prefix):])
            break
    if status in (401, 403):
        return 'auth'
    if status in (408, 429) or (status is not None and 500 <= status < 600):
        return 'congestion'
    if status is not None:
        return 'protocol'
    if reason in ('URLError', 'TimeoutError', 'ConnectionError',
                  'ConnectionResetError', 'ConnectionAbortedError',
                  'ConnectionRefusedError', 'RemoteDisconnected',
                  'OSError', 'SSLError', 'BrokenPipeError', 'InterruptedError',
                  'IncompleteRead', 'ProtocolError'):
        return 'congestion'
    return 'neutral'


def config(path):
    c=json.loads(Path(path).read_text())
    required=('namespace','api_base','api_model','student_base','student_model')
    if any(not isinstance(c.get(k),str) or not c[k].strip() for k in required):
        raise ValueError('missing configuration value')
    api=urlsplit(c['api_base'])
    if api.scheme!='https' or api.hostname!='api.agnes-ai.cn' or api.username or api.password:
        raise ValueError('only the authorized Agnes HTTPS endpoint is allowed')
    if urlsplit(c['student_base']).hostname not in ('127.0.0.1','localhost','::1'):
        raise ValueError('student endpoint must be local to the target server')
    if c.get('samples_per_candidate',16)!=16: raise ValueError('expected 16 samples')
    if c.get('api_workers',5)!=5: raise ValueError('API worker count must be 5')
    code={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
          for name in ('__main__.py','pipeline.py','store.py')}
    cfg={k:c[k] for k in required}
    cfg.update({'samples_per_candidate':16,'api_workers':5,
                'student_sampling':p.PARAMS,'student_prompt':p.STUDENT_PROMPT,'code_sha256':code,
                'generation_skill_sha256':hashlib.sha256(p.SKILL.encode()).hexdigest()})
    return cfg


def initialize(args,store,cfg):
    raw=Path(args.sources).read_bytes();rows=[json.loads(x) for x in raw.splitlines() if x.strip()]
    if not rows:raise ValueError('empty source file')
    ids=set(); indices=set()
    for row in rows:
        if row.get('source_split')!='train' or any(k not in row for k in ('source_id','source_row_index','problem','answer')):
            raise ValueError('expected official train source records')
        if not isinstance(row['source_id'],str) or type(row['source_row_index']) is not int:
            raise ValueError('invalid source identity')
        if row['source_id'] in ids or row['source_row_index'] in indices:raise ValueError('duplicate source identity')
        ids.add(row['source_id']);indices.add(row['source_row_index'])
    hold=set()
    if args.hold_source_ids:
        hold=set(Path(args.hold_source_ids).read_text().splitlines())
    elif not args.fresh_only:
        raise ValueError('provide historical hold list or explicitly assert fresh-only sources')
    store.bind('protocol',cfg)
    store.bind('source_sha256',hashlib.sha256(raw).hexdigest())
    store.bind('held_source_ids_sha256',hashlib.sha256(encode(sorted(hold)).encode()).hexdigest())
    for row in rows:
        if row['source_id'] in hold:continue
        # Data minimization follows the original generation request.
        source={k:row[k] for k in ('source_id','source_row_index','source_split','problem','answer','author_rates','direction_band') if k in row}
        payload={'source':source,'proposal':1,'feedback':[]}
        store.enqueue(p.job_id('generate',[source['source_id'],1]),'generate',payload)
        for paper in row.get('paper_transfers',[]):
            if any(not isinstance(paper.get(k),str) or not paper[k].strip() for k in ('candidate_id','question','answer')):
                raise ValueError('invalid author transfer')
            candidate={'candidate_id':paper['candidate_id'],'construction_status':'proposed',
                       'question':paper['question'],'answer':paper['answer'],
                       'solution':paper.get('solution',''),'notes':'Author-provided transfer record',
                       'origin':'paper_transfer'}
            item={'source':source,'candidate':candidate,'proposal':0,'feedback':[]}
            store.enqueue(p.job_id('review',candidate['candidate_id']),'review',item)
    print(json.dumps({'sources':len(rows),'historical_held':len(ids & hold),
                      'queued':len(ids-hold),'counts':store.counts()},ensure_ascii=False))


def run_one(store,cfg,key,kinds,observer=None):
    job=store.claim(kinds)
    if job is None:return False
    jid,kind,payload=job
    try:
        if kind=='student':p.student(store,jid,payload,cfg)
        else:getattr(p,kind)(store,jid,payload,cfg,key)
    except Exception as exc:
        # Inform the shared window before the SQLite write: other workers
        # must not claim new tasks after an authentication/congestion response.
        if observer is not None:observer(api_outcome(exc))
        # Keep any uncertain response as unknown. Never log the request, key, or data.
        reason=str(exc) if isinstance(exc,p.UnknownOutcome) and str(exc).replace('_','').isalnum() else None
        receipt=exc.receipt if isinstance(exc,p.UnknownOutcome) else None
        store.finish(jid,{'error_type':type(exc).__name__,'reason':reason,'receipt':receipt},state='unknown')
    else:
        # Truncation is graded locally, without an API success to grow the window.
        local_only = (kind == 'grade' and
                      payload.get('student', {}).get('finish_reason') == 'length')
        if observer is not None:observer('neutral' if local_only else 'success')
    return True


def loop(store,cfg,key,kinds,workers,once,api_window=None):
    def worker():
        while True:
            if api_window is None:
                found=run_one(store,cfg,key,kinds)
            else:
                lease=api_window.acquire()
                if lease is None:return
                try:
                    found=run_one(store,cfg,key,kinds,
                                  observer=lambda outcome:api_window.report(lease,outcome))
                finally:
                    api_window.release()
            if once:return
            if not found:time.sleep(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(worker) for _ in range(workers)]
        for f in futures:f.result()
    if api_window is not None and api_window.stop_reason is not None:
        messages = {
            'auth': 'Agnes 认证失败（401/403）',
            'model': 'Agnes 返回的模型与冻结配置不一致',
            'protocol': 'Agnes 连续三次响应或协议格式异常',
        }
        print(messages[api_window.stop_reason] + '：生产者已停止，未领取作业仍为 pending。',
              file=sys.stderr)
        return False
    return True


def export(store,path):
    # New-format export, one row per candidate; historical outputs are never overwritten.
    with store.connect() as db:
        rows=db.execute("SELECT id,kind,payload,state,result FROM jobs WHERE kind IN ('student','grade') ORDER BY id").fetchall()
    grouped={}
    for jid,kind,payload,state,result in rows:
        payload=json.loads(payload); c=payload['candidate']; key=(payload['source']['source_id'],c['candidate_id'])
        row=grouped.setdefault(key,{'source':payload['source'],'candidate':c,'samples':{}})
        sample=row['samples'].setdefault(str(payload['sample_index']),{'sample_id':payload['sample_id']})
        sample[kind]={'state':state,'result':json.loads(result) if result else None}
    eligible=set()
    for row in grouped.values():
        known=[]
        for sample in row['samples'].values():
            grade=sample.get('grade',{});result=grade.get('result') or {}
            decision=result.get('decision') or (result.get('verdict') or {}).get('decision')
            if grade.get('state')=='done' and decision in ('correct','incorrect'):known.append(decision)
        row['correct']=known.count('correct');row['known_n']=len(known)
        row['complete']=len(known)==16;row['pass_rate']=row['correct']/16 if row['complete'] else None
        if row['complete'] and row['candidate']['origin']=='self_generated':eligible.add(row['source']['source_id'])
    with Path(path).open('x',encoding='utf-8') as f:
        for row in grouped.values():
            row['source_has_complete_self_generated']=row['source']['source_id'] in eligible
            f.write(encode(row)+'\n')


def main():
    os.umask(0o077)
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--db',required=True);a.add_argument('--config',required=True)
    sub=a.add_subparsers(dest='command',required=True)
    init=sub.add_parser('init');init.add_argument('--sources',required=True);init.add_argument('--hold-source-ids');init.add_argument('--fresh-only',action='store_true')
    for name in ('api','student'):
        run=sub.add_parser(name);run.add_argument('--once',action='store_true')
        if name=='student':run.add_argument('--workers',type=int,default=32)
    sub.add_parser('status');out=sub.add_parser('export');out.add_argument('--out',required=True)
    args=a.parse_args();cfg=config(args.config);store=Store(args.db);store.bind('protocol',cfg)
    if args.command=='init':initialize(args,store,cfg);return
    if args.command=='status':print(json.dumps(store.counts()));return
    if args.command=='export':export(store,args.out);return
    if args.command=='student':
        if not 1<=args.workers<=64:raise ValueError('invalid student worker count')
        loop(store,cfg,None,['student'],args.workers,args.once);return
    key=os.environ.get('AGNES_API_KEY','').strip()
    if not key:raise ValueError('AGNES_API_KEY must be provided in the environment')
    # All runs of this program on this machine use the same lock; a single
    # producer contains exactly five threads for generation, review and grading.
    lock=Path.home()/'.cache'/'math-transfer-stream'/'api.lock'
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as f:
        # A distinct server/account consumer must coordinate externally.
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ok=loop(store,cfg,key,['generate','review','equivalence','grade'],5,args.once,
                api_window=AdaptiveAPIWindow())
        if not ok:raise SystemExit(2)

if __name__=='__main__':main()
