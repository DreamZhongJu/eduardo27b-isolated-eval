import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import auto_resume as a
import rejudge_auto as r


def report(state, **counts):
    stage = {'visited':1, **counts}
    return r.format_report({'EASY':stage}, state)


class FakeWait:
    def __init__(self, limit=99):
        self.delays = []
        self.limit = limit
    def is_set(self): return False
    def wait(self, seconds):
        self.delays.append(seconds)
        return len(self.delays) >= self.limit


class Tests(unittest.TestCase):
    def test_stopping_wrapper_terminates_only_its_child(self):
        stop = threading.Event()
        timer = threading.Timer(.2, stop.set)
        timer.start()
        started = time.monotonic()
        try:
            result = a.launch_child([sys.executable, '-c', 'import time; time.sleep(30)'], stop)
        finally:
            timer.cancel()
        self.assertNotEqual(result.returncode, 0)
        self.assertLess(time.monotonic() - started, 8)

    def test_transient_categories_and_permanent(self):
        for code in (408,429,500,502,503,504):
            self.assertEqual(a.fault_kind({'http_status':code}), 'transient')
        self.assertEqual(a.fault_kind({'retryable':True,'error_type':'URLError'}), 'transient')
        self.assertEqual(a.fault_kind({'retryable':True,'error_type':'TypeError'}), 'permanent')
        self.assertEqual(a.fault_kind({'http_status':401}), 'auth')
        self.assertEqual(a.fault_kind({'http_status':200}), 'format')
        self.assertEqual(a.fault_kind({'http_status':200, 'retryable':True,
                                      'content_issue':'response_not_json'}), 'transient_format')

    def test_transient_failure_waits_and_continues_with_original_venv(self):
        wait = FakeWait()
        replies = iter([SimpleNamespace(returncode=2,stdout=report('出现异常',service_deferred=1)),
                        SimpleNamespace(returncode=0,stdout=report('本轮扫描完成'))])
        seen=[]
        def launch(cmd): seen.append(cmd); return next(replies)
        with patch.object(a,'snapshot',return_value=set()), patch.object(a,'new_faults',return_value=['transient']):
            status=a.supervise(Path('/venv/bin/python'),5,lambda _:None,wait=wait,launch=launch,pilot=False)
        self.assertEqual(status,'DONE')
        self.assertEqual(wait.delays,[15])
        self.assertEqual(seen[0][0],'/venv/bin/python')
        self.assertEqual(seen[0][1],str(a.REJUDGE))
        self.assertIn('--workers',seen[0])

    def test_repeated_outage_uses_bounded_exponential_wait(self):
        wait=FakeWait(limit=7)
        bad=SimpleNamespace(returncode=2,stdout=report('出现异常',service_deferred=1))
        with patch.object(a,'snapshot',return_value=set()), patch.object(a,'new_faults',return_value=['transient']):
            self.assertEqual(a.supervise(Path('/v/python'),5,lambda _:None,wait=wait,launch=lambda _:bad,pilot=False),'STOPPED')
        self.assertEqual(wait.delays,[15,30,60,120,240,300,300])

    def test_format_fault_and_auth_failure_stop_not_loop(self):
        wait=FakeWait()
        bad=SimpleNamespace(returncode=2,stdout=report('出现异常',service_deferred=1))
        with patch.object(a,'snapshot',return_value=set()), patch.object(a,'new_faults',return_value=['format']):
            self.assertEqual(a.supervise(Path('/v/python'),5,lambda _:None,wait=wait,launch=lambda _:bad,pilot=False),'NON_RETRYABLE_SERVICE')
        auth=SimpleNamespace(returncode=2,stdout=report('出现异常',auth_deferred=1))
        with patch.object(a,'snapshot',return_value=set()):
            self.assertEqual(a.supervise(Path('/v/python'),5,lambda _:None,wait=wait,launch=lambda _:auth,pilot=False),'NON_RETRYABLE')
        self.assertEqual(wait.delays,[])

    def test_systemic_format_fuse_stops_without_cycling(self):
        wait=FakeWait()
        bad=SimpleNamespace(returncode=2,stdout=report('出现异常',service_deferred=1,format_fuse_trip=1))
        with patch.object(a,'snapshot',return_value=set()):
            self.assertEqual(a.supervise(Path('/v/python'),5,lambda _:None,wait=wait,launch=lambda _:bad,pilot=False),
                             'FORMAT_CIRCUIT_OPEN')
        self.assertEqual(wait.delays,[])

    def test_fuse_concurrent_success_rechecked_after_child_exit(self):
        wait=FakeWait()
        responses=iter([SimpleNamespace(returncode=2,stdout=report('出现异常',service_deferred=1,format_fuse_trip=1)),
                        SimpleNamespace(returncode=0,stdout=report('本轮扫描完成'))])
        with patch.object(a,'snapshot',return_value=set()):
            outcome=a.supervise(Path('/v/python'),5,lambda _:None,wait=wait,
                                launch=lambda _:next(responses),circuit_open=lambda _:False,pilot=False)
        self.assertEqual(outcome,'DONE')
        self.assertEqual(wait.delays,[15])

    def test_pilot_succeeds_then_unlimited(self):
        replies=iter([SimpleNamespace(returncode=0,stdout=report('已到本轮上限', api_samples=1, api_requests=1, rejudged_correct=1)),
                      SimpleNamespace(returncode=0,stdout=report('本轮扫描完成'))])
        seen=[]
        def launch(cmd):
            seen.append(cmd)
            return next(replies)
        with patch.object(a,'snapshot',return_value=set()), patch.object(a,'snapshot_decisions',return_value=set()):
            outcome=a.supervise(Path('/v/python'),5,lambda _:None,wait=FakeWait(),launch=launch,
                                verify_pilot=lambda before,report: True)
        self.assertEqual(outcome,'DONE')
        self.assertEqual(seen[0][seen[0].index('--max-items')+1],'1')
        self.assertEqual(seen[0][seen[0].index('--workers')+1],'1')
        self.assertEqual(seen[1][seen[1].index('--max-items')+1],'0')
        self.assertEqual(seen[1][seen[1].index('--workers')+1],'5')

    def test_pilot_does_not_start_bulk_without_persisted_verdict(self):
        reply=SimpleNamespace(returncode=0,stdout=report('已到本轮上限',api_samples=1,api_requests=1,format_deferred=1))
        seen=[]
        with patch.object(a,'snapshot',return_value=set()), patch.object(a,'snapshot_decisions',return_value=set()):
            outcome=a.supervise(Path('/v/python'),5,lambda _:None,wait=FakeWait(),
                                launch=lambda cmd:(seen.append(cmd) or reply),verify_pilot=lambda before,report:False)
        self.assertEqual(outcome,'PILOT_NOT_VERIFIED')
        self.assertEqual(len(seen),1)

    def test_pilot_verdict_checks_new_api_decision_and_report(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch.object(a,'OUTPUT',root):
                before=a.snapshot_decisions()
                p=root/'decisions'/'synthetic'/'DECISION.json'
                p.parent.mkdir(parents=True)
                p.write_text(json.dumps({'method':'agnes_rejudge','status':'correct'}))
                _,rows=a.parse_report(report('已到本轮上限',api_samples=1,api_requests=1,rejudged_correct=1))
                self.assertTrue(a.pilot_verdict(before,rows))
                _,rows=a.parse_report(report('已到本轮上限',api_samples=0,api_requests=1,rejudged_correct=1))
                self.assertFalse(a.pilot_verdict(before,rows))

    def test_results_compare_new_files_and_omit_semantic_null(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for n, response in enumerate(({'http_status':429}, {'http_status':200,'semantic':'null'})):
                p=root/str(n); p.mkdir(); (p/'RESULT.json').write_text(json.dumps({'status':'unknown'}))
                (p/'ATTEMPT-0001-RESPONSE.json').write_text(json.dumps(response))
            classify=lambda response: 'semantic_null' if response.get('semantic')=='null' else 'technical'
            self.assertEqual(a.new_faults(root,set(),classify),['transient'])
            before=a.snapshot(root)
            self.assertEqual(a.new_faults(root,before,classify),[])

    def test_pilot_allows_existing_receipt_recovered_alongside_new_api_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch.object(a,'OUTPUT',root):
                before=a.snapshot_decisions()
                for k,status in enumerate(('correct','incorrect')):
                    p=root/'decisions'/str(k)/'DECISION.json'
                    p.parent.mkdir(parents=True)
                    p.write_text(json.dumps({'method':'agnes_rejudge','status':status}))
                _,rows=a.parse_report(report('已到本轮上限',api_samples=1,api_requests=1,
                                             rejudged_correct=1,rejudged_incorrect=1))
                self.assertTrue(a.pilot_verdict(before,rows))

    def test_report_parser_requires_all_aggregate_rows_and_redacts_other_text(self):
        data=report('出现异常',service_deferred=1)
        state, rows=a.parse_report(data+'\nSECRET-WOULD-NOT-PRINT\n')
        self.assertEqual(state,'出现异常')
        self.assertEqual(rows['服务异常暂缓'][-1],1)
        self.assertNotIn('SECRET-WOULD-NOT-PRINT',repr(rows))
        with self.assertRaises(ValueError): a.parse_report('本轮状态：出现异常\n')

if __name__=='__main__': unittest.main()
