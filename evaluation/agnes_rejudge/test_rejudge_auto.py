import hashlib
import json
import os
import time
import threading
import unittest
from unittest.mock import patch

import rejudge_auto as auto
import auto_resume as supervisor
import test_rejudge as original_tests

j = original_tests.j


class TestRejudgeAuto(unittest.TestCase):
    reference_overlay = original_tests.TestRejudge.reference_overlay
    overlay_one = original_tests.TestRejudge.overlay_one

    def setUp(self):
        self.old_r = original_tests.r
        original_tests.r = auto
        original_tests.TestRejudge.setUp(self)

    def tearDown(self):
        original_tests.TestRejudge.tearDown(self)
        original_tests.r = self.old_r

    def call(self, caller):
        original_grade = j.grade_one
        def no_sleep(*args, **kwargs):
            return original_grade(*args, sleep=lambda _: None, **kwargs)
        class NoWaitWindow:
            def acquire(self): pass
            def release(self, *args, **kwargs): pass
        with patch.object(j, 'grade_one', no_sleep), patch.object(j, 'AdaptiveWindow', return_value=NoWaitWindow()):
            return auto.run(j, self.root, self.out, self.overlay, 'synthetic', 1, caller,
                            recovery=self.recovery)

    def test_real_pilot_limit_one_persists_one_new_api_verdict(self):
        def caller(body,key):
            self.calls += 1
            return 200, original_tests.response('true')
        original_grade=j.grade_one
        def no_sleep(*args, **kwargs):
            return original_grade(*args, sleep=lambda _:None, **kwargs)
        class NoWaitWindow:
            def acquire(self): pass
            def release(self,*args,**kwargs): pass
        stages={}
        with patch.object(supervisor,'OUTPUT',self.out):
            before=supervisor.snapshot_decisions()
            with patch.object(j,'grade_one',no_sleep),patch.object(j,'AdaptiveWindow',return_value=NoWaitWindow()):
                counts=auto.run(j,self.root,self.out,self.overlay,'synthetic',1,caller,max_items=1,
                                per_stage=stages,recovery=self.recovery)
            state,rows=supervisor.parse_report(auto.format_report(stages,'已到本轮上限'))
            self.assertTrue(supervisor.pilot_verdict(before,rows))
            self.assertEqual(counts['api_samples'],1)
            self.assertEqual(self.calls,1)

    def test_format_is_deferred_after_two_cycles_and_other_samples_continue(self):
        before = self.result.read_bytes()
        def caller(body, key):
            self.calls += 1
            return 200, original_tests.response('invalid' if self.calls <= 8 else 'true')
        first = self.call(caller)
        self.assertEqual(first['service_failed'], 1)
        self.assertEqual(self.calls, 4)
        second = self.call(caller)
        self.assertEqual(second['format_deferred'], 1)
        self.assertEqual(second['rejudged_correct'], 2)
        self.assertEqual(self.calls, 10)
        markers = list((self.out/'deferred').rglob('DEFERRED.json'))
        self.assertEqual(len(markers), 1)
        record = json.loads(markers[0].read_bytes())
        self.assertEqual(record['schema'], auto.DEFERRED_SCHEMA)
        self.assertEqual(record['reason'], 'retryable_response_format')
        self.assertEqual(record['format_cycles'], 2)
        self.assertEqual(self.result.read_bytes(), before)
        third = self.call(caller)
        self.assertEqual(third['format_deferred'], 1)
        self.assertEqual(third['already_correct'], 2)
        self.assertEqual(self.calls, 10)

    def test_crash_after_result_before_deferred_recovers_without_api(self):
        def caller(body, key):
            self.calls += 1
            return 200, original_tests.response('invalid' if self.calls <= 8 else 'true')
        self.call(caller)
        old_put = j.put_once
        def interrupted(path, data):
            if path.name == 'DEFERRED.json':
                raise OSError('synthetic interruption')
            return old_put(path, data)
        with patch.object(j, 'put_once', interrupted):
            second = self.call(caller)
        self.assertEqual(second['item_error'], 1)
        self.assertEqual(self.calls, 8)  # storage failure stops before further requests
        self.assertFalse(list((self.out/'deferred').rglob('DEFERRED.json')))
        third = self.call(caller)
        self.assertEqual(third['format_deferred'], 1)
        self.assertEqual(third['rejudged_correct'], 2)
        self.assertEqual(self.calls, 10)

    def test_three_distinct_format_failures_trip_persistent_fuse_and_success_resets(self):
        def record(root, sub, ident, schema, **extra):
            source_id = f'synthetic:{ident}'
            source_key = hashlib.sha256(source_id.encode()).hexdigest()[:24]
            identity = {'source_id':source_id, 'target':'source', 'phase':'pre', 'index':0}
            token = j.digest({'identity':identity})
            path = root/sub/'EASY'/source_key/token/('DEFERRED.json' if sub == 'deferred' else 'DECISION.json')
            value = {'schema':schema, 'stage':'EASY', 'source_key':source_key, 'identity':identity,
                     'request_sha256':'a'*64, 'source_result_sha256':'b'*64, 'input_sha256':'c'*64, **extra}
            original_tests.save(path,value)
            return path
        paths=[]
        for i in range(3):
            paths.append(record(self.out, 'deferred', i, auto.DEFERRED_SCHEMA,
                                reason='retryable_response_format',format_cycles=2))
        self.assertTrue(auto.format_circuit_open(self.out))
        good=record(self.out,'decisions',10,auto.SCHEMA,method='agnes_rejudge',status='correct',rounds=1)
        self.assertFalse(auto.write_format_checkpoint(self.out,j,'api_verdict',good))
        os.utime(paths[0], ns=(time.time_ns()+10**15,time.time_ns()+10**15))
        self.assertTrue(auto.format_circuit_open(self.out))
        auto.write_format_checkpoint(self.out,j,'operator_reset')
        self.assertFalse(auto.format_circuit_open(self.out))

    def test_checkpoint_waits_for_deferred_lock_and_keeps_latched_fuse(self):
        for i in range(3):
            sid=f'concurrent:{i}'
            source_key=hashlib.sha256(sid.encode()).hexdigest()[:24]
            ident={'source_id':sid,'target':'source','phase':'pre','index':0}
            token=j.digest({'identity':ident})
            path=self.out/'deferred'/'EASY'/source_key/token/'DEFERRED.json'
            original_tests.save(path, {'schema':auto.DEFERRED_SCHEMA, 'stage':'EASY', 'source_key':source_key,
                'identity':ident,'request_sha256':'a'*64,'source_result_sha256':'b'*64,
                'input_sha256':'c'*64,'reason':'retryable_response_format','format_cycles':2})
        finished=threading.Event()
        errors=[]
        def success():
            try:
                # A success arriving after the third deferred cannot reset the fuse.
                auto.write_format_checkpoint(self.out,j,'api_verdict',None)
            except Exception as e:
                errors.append(e)
            finally:
                finished.set()
        with auto.format_lock(self.out):
            thread=threading.Thread(target=success)
            thread.start()
            self.assertFalse(finished.wait(.1))
        thread.join(2)
        self.assertTrue(finished.is_set())
        self.assertFalse(errors)
        self.assertTrue(auto.format_circuit_open(self.out))

    def test_open_fuse_blocks_all_api_until_explicit_append_only_reset(self):
        for i in range(3):
            sid=f'failed:{i}'
            source_key=hashlib.sha256(sid.encode()).hexdigest()[:24]
            ident={'source_id':sid,'target':'source','phase':'pre','index':0}
            token=j.digest({'identity':ident})
            path=self.out/'deferred'/'EASY'/source_key/token/'DEFERRED.json'
            original_tests.save(path, {'schema':auto.DEFERRED_SCHEMA, 'stage':'EASY', 'source_key':source_key,
                'identity':ident, 'request_sha256':'a'*64, 'source_result_sha256':'b'*64,
                'input_sha256':'c'*64, 'reason':'retryable_response_format', 'format_cycles':2})
            older=time.time_ns()-1_000_000_000
            os.utime(path, ns=(older,older))
        old_result=self.result.read_bytes()
        def caller(body,key):
            self.calls+=1
            return 200, original_tests.response('true')
        stages={}
        blocked=auto.run(j,self.root,self.out,self.overlay,'synthetic',1,caller,
                         recovery=self.recovery,per_stage=stages)
        self.assertEqual(blocked['service_failed'],1)
        self.assertEqual(blocked['format_fuse_trip'],1)
        self.assertEqual(self.calls,0)
        self.assertEqual(stages['EASY']['format_fuse_trip'],1)
        resumed=auto.run(j,self.root,self.out,self.overlay,'synthetic',1,caller,
                         recovery=self.recovery,reset_format_circuit=True)
        self.assertEqual(resumed.get('rejudged_correct'),3,resumed)
        self.assertEqual(self.calls,3)
        self.assertEqual(len(list((self.out/'control'/'format-checkpoints').glob('*.json'))),1)
        self.assertEqual(self.result.read_bytes(),old_result)

if __name__ == '__main__': unittest.main()
