import hashlib
import importlib.util
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('summary', HERE/'summary.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
spec = importlib.util.spec_from_file_location('inventory', HERE.parent/'diagnostics'/'unresolved_inventory.py')
inv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inv)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(inv.canonical(data) + b'\n')


def body(value):
    return {'model':inv.MODEL, 'choices':[{'finish_reason':'stop','message':{'content':value}}]}


class SummaryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root, self.overlay, self.out = base/'root', base/'overlay', base/'new'
        self.root.mkdir()
        self.overlay.mkdir()
        self.out.mkdir()
        self.sid = 'synthetic-source'
        self.key = hashlib.sha256(self.sid.encode()).hexdigest()[:24]
        self.stage = 'EASY'
        self.make_source(self.stage)

    def tearDown(self):
        self.tmp.cleanup()

    def make_source(self, stage):
        folder = self.root/stage/'sources'/self.key
        selected = {'source_id':self.sid, 'problem':'synthetic question', 'answer':'1',
                    'transfer_problem':'synthetic question', 'transfer_answer':'1',
                    'transfer_target':'ours_easy', 'transfer_candidate_id':'candidate',
                    'transfer_position':3, 'student_correct':10, 'student_n_scored':16, 'score_tie':False}
        source_hash = inv.digest(selected)
        save(folder/'SOURCE.json', {'source_id':self.sid, 'sha256':source_hash})
        save(folder/'RAW_COMPLETE.json', {'source_id':self.sid,'samples':96,
             'grading_status':'ungraded', 'source_sha256':source_hash})
        rows = []
        results = []
        for t in ('source','ours_easy'):
            for phase in ('pre','control','post'):
                for n in range(16):
                    row = {'source_id':self.sid,'target':t,'phase':phase,'index':n,
                           'problem':'synthetic question', 'reference_answer':'1', 'student_response':'1'}
                    if t != 'source':
                        row.update(candidate_id='candidate', difficulty='easy', position_by_student_correct=3,
                                   historical_student_correct=10, historical_student_n_scored=16,
                                   historical_group_score_tie=False)
                    rows.append(row)
                    ident = inv.identity(row)
                    token = inv.digest({'identity':ident})
                    status = 'unknown' if t == 'source' and phase == 'pre' and n in (0,1) else 'correct'
                    binding = {'identity':ident, 'request_sha256':inv.digest(inv.payload(row)),
                               'model':inv.MODEL,'grader':'agnes_full_response_v2'}
                    result = {**binding,'status':status}
                    if status == 'unknown':
                        result['reason'] = 'undecidable_or_http_error'
                        old = self.root/'AGNES'/'receipts'/stage/self.key/token
                        save(old/'BINDING.json',binding)
                        save(old/'RESULT.json',result)
                        save(old/'ATTEMPT-0001-INTENT.json', {'attempt':1,'request_sha256':binding['request_sha256']})
                        save(old/'ATTEMPT-0001-RESPONSE.json', {'attempt':1,'http_status':200,'retryable':False,
                             'raw_response':body('true' if n == 0 else 'null')})
                    results.append(result)
        save(folder/'SCORE_INPUT.json', rows)
        report = {'source_id':self.sid,'input_sha256':s.sha((folder/'SCORE_INPUT.json').read_bytes()),
                  'model':inv.MODEL,'grader':'agnes_full_response_v2','results':results}
        report_path = self.root/'AGNES'/stage/'sources'/self.key/'RESULTS.json'
        save(report_path,report)
        candidate = inv.source_overlay(self.root,stage,self.key)
        save(self.overlay/stage/self.key/'OVERLAY.json',candidate)
        return rows, results

    def add_decision(self):
        score_raw = (self.root/'EASY'/'sources'/self.key/'SCORE_INPUT.json').read_bytes()
        report_raw = (self.root/'AGNES'/'EASY'/'sources'/self.key/'RESULTS.json').read_bytes()
        originals = json.loads(report_raw)['results']
        old = originals[1]
        token = inv.digest({'identity':old['identity']})
        dec = {'schema':s.SCHEMA,'stage':'EASY','source_key':self.key,'identity':old['identity'],
               'request_sha256':old['request_sha256'],'source_result_sha256':s.sha(report_raw),
               'input_sha256':s.sha(score_raw),'status':'incorrect','method':'agnes_rejudge','rounds':1}
        save(self.out/'decisions'/'EASY'/self.key/token/'DECISION.json',dec)
        folder = self.out/'receipts'/'EASY'/self.key/'round-1'/'auth-0000'/token
        save(folder/'RESULT.json',{k:old[k] for k in ('identity','request_sha256','model','grader')} | {'status':'incorrect'})
        save(folder/'ATTEMPT-0001-INTENT.json',{'attempt':1,'request_sha256':old['request_sha256']})
        save(folder/'ATTEMPT-0001-RESPONSE.json',{'attempt':1,'http_status':200,'retryable':False,
                                                   'raw_response':body('{"correct":false}')})
        return token, dec

    def test_overlay_and_rejudge_keep_old_and_denominators(self):
        self.add_decision()
        before = {p:p.read_bytes() for root in (self.root,self.overlay,self.out) for p in root.rglob('*') if p.is_file()}
        with patch.object(socket, 'socket', side_effect=AssertionError('network called')):
            groups, totals = s.collect(inv,self.root,self.overlay,self.out)
        self.assertEqual(groups['简单']['纳入来源'],1)
        self.assertEqual(totals['简单'][('原题','pre')],{'correct':15,'incorrect':1})
        self.assertEqual(sum(totals['简单'][('迁移题','post')].values()),16)
        self.assertEqual({p:p.read_bytes() for p in before},before)
        rendered=s.render(groups,totals)
        self.assertIn('| 简单',rendered)
        self.assertNotIn(self.sid,rendered)
        self.assertNotIn('synthetic question',rendered)
        self.assertNotIn(self.key,rendered)
        self.assertIn('未逐条对照冻结499清单', rendered)

    def test_corrupt_rejudge_receipt_remains_unknown(self):
        token,_=self.add_decision()
        p=self.out/'receipts'/'EASY'/self.key/'round-1'/'auth-0000'/token/'ATTEMPT-0001-RESPONSE.json'
        data=json.loads(p.read_bytes())
        data['raw_response']=body('{"correct":true}')
        save(p,data)
        groups, totals=s.collect(inv,self.root,self.overlay,self.out)
        self.assertEqual(groups['简单']['补判记录异常作答'],1)
        self.assertEqual(totals['简单'][('原题','pre')],{'correct':15,'unknown':1})

    def test_auth_epochs_are_selected_per_sample(self):
        token, _ = self.add_decision()
        report = json.loads((self.root/'AGNES'/'EASY'/'sources'/self.key/'RESULTS.json').read_bytes())
        other_token = inv.digest({'identity': report['results'][2]['identity']})
        self.assertNotEqual(other_token, token)
        other = self.out/'receipts'/'EASY'/self.key/'round-1'/'auth-0001'/other_token
        save(other/'RESULT.json', {'status':'correct'})
        groups, totals = s.collect(inv, self.root, self.overlay, self.out)
        self.assertEqual(groups['简单']['补判记录异常作答'], 0)
        self.assertEqual(totals['简单'][('原题','pre')], {'correct':15,'incorrect':1})

    def test_newer_incomplete_epoch_for_same_sample_fails_closed(self):
        token, _ = self.add_decision()
        newer = self.out/'receipts'/'EASY'/self.key/'round-1'/'auth-0001'/token
        save(newer/'ATTEMPT-0001-INTENT.json', {'attempt':1, 'request_sha256':'synthetic'})
        groups, totals = s.collect(inv, self.root, self.overlay, self.out)
        self.assertEqual(groups['简单']['补判记录异常作答'], 1)
        self.assertEqual(totals['简单'][('原题','pre')], {'correct':15,'unknown':1})

    def test_overlay_missing_preserves_original_unknown(self):
        (self.overlay/'EASY'/self.key/'OVERLAY.json').unlink()
        groups, totals=s.collect(inv,self.root,self.overlay,self.out)
        self.assertEqual(groups['简单']['叠层未能核验来源'],1)
        self.assertEqual(totals['简单'][('原题','pre')],{'correct':14,'unknown':2})

    def test_invalid_score_skips_entire_source(self):
        p=self.root/'EASY'/'sources'/self.key/'SCORE_INPUT.json'
        p.write_bytes(p.read_bytes()+b' ')
        groups,totals=s.collect(inv,self.root,self.overlay,self.out)
        self.assertEqual(groups['简单']['判分或输入异常'],1)
        self.assertEqual(groups['简单']['纳入来源'],0)
        self.assertFalse(totals['简单'][('原题','pre')])

    def test_changed_source_binding_skips_entire_source(self):
        p = self.root/'EASY'/'sources'/self.key/'SOURCE.json'
        doc = json.loads(p.read_bytes())
        doc['sha256'] = '0' * 64
        save(p, doc)
        groups, totals = s.collect(inv, self.root, self.overlay, self.out)
        self.assertEqual(groups['简单']['判分或输入异常'], 1)
        self.assertEqual(groups['简单']['纳入来源'], 0)
        self.assertFalse(totals['简单'][('原题','pre')])

    def test_duplicate_easy_pilot_excludes_both(self):
        self.make_source('PILOT')
        groups,totals=s.collect(inv,self.root,self.overlay,self.out)
        self.assertEqual(groups['简单']['重复来源记录'],2)
        self.assertEqual(groups['简单']['纳入来源'],0)
        self.assertFalse(totals['简单'][('原题','pre')])

    def test_partial_and_missing_grade_counted_not_included(self):
        folder=self.root/'HARD'/'sources'/('a'*24)
        save(folder/'SOURCE_HELD.json',{'error_type':'synthetic'})
        folder2=self.root/'HARD'/'sources'/('b'*24)
        save(folder2/'SOURCE.json',{'source_id':'x'})
        groups,totals=s.collect(inv,self.root,self.overlay,self.out)
        self.assertEqual(groups['困难']['暂停'],1)
        self.assertEqual(groups['困难']['不完整'],1)
        self.assertEqual(groups['困难']['纳入来源'],0)
        self.assertFalse(totals['困难'][('原题','pre')])

if __name__ == '__main__': unittest.main()
