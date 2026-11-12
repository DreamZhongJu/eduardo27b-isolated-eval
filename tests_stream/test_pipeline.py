import json
import argparse
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from stream_train import pipeline as p
from stream_train import __main__ as cli
from stream_train.store import Store


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.store=Store(self.root/'run.db')
        self.cfg={'namespace':'test','api_base':'https://example.invalid/v1','api_model':'model',
                  'student_base':'http://127.0.0.1:8000/v1','student_model':'student'}
        self.source={'source_id':'train:1','source_row_index':1,'source_split':'train',
                     'problem':'Original','answer':'5','direction_band':'mixed'}
    def tearDown(self):self.tmp.cleanup()

    def test_end_to_end_and_no_reclaim(self):
        job=p.job_id('generate',[self.source['source_id'],1])
        self.store.enqueue(job,'generate',{'source':self.source,'proposal':1,'feedback':[]})
        generated={'construction_status':'proposed','question':'Question','answer':'5','solution':'Solution','notes':''}
        reviewed={'solvability':'solvable','independent_answer':'5','independent_solution':'5',
                  'core_relation':'related','direct_repeat':'no','explanation':''}
        graded={'final_answer_complete':'yes','decision':'correct','student_final_answer':'5','reason':''}
        def fake_api(base,key,model,prompt,**kwargs):
            if prompt.startswith('Create one'):return generated,{}
            if prompt.startswith('Independently'):return reviewed,{}
            return graded,{}
        def fake_request(url,body,key,**kwargs):
            return {'model':'student','choices':[{'message':{'content':'\\boxed{5}'},'finish_reason':'stop'}]},[200]
        with patch.object(p,'api_message',side_effect=fake_api),patch.object(p,'request_json',side_effect=fake_request):
            self.assertTrue(cli.run_one(self.store,self.cfg,'key',['generate']))
            self.assertTrue(cli.run_one(self.store,self.cfg,'key',['review']))
            for _ in range(16):self.assertTrue(cli.run_one(self.store,self.cfg,None,['student']))
            for _ in range(16):self.assertTrue(cli.run_one(self.store,self.cfg,'key',['grade']))
        self.assertFalse(cli.run_one(self.store,self.cfg,None,['student']))
        out=self.root/'out.jsonl';cli.export(self.store,out)
        row=json.loads(out.read_text());self.assertTrue(row['complete']);self.assertEqual(row['correct'],16)
        self.assertEqual(row['pass_rate'],1.0);self.assertTrue(row['source_has_complete_self_generated'])

    def test_interrupted_request_is_held(self):
        self.store.enqueue('one','student',{'x':1})
        self.assertEqual(self.store.claim(['student'])[0],'one')
        self.assertIsNone(self.store.claim(['student']))
        self.assertEqual(self.store.counts()[0]['state'],'started')

    def test_source_collision_is_rejected(self):
        self.store.enqueue('one','student',{'x':1})
        with self.assertRaises(ValueError):self.store.enqueue('one','student',{'x':2})

    def test_truncated_is_incorrect_without_api(self):
        self.store.enqueue('one','grade',{'candidate':{'question':'q','answer':'a'},
                           'student':{'text':'a','finish_reason':'length'}})
        with patch.object(p,'api_message') as api:
            cli.run_one(self.store,self.cfg,'key',['grade']);api.assert_not_called()
        with self.store.connect() as db:
            result=json.loads(db.execute('select result from jobs').fetchone()[0])
        self.assertEqual(result['decision'],'incorrect')

    def test_model_says_no_final_answer_even_if_decision_correct(self):
        self.store.enqueue('one','grade',{'candidate':{'question':'q','answer':'a'},
                           'student':{'text':'reasoning','finish_reason':'stop'}})
        verdict={'decision':'correct','final_answer_complete':'no','student_final_answer':'','reason':''}
        with patch.object(p,'api_message',return_value=(verdict,{})):
            cli.run_one(self.store,self.cfg,'key',['grade'])
        with self.store.connect() as db:
            state,result=db.execute('select state,result from jobs').fetchone()
        self.assertEqual(state,'done');self.assertEqual(json.loads(result)['decision'],'incorrect')

    def test_config_accepts_only_authorized_api_and_local_student(self):
        path=self.root/'config.json'
        config={'namespace':'x','api_base':'https://api.agnes-ai.cn/v1',
                'api_model':'agnes-3.0-flash','student_base':'http://127.0.0.1:8000/v1',
                'student_model':'student-8b'}
        path.write_text(json.dumps(config));self.assertEqual(cli.config(path)['api_workers'],5)
        config['student_base']='https://remote.example/v1';path.write_text(json.dumps(config))
        with self.assertRaises(ValueError):cli.config(path)

    def test_historical_hold_blocks_all_tasks_for_source(self):
        source_file=self.root/'input.jsonl'; source_file.write_text(json.dumps(self.source)+'\n')
        hold=self.root/'hold.txt'; hold.write_text(self.source['source_id']+'\n')
        args=argparse.Namespace(sources=source_file,hold_source_ids=hold,fresh_only=False)
        cli.initialize(args,self.store,self.cfg)
        self.assertEqual(self.store.counts(),[])

    def test_author_candidate_is_separate_from_own_proposal(self):
        source=dict(self.source,paper_transfers=[{'candidate_id':'author-1','question':'q','answer':'a'}])
        source_file=self.root/'input.jsonl';source_file.write_text(json.dumps(source)+'\n')
        args=argparse.Namespace(sources=source_file,hold_source_ids=None,fresh_only=True)
        cli.initialize(args,self.store,self.cfg)
        self.assertEqual({r['kind'] for r in self.store.counts()},{'generate','review'})

    def test_five_api_workers_share_the_same_queue(self):
        for i in range(5):self.store.enqueue('g'+str(i),'generate',{'i':i})
        barrier=threading.Barrier(5,timeout=15); lock=threading.Lock();active=[0,0]
        def generate(store,job,payload,cfg,key):
            with lock:
                active[0]+=1;active[1]=max(active[1],active[0])
            barrier.wait()
            with lock:active[0]-=1
            store.finish(job,{'ok':True})
        with patch.object(p,'generate',side_effect=generate):
            cli.loop(self.store,self.cfg,'fake',['generate'],5,True)
        self.assertEqual(active[1],5)
        self.assertEqual(self.store.counts(),[{'kind':'generate','state':'done','count':5}])

if __name__=='__main__':unittest.main()
