import importlib.util, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

path = Path(__file__).with_name('pilot.py')
spec = importlib.util.spec_from_file_location('pilot', path)
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)

class FakeProc:
    pid=999999
    returncode=0
    def poll(self): return 0

class PilotTests(unittest.TestCase):
    def test_invalid_gpu_and_existing_dir(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/'pilots').mkdir()
            with patch.object(p,'ROOT',root):
                with self.assertRaisesRegex(RuntimeError,'四至六个不同'): p.run([0,0,1,2,3,4],1,root/'pilots'/'new')
                (root/'pilots'/'old').mkdir()
                with self.assertRaisesRegex(RuntimeError,'新目录'): p.run([0,1,2,3,4,5],1,root/'pilots'/'old')

    def test_training_command_frozen_values(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'pilots').mkdir(); work=root/'pilots'/'smoke'
            cmd=[]; env={}; fake=FakeProc(); services=[]; smokes=[]
            def trainer(args, **kwargs): cmd.extend(args); env.update(kwargs['env']); return fake
            def service(*args): services.append(args); return fake
            def smoke(*args): smokes.append(args)
            with patch.dict('os.environ', {'USER':'pilot_test'}), patch.object(p,'ROOT',root), patch.object(p,'validate'), patch.object(p,'check_ports'), patch.object(p,'check_cards'), patch.object(p,'service',side_effect=service), patch.object(p,'ready'), patch.object(p,'smoke',side_effect=smoke), patch.object(p,'stop_children'), patch.object(p.subprocess,'Popen',side_effect=trainer):
                p.run([0,1,2,3,4,5],1,work)
            self.assertIn('actor_rollout_ref.rollout.n=16',cmd)
            self.assertIn('custom_reward_function.reward_kwargs.num_student_attempts=16',cmd)
            self.assertIn('trainer.total_training_steps=1',cmd)
            self.assertIn('trainer.save_freq=1',cmd)
            self.assertTrue(any(x.startswith('data.train_files=') for x in cmd))
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0,1')
            expected = 'Llama-3.1-8B-Instruct-pilot_test'
            self.assertEqual(services[0][4], expected)
            self.assertEqual(smokes[0][1], expected)
            self.assertIn('vars.student_model_name='+expected,cmd)
            self.assertEqual(json.loads((work/'status.json').read_text())['phase'],'finished')

    def test_automatic_wait_and_card_split(self):
        for count in (4, 5, 6):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as d:
                root=Path(d); (root/'pilots').mkdir(); work=root/'pilots'/'smoke'
                calls=[]; fake=FakeProc()
                def service(*args): calls.append(args); return fake
                with patch.dict('os.environ', {'USER':'pilot_test'}), patch.object(p,'ROOT',root), patch.object(p,'validate'), patch.object(p,'check_ports'), patch.object(p,'wait_cards',return_value=list(range(count))), patch.object(p,'service',side_effect=service), patch.object(p,'ready'), patch.object(p,'smoke'), patch.object(p,'stop_children'), patch.object(p.subprocess,'Popen',return_value=fake):
                    p.run(None,1,work)
                self.assertEqual(len(calls[0][2]), 2 if count == 6 else 1)
                self.assertEqual(len(calls[1][2]), 1 if count == 4 else 2)
                self.assertEqual(calls[0][7], len(calls[0][2]))
                self.assertEqual(calls[1][6], len(calls[1][2]))

    def test_occupied_cards_are_not_eligible(self):
        self.assertTrue(p.eligible((81920,81050,0,False)))
        self.assertFalse(p.eligible((81920,81050,0,True)))
        self.assertFalse(p.eligible((81920,81050,1,False)))
        self.assertFalse(p.eligible((81920,70000,0,False)))

if __name__ == '__main__': unittest.main()
