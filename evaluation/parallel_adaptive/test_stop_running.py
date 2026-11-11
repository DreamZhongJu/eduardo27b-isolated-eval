import json
import os
import signal
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import stop_running as s


class StopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.proc = Path(self.tmp.name) / 'proc'
        self.proc.mkdir()

    def process(self, pid, args, env=()):
        p = self.proc / str(pid)
        p.mkdir()
        (p / 'cmdline').write_bytes(b'\0'.join(args) + b'\0')
        (p / 'environ').write_bytes(b'\0'.join(env) + b'\0')
        return p

    def argv(self):
        return [b'/workspace/eduardo-vllm018/bin/python', os.fsencode(str(s.SCRIPT)), b'--run', b'--wait-gpus']

    def test_exact_match_and_different_root(self):
        self.assertTrue(s.exact_supervisor(self.argv()))
        self.assertFalse(s.exact_supervisor(self.argv() + [b'--root', b'/other']))
        self.assertFalse(s.exact_supervisor(self.argv() + [b'--root=/other']))
        self.assertFalse(s.exact_supervisor(self.argv() + [b'--root', os.fsencode(str(s.ROOT)), b'--root=/other']))
        self.assertFalse(s.exact_supervisor([b'python', b'/other/supervisor.py', b'--run']))

    def test_only_exact_process_selected(self):
        self.process(11, [b'python', b'/other/supervisor.py', b'--run'])
        self.process(12, self.argv())
        self.assertEqual(s.list_matches(self.proc), [12])

    def test_tagged_worker_is_conflict(self):
        self.process(13, [b'python', b'-m', b'vllm.entrypoints.openai.api_server'],
                     [b'FIVEGPU_RUN_ROOT=' + os.fsencode(str(s.ROOT))])
        self.assertTrue(s.conflict(self.proc))

    def test_never_signals_multiple_matches(self):
        self.process(11, self.argv())
        self.process(12, self.argv())
        with mock.patch.object(s, 'verify_script'), mock.patch.object(s, 'send_pidfd') as send:
            self.assertEqual(s.stop(proc=self.proc), 'MULTIPLE_MATCHES')
            send.assert_not_called()

    def test_signals_exact_parent_once_and_verifies_cleanup(self):
        self.process(12, self.argv())
        with mock.patch.object(s, 'verify_script'), \
             mock.patch.object(s, 'open_pidfd', side_effect=lambda _: os.open('/dev/null', os.O_RDONLY)), \
             mock.patch.object(s, 'exited', side_effect=[False, True]), \
             mock.patch.object(s, 'send_pidfd') as send, \
             mock.patch.object(s, 'conflict', return_value=False), \
             mock.patch.object(s, 'lock_free', return_value=True):
            self.assertEqual(s.stop(proc=self.proc), 'STOPPED_READY')
            self.assertEqual(send.call_count, 1)
            self.assertEqual(send.call_args.args[1], signal.SIGTERM)

    def test_timeout_does_not_claim_ready(self):
        self.process(12, self.argv())
        with mock.patch.object(s, 'verify_script'), \
             mock.patch.object(s, 'open_pidfd', side_effect=lambda _: os.open('/dev/null', os.O_RDONLY)), \
             mock.patch.object(s, 'exited', return_value=False), \
             mock.patch.object(s, 'send_pidfd') as send:
            self.assertEqual(s.stop(timeout=0, proc=self.proc), 'STOP_PENDING_OR_CONFLICT')
            self.assertEqual(send.call_count, 1)

    def test_never_starts_if_existing_occupies_lock(self):
        with mock.patch.object(s, 'verify_script'), mock.patch.object(s, 'lock_free', return_value=False):
            self.assertEqual(s.stop(proc=self.proc), 'CONFLICT_OR_LOCKED')


if __name__ == '__main__':
    unittest.main()
