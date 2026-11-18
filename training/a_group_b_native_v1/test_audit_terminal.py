import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import audit_posthoc as audit


class TerminalAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.work = self.root / 'pilots' / 'b'
        self.work.mkdir(parents=True)
        ray = self.root / 'r000abc'
        self.logs = ray / 'session_a' / 'logs'
        self.logs.mkdir(parents=True)
        (self.work / 'ray_tmp_path.txt').write_text(str(ray))
        (self.work / 'trainer.log').write_text('step:1 - training/global_step:1 - timing_s/update_weights:3.25\n')
        self.err = self.logs / 'worker-a-422.err'
        self.err.write_text('')
        self.out = self.logs / 'worker-a-422.out'
        self.out.write_text('worker started\n')

    def run_audit(self, strict=True):
        argv = ['audit_posthoc.py','--root',str(self.root),'--work',str(self.work)]
        if strict: argv.append('--check-terminal-failures')
        buf = io.StringIO()
        with patch.object(sys, 'argv', argv), patch.object(audit, 'load_stripper', return_value=str.strip), \
             patch.object(audit, 'load_tokenizer', return_value=None), contextlib.redirect_stdout(buf):
            rc = audit.main()
        self.assertNotIn('secret-payload', buf.getvalue())
        return rc, buf.getvalue()

    def test_retry_and_recovery_are_not_terminal(self):
        self.err.write_text('[judge] transient failure (TIMEOUT), retry 1/3: secret-payload\n'
                            'Judge JSON malformed, recovered decision via regex: OK\n')
        rc, text = self.run_audit()
        self.assertEqual(rc, 0)
        self.assertIn('| 裁判正则恢复 | 0 | 1 |', text)
        self.assertIn('| 1 | 3.250 |', text)

    def test_unlabelled_and_role_specific_failures(self):
        for marker in ('gave up after 3 retries, using fallback.',
                       'non-retriable 403; using fallback',
                       'EduardoInteraction: student call failed hard: secret-payload',
                       'Failed to parse judge response, defaulting to REJECT: secret-payload'):
            with self.subTest(marker=marker):
                self.err.write_text(marker+'\n')
                self.assertEqual(self.run_audit()[0], 2)
                self.assertEqual(self.run_audit(strict=False)[0], 0)

    def test_log_coverage_and_failure_precedence(self):
        self.err.unlink()
        self.assertEqual(self.run_audit()[0], 3)
        self.err.write_text('')
        rotated = self.logs / 'worker-a-422.err.2'
        rotated.write_text('worker\n')
        self.assertEqual(self.run_audit()[0], 3)
        rotated.rename(self.logs / 'worker-a-422.err.1')
        self.assertEqual(self.run_audit()[0], 0)
        (self.logs / 'worker-a-422.err.2.zst').touch()
        self.assertEqual(self.run_audit()[0], 3)
        self.err.write_text('EduardoInteraction: student call failed hard: secret-payload\n')
        self.assertEqual(self.run_audit()[0], 2)

    def test_missing_session_is_unknown(self):
        (self.work / 'ray_tmp_path.txt').unlink()
        self.assertEqual(self.run_audit()[0], 3)
