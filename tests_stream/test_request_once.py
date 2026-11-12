import io
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from stream_train.pipeline import UnknownOutcome, api_message, request_json


class RequestOnceTest(unittest.TestCase):
    def test_http_errors_are_preserved_without_resubmission(self):
        for status in (401, 408, 429, 500, 502, 503, 504):
            with self.subTest(status=status):
                opener = MagicMock()
                opener.open.side_effect = urllib.error.HTTPError(
                    'https://example.invalid', status, 'error', {},
                    io.BytesIO(b'upstream echoed sensitive-key'))
                with patch('stream_train.pipeline.urllib.request.build_opener', return_value=opener):
                    with self.assertRaises(UnknownOutcome) as caught:
                        request_json('https://example.invalid', {'x': 1}, 'sensitive-key')
                self.assertEqual(opener.open.call_count, 1)
                self.assertEqual(str(caught.exception), 'http_%s' % status)
                self.assertEqual(caught.exception.receipt['http_statuses'], [status])
                self.assertNotIn('sensitive-key', caught.exception.receipt['raw_response'])

    def test_ambiguous_transport_error_is_not_retried(self):
        opener = MagicMock()
        opener.open.side_effect = TimeoutError('request may have reached server')
        with patch('stream_train.pipeline.urllib.request.build_opener', return_value=opener):
            with self.assertRaises(UnknownOutcome) as caught:
                request_json('https://example.invalid', {}, 'secret')
        self.assertEqual(opener.open.call_count, 1)
        self.assertEqual(str(caught.exception), 'TimeoutError')

    def test_unreadable_error_body_preserves_http_status(self):
        opener = MagicMock()
        error = urllib.error.HTTPError('https://example.invalid', 429, 'error', {}, None)
        error.read = MagicMock(side_effect=TimeoutError())
        opener.open.side_effect = error
        with patch('stream_train.pipeline.urllib.request.build_opener', return_value=opener):
            with self.assertRaises(UnknownOutcome) as caught:
                request_json('https://example.invalid', {}, 'secret')
        self.assertEqual(opener.open.call_count, 1)
        self.assertEqual(str(caught.exception), 'http_429')
        self.assertEqual(caught.exception.receipt['http_statuses'], [429])

    def test_valid_response_is_parsed(self):
        opener = MagicMock()
        response = opener.open.return_value.__enter__.return_value
        response.status = 200
        response.read.return_value = b'{"ok":true}'
        with patch('stream_train.pipeline.urllib.request.build_opener', return_value=opener):
            self.assertEqual(request_json('https://example.invalid', {}, 'secret'),
                             ({'ok': True}, [200]))
        self.assertEqual(opener.open.call_count, 1)

    def test_malformed_success_remains_unknown(self):
        opener = MagicMock()
        response = opener.open.return_value.__enter__.return_value
        response.status = 200
        response.read.return_value = b'not-json secret'
        with patch('stream_train.pipeline.urllib.request.build_opener', return_value=opener):
            with self.assertRaises(UnknownOutcome) as caught:
                request_json('https://example.invalid', {}, 'secret')
        self.assertEqual(opener.open.call_count, 1)
        self.assertEqual(str(caught.exception), 'invalid_json_response')
        self.assertNotIn('secret', caught.exception.receipt['raw_response'])

    def test_invalid_provider_shapes_are_classified(self):
        cases = [([], 'response_schema'),
                 ({'model': 'm', 'choices': [None]}, 'incomplete_response'),
                 ({'model': 'm', 'choices': [{'finish_reason': 'stop', 'message': None}]},
                  'missing_content')]
        for raw, expected in cases:
            with self.subTest(expected=expected):
                with patch('stream_train.pipeline.request_json', return_value=(raw, [200])):
                    with self.assertRaises(UnknownOutcome) as caught:
                        api_message('https://example.invalid/v1', 'secret', 'm', 'x', thinking=False)
                self.assertEqual(str(caught.exception), expected)


if __name__ == '__main__':
    unittest.main()
