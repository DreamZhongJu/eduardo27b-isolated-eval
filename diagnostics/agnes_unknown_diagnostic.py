#!/usr/bin/env python3
"""Read-only, aggregate-only diagnosis of final Agnes unknown receipts."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path('/workspace/results/own-test-499-v1')
MODEL = 'agnes-3.0-flash'
REASONS = {'retry_limit', 'undecidable_or_http_error', 'incomplete_previous_response'}
ERRORS = {'TimeoutError', 'URLError', 'SSLError', 'ConnectionError', 'ConnectionResetError',
          'ConnectionRefusedError', 'RemoteDisconnected', 'OSError', 'Interrupted'}


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, UnicodeError):
        return None


def code_label(data):
    if not isinstance(data, dict):
        return '回执无法读取'
    code = data.get('http_status')
    if type(code) is int:
        if code in (200, 401, 403, 408, 429):
            return f'HTTP_{code}'
        if 400 <= code <= 499:
            return 'HTTP_其他4xx'
        if 500 <= code <= 599:
            return 'HTTP_5xx'
        return 'HTTP_其他'
    error = data.get('error_type')
    if error in ERRORS:
        return error
    return '网络或状态未知'


def response_class(data):
    raw = data.get('raw_response') if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        return '响应缺失或非JSON', '无'
    if raw.get('model') != MODEL:
        return '模型标识不匹配', '未检查'
    choices = raw.get('choices')
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return '响应结构缺失', '无'
    choice = choices[0]
    finish = choice.get('finish_reason')
    finish_label = finish if finish in ('stop', 'length', 'content_filter', 'tool_calls') else '其他或缺失'
    if finish == 'length':
        return '判分输出截断', finish_label
    if finish != 'stop':
        return '判分未正常结束', finish_label
    message = choice.get('message')
    content = message.get('content') if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        return '判分正文为空', finish_label
    text = content.strip()
    if text.startswith('```') and text.endswith('```'):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].strip().lower() in ('```', '```json') and lines[-1].strip() == '```':
            text = '\n'.join(lines[1:-1]).strip()
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return '判分正文非JSON', finish_label
    if not isinstance(value, dict) or set(value) != {'correct'}:
        return 'JSON格式不符', finish_label
    if value['correct'] is None:
        return '判分模型返回null', finish_label
    if type(value['correct']) is bool:
        return '布尔结果但记录未决', finish_label
    return 'JSON格式不符', finish_label


def diagnose(root):
    outputs = {}
    for label, stages in (('简单', ('PILOT', 'EASY')), ('中等', ('MEDIUM',)), ('困难', ('HARD',))):
        reasons, latest, finishes, classes, attempts = Counter(), Counter(), Counter(), Counter(), Counter()
        unknown = missing = unreadable = 0
        for stage in stages:
            source_root = root / 'AGNES' / stage / 'sources'
            for result_path in source_root.glob('*/RESULTS.json'):
                doc = read_json(result_path)
                if not isinstance(doc, dict) or not isinstance(doc.get('results'), list):
                    unreadable += 1
                    continue
                source_key = result_path.parent.name
                for row in doc['results']:
                    if not isinstance(row, dict) or row.get('status') != 'unknown':
                        continue
                    unknown += 1
                    reason = row.get('reason')
                    reasons[reason if reason in REASONS else '其他'] += 1
                    identity = row.get('identity')
                    if not isinstance(identity, dict):
                        missing += 1
                        continue
                    # grade_one() addresses receipts by digest({'identity': ident}),
                    # not by the digest of ident alone. Preserve the grader's exact
                    # UTF-8 canonical form without importing or calling the grader.
                    blob = json.dumps({'identity': identity}, ensure_ascii=False,
                                      sort_keys=True, separators=(',', ':')).encode('utf-8')
                    request_key = hashlib.sha256(blob).hexdigest()
                    folder = root / 'AGNES' / 'receipts' / stage / source_key / request_key
                    files = sorted(folder.glob('ATTEMPT-*-RESPONSE.json'))
                    if not files:
                        missing += 1
                        continue
                    records = [read_json(p) for p in files]
                    for record in records:
                        attempts[code_label(record)] += 1
                    final = records[-1]
                    status = code_label(final)
                    latest[status] += 1
                    if status == 'HTTP_200':
                        cls, fin = response_class(final)
                        classes[cls] += 1
                        finishes[fin] += 1
        outputs[label] = {'未决作答数': unknown, '未决原因': dict(reasons), '最后一次请求': dict(latest),
                          'HTTP200终止类型': dict(finishes), 'HTTP200响应类别': dict(classes),
                          '未决样本全部尝试次数': dict(attempts), '缺少尝试回执的未决样本': missing,
                          '无法读取的来源结果文件数': unreadable}
    return outputs


if __name__ == '__main__':
    try:
        print(json.dumps(diagnose(ROOT), ensure_ascii=False, sort_keys=True))
    except Exception:
        print(json.dumps({'状态': '汇总失败'}, ensure_ascii=False))
        sys.exit(1)
