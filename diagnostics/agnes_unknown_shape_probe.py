#!/usr/bin/env python3
"""Read-only aggregate classification of already stored Agnes unknown responses.

Only fixed labels and counts are returned. No API request, grading decision,
source text, response text, source identifier, path, or arbitrary key is emitted.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path('/workspace/results/own-test-499-v1')
MODEL = 'agnes-3.0-flash'
EXPLANATION_KEYS = frozenset(('reason', 'reasoning', 'explanation', 'rationale', 'analysis'))
DECISION_KEYS = frozenset(('is_correct', 'verdict', 'judgment', 'judgement', 'answer',
                           'final_answer', 'score', 'label', 'grade', 'result'))


class DuplicateJSONKey(ValueError):
    """Reject duplicate keys at every object nesting level without exposing them."""


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise DuplicateJSONKey('duplicate key')
        obj[key] = value
    return obj


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, UnicodeError):
        return None


def response_shape(record):
    """Return exactly one fixed label and an empty-content reasoning flag."""
    if not isinstance(record, dict):
        return '回执无法读取', None
    if record.get('http_status') != 200:
        return '非HTTP200或网络异常', None
    raw = record.get('raw_response')
    if not isinstance(raw, dict):
        return '响应缺失或非JSON', None
    if raw.get('model') != MODEL:
        return '模型标识不匹配', None
    choices = raw.get('choices')
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return '响应结构缺失', None
    choice = choices[0]
    finish = choice.get('finish_reason')
    if finish == 'length':
        return '判分输出截断', None
    if finish != 'stop':
        return '判分未正常结束', None
    message = choice.get('message')
    if not isinstance(message, dict):
        return '响应结构缺失', None
    content = message.get('content')
    if not isinstance(content, str) or not content.strip():
        reasoning = message.get('reasoning_content')
        if not isinstance(reasoning, str) or not reasoning.strip():
            reasoning = message.get('reasoning')
        return '判分正文为空', bool(isinstance(reasoning, str) and reasoning.strip())
    text = content.strip()
    if text.startswith('```') and text.endswith('```'):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].strip().lower() in ('```', '```json') and lines[-1].strip() == '```':
            text = '\n'.join(lines[1:-1]).strip()
    try:
        value = json.loads(text, object_pairs_hook=unique_object)
    except DuplicateJSONKey:
        return 'JSON含重复键', None
    except (ValueError, TypeError):
        token = text.strip().lower()
        if token in ('yes', 'no', 'correct', 'incorrect'):
            return '非JSON精确yes_no_correct_incorrect候选', None
        return '非JSON其他', None
    if type(value) is bool:
        return 'JSON顶层布尔候选', None
    if value is None:
        return 'JSON顶层null', None
    if not isinstance(value, dict):
        return '其他JSON结构', None
    keys = set(value)
    if 'correct' in value:
        item = value['correct']
        rest = keys - {'correct'}
        if type(item) is bool:
            if not rest:
                return '标准布尔格式但记录未决', None
            if rest <= EXPLANATION_KEYS:
                return 'correct布尔与说明字段候选', None
            if rest & DECISION_KEYS:
                return 'correct布尔与其他判定字段', None
            return 'correct布尔与其他字段', None
        if item is None:
            return 'correct为null', None
        if isinstance(item, str) and item.strip().lower() in ('true', 'false'):
            if not rest or rest <= EXPLANATION_KEYS:
                return 'correct字符串true_false候选', None
            return 'correct字符串与其他字段', None
        return 'correct其他值类型', None
    if type(value.get('is_correct')) is bool:
        rest = keys - {'is_correct'}
        if not rest or rest <= EXPLANATION_KEYS:
            return 'is_correct布尔候选', None
        return 'is_correct布尔与其他字段', None
    return '其他JSON对象', None


def diagnose(root):
    outputs = {}
    for label, stages in (('简单', ('PILOT', 'EASY')), ('中等', ('MEDIUM',)), ('困难', ('HARD',))):
        counts = Counter()
        unknown = empty_reasoning = empty_no_reasoning = unreadable = 0
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
                    identity = row.get('identity')
                    if not isinstance(identity, dict):
                        counts['身份字段缺失'] += 1
                        continue
                    raw = json.dumps({'identity': identity}, ensure_ascii=False,
                                     sort_keys=True, separators=(',', ':')).encode('utf-8')
                    request_key = hashlib.sha256(raw).hexdigest()
                    folder = root / 'AGNES' / 'receipts' / stage / source_key / request_key
                    files = sorted(folder.glob('ATTEMPT-*-RESPONSE.json'))
                    if not files:
                        counts['缺少尝试回执'] += 1
                        continue
                    label_out, reasoning = response_shape(read_json(files[-1]))
                    counts[label_out] += 1
                    if reasoning is True:
                        empty_reasoning += 1
                    elif reasoning is False:
                        empty_no_reasoning += 1
        outputs[label] = {'未决作答数': unknown, '结构分类': dict(sorted(counts.items())),
                          '空正文中有推理字段': empty_reasoning, '空正文中无推理字段': empty_no_reasoning,
                          '无法读取的来源结果文件数': unreadable}
    return outputs


if __name__ == '__main__':
    try:
        print(json.dumps(diagnose(ROOT), ensure_ascii=False, sort_keys=True))
    except Exception:
        print(json.dumps({'状态': '汇总失败'}, ensure_ascii=False))
        sys.exit(1)
