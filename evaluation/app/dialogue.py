#!/usr/bin/env python3
"""Official Eduardo rollout with local vLLM API calls journaled per turn.

Import only: the resource supervisor must independently provide isolated local
teacher and student servers and the appropriate GPU allocation before calling this.
"""
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

import openai

OFFICIAL = Path(__file__).resolve().parent.parent / 'official'
sys.path.insert(0, str(OFFICIAL))
import protocol as official  # noqa: E402
from protocol import thinking_text  # noqa: E402
from scoring import put_once, read, sha, stable, source_dir  # noqa: E402


async def once(folder, client, binding, **kwargs):
    intent, result = folder / 'INTENT.json', folder / 'RESULT.json'
    if intent.exists():
        raise RuntimeError('existing dialogue request cannot be replayed')
    put_once(intent, {**binding, 'request_sha256': sha(stable(kwargs)), 'time': time.time()})
    try:
        resp = await client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        content = msg.content or ''
        reasoning = getattr(msg, 'reasoning_content', None) or getattr(msg, 'reasoning', None)
        if reasoning and '</think>' not in content:
            content = f'<think>{reasoning}</think>{content}'
        receipt = {'status': 'completed', 'content': content, 'finish_reason': choice.finish_reason,
                   'model': getattr(resp, 'model', None), 'id': getattr(resp, 'id', None),
                   'time': time.time()}
    except Exception as exc:
        put_once(result, {'status': 'unknown', 'error_type': type(exc).__name__, 'time': time.time()})
        raise
    put_once(result, receipt)
    return receipt['content']


class LocalAPITeacher:
    def __init__(self, args, source):
        from transformers import AutoTokenizer
        endpoint = urlparse(args.teacher_api_base)
        if (endpoint.scheme != 'http' or endpoint.hostname not in ('localhost', '127.0.0.1')
                or endpoint.username or endpoint.password):
            raise ValueError('teacher endpoint must be loopback')
        self.args, self.source = args, source
        self.tokenizer = AutoTokenizer.from_pretrained(args.teacher_tokenizer_path,
                                                       trust_remote_code=True,
                                                       local_files_only=True)
        self.client = openai.AsyncClient(api_key=os.environ[args.teacher_api_key_env],
                                          base_url=args.teacher_api_base, timeout=1800, max_retries=0)

    async def generate(self, conversations):
        if len(conversations) != 1:
            raise ValueError('pilot supports exactly one dialogue')
        conv = conversations[0]
        turn = conv.num_teacher_turns
        folder = source_dir(self.source) / 'dialogue' / 'teacher' / f'{turn:02d}'
        text = await once(folder, self.client, {'role': 'teacher', 'turn': turn},
                          model=self.args.model, messages=conv.teacher_view(),
                          temperature=self.args.teacher_temperature,
                          top_p=self.args.teacher_top_p, max_tokens=self.args.max_tokens_per_turn,
                          extra_body={'chat_template_kwargs': {'enable_thinking': self.args.enable_thinking}})
        return [text.strip()]

    def count_tokens(self, conv):
        return sum(len(self.tokenizer.encode(m['content'])) for m in conv.messages)

    def count_text_tokens(self, text):
        return len(self.tokenizer.encode(text)) if text else 0

    def thinking_tokens(self, index, text):
        return self.count_text_tokens(thinking_text(text)), 'tokenizer'

    def close(self):
        pass


async def student_turn(source, conv, client, args, semaphore):
    turn = conv.num_teacher_turns
    folder = source_dir(source) / 'dialogue' / 'student' / f'{turn:02d}'
    try:
        async with semaphore:
            text = await once(folder, client, {'role': 'student', 'turn': turn},
                              model=args.student_model, messages=conv.student_view(),
                              temperature=args.student_temperature, max_tokens=args.student_max_tokens,
                              extra_body={'chat_template_kwargs': {'enable_thinking': False}})
        conv.messages.append({'role': 'student', 'content': text.strip()})
    except Exception:
        conv.finish('student_failed')


async def make_dialogue(source, row, args):
    """Run exactly one independent dialogue, holding a partial rollout on error."""
    if row['extra_info']['problem'] != source['problem'] or str(row['extra_info']['answer']) != source['answer']:
        raise ValueError('official evaluation row/source mismatch')
    folder = source_dir(source) / 'dialogue'
    intent, result = folder / 'ROLLOUT_INTENT.json', folder / 'ROLLOUT_RESULT.json'
    binding = {'source_id': source['source_id'], 'source_sha256': sha(stable(source)),
               'teacher_model': args.model, 'student_model': args.student_model}
    if intent.exists():
        previous = read(intent)
        if any(previous.get(k) != v for k, v in binding.items()):
            raise ValueError('prior rollout binding differs')
        if result.exists() and read(result).get('stage') == 'completed':
            return read(result)
        raise RuntimeError('existing partial/unknown rollout must not be replayed')
    put_once(intent, {**binding, 'time': time.time()})
    old_teacher, old_student = official.make_teacher, official._student_turn
    official.make_teacher = lambda a: LocalAPITeacher(a, source)
    official._student_turn = lambda conv, client, a, sem: student_turn(source, conv, client, a, sem)
    try:
        conv = (await official.transfer_rollout(args, [row]))[0]
        rec = conv.to_record()
        rec.update(source_id=source['source_id'], reward_messages=conv.reward_view(),
                   judge_messages=conv.judge_view())
        rec['stage'] = 'completed' if conv.end_reason not in ('teacher_failed', 'student_failed') else 'unknown'
        put_once(result, rec)
        if rec['stage'] != 'completed':
            raise RuntimeError('teacher/student dialogue failed; receipt retained')
        return rec
    finally:
        official.make_teacher, official._student_turn = old_teacher, old_student
