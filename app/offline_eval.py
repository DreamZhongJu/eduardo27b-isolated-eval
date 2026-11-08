"""Fresh, locally journaled author-transfer evaluation; deliberately no online judge."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import openai
from dialogue import make_dialogue
from scoring import put_once, read, sha, source_dir, stable
from protocol import _SOLVE_PROMPT
from protocol import _build_final_attempt_messages
from protocol import _row

MODEL = 'Llama-3.1-8B-Instruct'
TEACHER = 'Eduardo-27b'


def local_url(value):
    p = urlparse(value)
    if p.scheme != 'http' or p.hostname not in ('127.0.0.1', 'localhost') or p.username or p.password:
        raise ValueError('model endpoints must be loopback HTTP')
    return value.rstrip('/')


def seed(source, target, phase, index):
    data = f'offline-author-v1|{source}|{target}|{phase}|{index}'.encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:4], 'big') & 0x7fffffff


def parse_sources(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows or len({x['source_id'] for x in rows}) != len(rows):
        raise ValueError('source file is empty or has duplicated identifiers')
    for row in rows:
        for key in ('source_id', 'problem', 'answer', 'transfer_problem', 'transfer_answer'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f'missing or empty source field: {key}')
    return rows


async def student_sample(client, source, target, phase, index, messages, semaphore):
    folder = source_dir(source) / 'samples' / target / phase / f'{index:02d}'
    params = {'model': MODEL, 'messages': messages, 'temperature': 0.6, 'max_tokens': 1024,
              'seed': seed(source['source_id'], target, phase, index),
              'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}}
    binding = {'source_id': source['source_id'], 'target': target, 'phase': phase, 'index': index,
               'request_sha256': sha(stable(params))}
    intent, response = folder / 'INTENT.json', folder / 'RESPONSE.json'
    async with semaphore:
        if intent.exists():
            if any(read(intent).get(k) != v for k, v in binding.items()):
                raise ValueError('saved student request differs')
            if response.exists():
                return read(response)
            raise RuntimeError('unresolved previous student request; do not replay automatically')
        put_once(intent, binding)
        try:
            result = await client.chat.completions.create(**params)
            choice = result.choices[0]
            if choice.finish_reason not in ('stop', 'length') or not isinstance(choice.message.content, str):
                raise ValueError('invalid student response')
            receipt = {'identity': binding, 'text': choice.message.content,
                       'finish_reason': choice.finish_reason, 'returned_model': result.model}
            put_once(response, receipt)
            return receipt
        except Exception as exc:
            put_once(folder / 'ERROR.json', {'identity': binding, 'status': 'unknown',
                                             'error_type': type(exc).__name__})
            raise


async def run_one(row, teacher_path, teacher_url, student_url, student_client):
    source = {'source_id': row['source_id'], 'problem': row['problem'], 'answer': row['answer']}
    folder = source_dir(source)
    folder.mkdir(parents=True, exist_ok=True)
    source_hash = sha(stable(row))
    if (folder / 'SOURCE.json').exists():
        if read(folder / 'SOURCE.json')['sha256'] != source_hash:
            raise ValueError('source identity changed')
    else:
        put_once(folder / 'SOURCE.json', {'sha256': source_hash, 'source_id': row['source_id']})
    sem = asyncio.Semaphore(4)
    targets = [('source', row['problem'], row['answer']),
               ('paper', row['transfer_problem'], row['transfer_answer'])]
    collected = []
    for name, problem, answer in targets:
        prompt = [{'role': 'user', 'content': _SOLVE_PROMPT.format(problem=problem)}]
        for phase in ('pre', 'control'):
            samples = await asyncio.gather(*(student_sample(student_client, source, name, phase, i, prompt, sem)
                                             for i in range(16)))
            collected.extend((name, phase, problem, answer, i, sample) for i, sample in enumerate(samples))
    official_row = _row(row['problem'], row['answer'], 0.0, row['transfer_problem'],
                        row['transfer_answer'], 0.0, 'test')
    args = SimpleNamespace(model=TEACHER, teacher_tokenizer_path=teacher_path,
        teacher_api_key_env='LOCAL_ONLY_KEY', teacher_api_base=teacher_url,
        student_api_key_env='LOCAL_ONLY_KEY', student_api_base=student_url,
        student_model=MODEL, student_concurrency=1, teacher_temperature=0.6, teacher_top_p=0.95,
        student_temperature=0.6, max_tokens_per_turn=4192, student_max_tokens=1024,
        max_conversation_tokens=16384, max_teacher_turns=11, min_teacher_turns_before_eoc=3,
        enable_thinking=True, num_samples_per_problem=1, assume_thinking='auto', seed=42, num_retries=0)
    dialogue = await make_dialogue(source, official_row, args)
    for name, problem, answer in targets:
        messages = _build_final_attempt_messages(dialogue['reward_messages'], source['problem'],
            transfer_problem=problem if name == 'paper' else None, omit_teacher_turns=True)
        samples = await asyncio.gather(*(student_sample(student_client, source, name, 'post', i, messages, sem)
                                         for i in range(16)))
        collected.extend((name, 'post', problem, answer, i, sample) for i, sample in enumerate(samples))
    if len(collected) != 96:
        raise ValueError('wrong number of samples')
    # Local material for later offline scoring. Correctness is never guessed.
    score_input = [{'source_id': row['source_id'], 'target': target, 'phase': phase, 'index': index,
                    'problem': problem, 'reference_answer': answer, 'student_response': sample['text'],
                    'finish_reason': sample['finish_reason'], 'grading_status': 'ungraded'}
                   for target, phase, problem, answer, index, sample in collected]
    if (folder / 'SCORE_INPUT.json').exists():
        if read(folder / 'SCORE_INPUT.json') != score_input:
            raise ValueError('score input changed')
    else:
        put_once(folder / 'SCORE_INPUT.json', score_input)
    if not (folder / 'RAW_COMPLETE.json').exists():
        put_once(folder / 'RAW_COMPLETE.json', {'source_id': row['source_id'], 'samples': 96,
                 'grading_status': 'ungraded', 'source_sha256': source_hash})


async def main(args):
    os.environ['LOCAL_ONLY_KEY'] = 'loopback-only'
    rows = parse_sources(args.sources)
    student = openai.AsyncClient(api_key='loopback-only', base_url=local_url(args.student_url),
                                 timeout=300, max_retries=0)
    try:
        for row in rows:
            try:
                await run_one(row, args.teacher_model, local_url(args.teacher_url), local_url(args.student_url), student)
            except Exception as exc:
                # Per-source error retains all existing evidence; never replay an unknown request.
                folder = source_dir(row)
                failure = folder / 'SOURCE_HELD.json'
                if not failure.exists():
                    put_once(failure, {'source_id': row['source_id'], 'status': 'held', 'error_type': type(exc).__name__})
    finally:
        await student.close()


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('invalid arguments')


if __name__ == '__main__':
    try:
        p = SafeParser(add_help=False, prog='LOCAL_EVAL')
        p.add_argument('--sources', required=True)
        p.add_argument('--teacher-model', required=True)
        p.add_argument('--teacher-url', default='http://127.0.0.1:8001/v1')
        p.add_argument('--student-url', default='http://127.0.0.1:8002/v1')
        args = p.parse_args()
        local_url(args.teacher_url); local_url(args.student_url)
        asyncio.run(main(args))
    except Exception:
        print(json.dumps({'state': 'ERROR', 'code': 'LOCAL_EVAL_FAILED'}))
        raise SystemExit(1)
