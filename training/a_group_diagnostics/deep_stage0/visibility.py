#!/usr/bin/env python3
"""Read-only aggregate inspection of teacher visibility in smoke_shared3."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

ROOT = Path('/workspace/A_group_repro_20261117')
WORK = ROOT / 'pilots/smoke_shared3'
SOURCE = ROOT / 'source/EduardoRL'
MODEL = ROOT / 'models/Qwen3.5-4B'
COMMIT = 'e8c89b7293e65e632149571b16a1c2d4e33c77fe'
TOKENIZER_CONFIG_BLOB = 'eda48d3e75a8e59a8479ee4ec8b37f76e711d9c1'


def table(title, rows):
    print('\n' + title)
    print('| 项目 | 数量 |')
    print('|------|-----:|')
    for name, n in rows:
        print(f'| {name} | {n} |')


def author_strip():
    result = subprocess.run(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'],
                            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, timeout=20)
    if result.stdout.strip() != COMMIT:
        raise ValueError('源码版本不符')
    dirty = subprocess.run(['git', '-C', str(SOURCE), 'status', '--porcelain', '--untracked-files=no'],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           text=True, timeout=20)
    if dirty.stdout.strip():
        raise ValueError('源码已修改')
    sys.path.insert(0, str(SOURCE))
    from eduardo.utils import strip_teacher_controls
    return strip_teacher_controls


def config_blob():
    raw = (MODEL / 'tokenizer_config.json').read_bytes()
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


def load_tokenizer():
    try:
        if config_blob() != TOKENIZER_CONFIG_BLOB:
            return None, 'tokenizer配置与冻结版本不符'
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
        return tokenizer, '加载成功'
    except Exception as exc:
        return None, '加载失败（' + type(exc).__name__ + '）'


def main():
    try:
        strip = author_strip()
    except Exception as exc:
        print('作者清理函数不可用：', type(exc).__name__)
        return 1
    tokenizer, tokenizer_status = load_tokenizer()
    print('本地冻结 tokenizer：', tokenizer_status)
    if tokenizer is not None:
        try:
            ids = tokenizer.encode('<think>x</think>y', add_special_tokens=False)
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            print('合成解码保留 <think>：', '是' if '<think>' in decoded else '否')
            print('合成解码保留 </think>：', '是' if '</think>' in decoded else '否')
            print('两个标签被列为 special token：',
                  '是' if any(tag in tokenizer.all_special_tokens for tag in ('<think>', '</think>')) else '否')
            prompt = tokenizer.apply_chat_template([{'role':'user','content':'2+3?'}],
                       tokenize=False, add_generation_prompt=True, enable_thinking=True)
            print('合成开启思考提示以 <think> 开头：', '是' if str(prompt).rstrip().endswith('<think>') else '否')
        except Exception as exc:
            print('合成模板/解码检查：失败（' + type(exc).__name__ + '）')

    files = sorted((WORK / 'eduardo_conversations').glob('conversations_*.jsonl'))
    seen, counts = set(), Counter()
    for path in files:
        try:
            f = path.open('r', encoding='utf-8')
        except OSError:
            counts['文件无法读取'] += 1
            continue
        with f:
            for line in f:
                try:
                    row = json.loads(line)
                except (ValueError, TypeError):
                    counts['无效行'] += 1
                    continue
                if not isinstance(row, dict):
                    counts['无效行'] += 1
                    continue
                uid = row.get('uuid')
                if not isinstance(uid, str) or not uid:
                    counts['UUID缺失'] += 1
                    continue
                if uid in seen:
                    counts['重复UUID'] += 1
                    continue
                seen.add(uid)
                messages = row.get('messages')
                if not isinstance(messages, list):
                    counts['消息字段缺失'] += 1
                    continue
                teachers = [m for m in messages if isinstance(m, dict) and m.get('role') == 'assistant']
                if not teachers:
                    counts['无教师回合'] += 1
                    continue
                texts = [m.get('content') if isinstance(m.get('content'), str) else '' for m in teachers]
                if not any(x.strip() for x in texts):
                    counts['教师原文全空'] += 1
                    continue
                if any(strip(x).strip() for x in texts):
                    counts['有可见教师发言'] += 1
                    continue
                counts['原始非空但可见全空'] += 1
                gated = row.get('gated')
                if gated in (1, True): counts['可见全空且门槛拦截'] += 1
                elif gated in (0, False): counts['可见全空且门槛放行'] += 1
                else: counts['可见全空且门槛未知'] += 1
                tags = []
                for text in texts:
                    if not text.strip():
                        tag = '原文空'
                    elif '<think>' not in text and '</think>' not in text:
                        tag = '无任何思考标签'
                    elif '<think>' in text and '</think>' not in text:
                        tag = '含开标签但无闭标签'
                    elif '</think>' in text:
                        tag = '含闭标签'
                    else:
                        tag = '其他标签情况'
                    tags.append(tag)
                    counts['回合：' + tag] += 1
                    if tokenizer is not None and text.strip():
                        try:
                            n = len(tokenizer.encode(text, add_special_tokens=False))
                            for threshold in (630,650,660,662):
                                if n >= threshold:
                                    counts[f'回合：近似长度≥{threshold}'] += 1
                                    if tag == '无任何思考标签':
                                        counts[f'无标签回合：近似长度≥{threshold}'] += 1
                        except Exception:
                            counts['回合：重新分词失败'] += 1
                for tag in set(tags): counts['轨迹含：' + tag] += 1
                if tags and all(t == '无任何思考标签' for t in tags): counts['轨迹：所有教师回合均无标签'] += 1
                if not any('</think>' in x for x in texts): counts['轨迹：全部回合无闭标签'] += 1
                if 'finish_reason' in row or 'stop_reason' in row: counts['轨迹记录含停止原因键'] += 1
                counts['教师回合记录含停止原因键'] += sum(
                    'finish_reason' in m or 'stop_reason' in m for m in teachers)
    print('\n本次只读检查的对话文件数：', len(files))
    print('可解析并去重的记录数：', len(seen))
    table('教师可见性和门槛（轨迹数）', [(k, counts[k]) for k in (
      '原始非空但可见全空','有可见教师发言','无教师回合','教师原文全空',
      '可见全空且门槛拦截','可见全空且门槛放行','可见全空且门槛未知')])
    table('可见全空轨迹的标签分布（可重叠）', [(k, counts[k]) for k in (
      '轨迹含：无任何思考标签','轨迹含：含开标签但无闭标签','轨迹含：含闭标签',
      '轨迹：所有教师回合均无标签','轨迹：全部回合无闭标签')])
    table('可见全空轨迹中的教师回合', [(k, counts[k]) for k in (
      '回合：原文空','回合：无任何思考标签','回合：含开标签但无闭标签','回合：含闭标签',
      '回合：近似长度≥630','回合：近似长度≥650','回合：近似长度≥660','回合：近似长度≥662',
      '无标签回合：近似长度≥630','无标签回合：近似长度≥650',
      '无标签回合：近似长度≥660','无标签回合：近似长度≥662','回合：重新分词失败')])
    table('停止信息与文件完整性', [(k, counts[k]) for k in (
      '轨迹记录含停止原因键','教师回合记录含停止原因键','UUID缺失','重复UUID','无效行',
      '消息字段缺失','文件无法读取')])
    if tokenizer is None:
        print('注意：tokenizer 不可用；所有近似长度计数为未计算，不应当理解为零。')
    print('说明：单轮配置上限为 150+512=662 token，剩余上下文还可能降低实际允许长度。')
    print('重新分词只是近似，不等同生成时的 token 数或停止原因；低于662不能排除被截断。')
    print('无标签可能是未闭合思考，也可能是其他格式；这些汇总不能单独证明哪种情况。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
