"""T-017 wrap-up helper: Step 1 inventory, Step 2 HF export verification, Step 3 guarded checkpoint cleanup."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pro5000_config as pcfg
from pro5000_trainer_entry import write_json
from stage2_pilot import default_base
from stage2_precheck import snapshot_gpus

STEP_DIR_RE = re.compile(r'^global_step_[0-9]+$')
FORMAL_RUN_DEFAULT = 'stage3_formal_20261121_144311'


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dir_size_bytes(path: Path) -> int:
    if path.is_symlink():
        return 0
    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        rpath = Path(root)
        for name in (*dirs, *files):
            p = rpath / name
            if p.is_symlink():
                continue
            try:
                total += p.lstat().st_size
            except OSError:
                pass
    return total


def format_bytes(n: int) -> str:
    if n >= 10**9:
        return f'{n / 1e9:.2f} GB'
    if n >= 1024**2:
        return f'{n / 1024**2:.1f} MiB'
    if n >= 1024:
        return f'{n / 1024:.1f} KiB'
    return f'{n} B'


def pick_idle_gpu() -> int:
    first = snapshot_gpus()
    time.sleep(1)
    second = snapshot_gpus()
    for idx in (0, 1, 2, 3, 4, 5, 7, 6):
        a, b = first.get(idx), second.get(idx)
        if not a or not b:
            continue
        if a['apps'] or b['apps'] or a['used_mib'] >= 500 or b['used_mib'] >= 500:
            continue
        if idx != 6 and (a['util'] >= 10 or b['util'] >= 10):
            continue
        return idx
    raise RuntimeError('没有可用空闲 GPU（已有外部进程或显存占用）')


def run_inventory(base: Path) -> int:
    base = base.resolve()
    runs_dir = base / 'runs'
    print('=== DF /data ===')
    du_root = Path('/data') if Path('/data').exists() else base
    du = shutil.disk_usage(du_root)
    print(f'{du_root}: free={du.free / 1e9:.2f} GB ({du.free / 1024**3:.2f} GiB) total={du.total / 1e9:.2f} GB')

    print('=== BASE_SUBDIRS ===')
    for entry in sorted(base.iterdir()):
        if entry.is_symlink():
            print(f'  symlink {entry} -> {os.readlink(entry)}')
        elif entry.is_dir():
            print(f'  {format_bytes(dir_size_bytes(entry)):>10}  {entry}')
        else:
            print(f'  {format_bytes(entry.lstat().st_size):>10}  {entry}')

    print('=== RUNS_CKPTS_AND_EXPORTS ===')
    if runs_dir.is_dir():
        found = []
        for root, dirs, _ in os.walk(runs_dir, followlinks=False):
            rpath = Path(root)
            for d in sorted(dirs):
                p = rpath / d
                if STEP_DIR_RE.fullmatch(d) or d == 'exports':
                    found.append(p)
        for p in sorted(found):
            tag = 'SYMLINK!' if p.is_symlink() else format_bytes(dir_size_bytes(p))
            print(f'  {tag:>10}  {p}')
            if p.name == 'exports' and p.is_dir() and not p.is_symlink():
                for child in sorted(p.iterdir()):
                    if child.is_dir():
                        print(f'    {format_bytes(dir_size_bytes(child)):>10}  {child}')

    print('=== SHM ===')
    shm = Path('/dev/shm/zhujie_a_group')
    if shm.is_dir():
        print(f'  {format_bytes(dir_size_bytes(shm)):>10}  {shm}')
        for sub in sorted(shm.iterdir()):
            if sub.is_dir():
                print(f'    {format_bytes(dir_size_bytes(sub)):>10}  {sub}')
    else:
        print('  /dev/shm/zhujie_a_group 不存在')
    return 0


def compare_tokenizer_and_template(export_dir: Path, base_dir: Path) -> dict:
    from transformers import AutoTokenizer

    def _norm_json(p: Path):
        return json.loads(p.read_text(encoding='utf-8'))

    tok_files_equal = True
    file_details = {}
    for fname in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        ep, bp = export_dir / fname, base_dir / fname
        if ep.is_file() != bp.is_file():
            tok_files_equal = False
            file_details[fname] = '存在性不同'
            continue
        if not ep.is_file():
            file_details[fname] = '均不存在'
            continue
        if ep.read_bytes() == bp.read_bytes():
            file_details[fname] = '字节一致'
        elif fname.endswith('.json') and _norm_json(ep) == _norm_json(bp):
            file_details[fname] = 'JSON语义一致'
        elif fname == 'chat_template.jinja' and ep.read_text(encoding='utf-8').strip() == bp.read_text(encoding='utf-8').strip():
            file_details[fname] = '模板文本一致'
        else:
            tok_files_equal = False
            file_details[fname] = '不同'

    tok_exp = AutoTokenizer.from_pretrained(str(export_dir), local_files_only=True)
    tok_base = AutoTokenizer.from_pretrained(str(base_dir), local_files_only=True)
    tpl_equal = (getattr(tok_exp, 'chat_template', None) == getattr(tok_base, 'chat_template', None))

    sample_msgs = [
        {'role': 'system', 'content': 'Be concise.'},
        {'role': 'user', 'content': 'Compute 12 + 25.'},
    ]
    ids_exp = tok_exp.apply_chat_template(
        sample_msgs, tokenize=True, add_generation_prompt=True, enable_thinking=True
    )
    ids_base = tok_base.apply_chat_template(
        sample_msgs, tokenize=True, add_generation_prompt=True, enable_thinking=True
    )
    ids_exp_nothink = tok_exp.apply_chat_template(
        sample_msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    ids_base_nothink = tok_base.apply_chat_template(
        sample_msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    vocab_equal = (tok_exp.get_vocab() == tok_base.get_vocab())
    render_ids_equal = (ids_exp == ids_base) and (ids_exp_nothink == ids_base_nothink)
    eos_equal = (tok_exp.eos_token_id == tok_base.eos_token_id)
    pad_equal = (tok_exp.pad_token_id == tok_base.pad_token_id)
    special_ids_equal = (tok_exp.all_special_ids == tok_base.all_special_ids)
    added_dec_equal = (
        {k: v.content for k, v in tok_exp.added_tokens_decoder.items()}
        == {k: v.content for k, v in tok_base.added_tokens_decoder.items()}
    )
    probe = '<think>x+1=2</think>\n\\boxed{1}<|im_end|>'
    encode_equal = (tok_exp.encode(probe) == tok_base.encode(probe))
    decode_equal = (
        tok_exp.decode(tok_exp.encode(probe), skip_special_tokens=True)
        == tok_base.decode(tok_base.encode(probe), skip_special_tokens=True)
        and tok_exp.decode(tok_exp.encode(probe), skip_special_tokens=False)
        == tok_base.decode(tok_base.encode(probe), skip_special_tokens=False)
    )

    chat_tpl_same = tpl_equal and render_ids_equal
    tok_functional_same = (
        vocab_equal
        and render_ids_equal
        and eos_equal
        and pad_equal
        and special_ids_equal
        and added_dec_equal
        and encode_equal
        and decode_equal
    )
    return {
        'chat_template': '一样' if chat_tpl_same else '不一样',
        'tokenizer_files': '一样' if tok_files_equal else '不一样',
        'tokenizer_behavior': '一样' if tok_functional_same else '不一样',
        'chat_template_same': chat_tpl_same,
        'tokenizer_files_same': tok_files_equal,
        'tokenizer_same': tok_functional_same,
        'render_ids_equal': render_ids_equal,
        'vocab_equal': vocab_equal,
        'eos_equal': eos_equal,
        'pad_equal': pad_equal,
        'special_ids_equal': special_ids_equal,
        'added_dec_equal': added_dec_equal,
        'encode_equal': encode_equal,
        'decode_equal': decode_equal,
        'file_details': file_details,
    }


def compute_export_sha256(export_dir: Path) -> dict[str, str]:
    if not export_dir.is_dir():
        raise RuntimeError(f'找不到导出目录: {export_dir}')
    hashes: dict[str, str] = {}
    for p in sorted(export_dir.rglob('*')):
        if p.is_symlink():
            raise RuntimeError(f'导出目录不应包含软链接: {p}')
        if p.is_file():
            rel = str(p.relative_to(export_dir))
            hashes[rel] = sha256_file(p)
    if 'config.json' not in hashes or not any(k.endswith('.safetensors') for k in hashes):
        raise RuntimeError(f'导出目录缺少 config.json 或 *.safetensors: {export_dir}')
    return hashes


def verify_vllm_single_gpu(export_dir: Path, gpu_id: int) -> dict:
    script = (
        'import json, sys\n'
        'from transformers import AutoTokenizer\n'
        'from vllm import LLM, SamplingParams\n'
        'mdir = sys.argv[1]\n'
        'tok = AutoTokenizer.from_pretrained(mdir, local_files_only=True)\n'
        'prompt = tok.apply_chat_template([{"role":"user","content":"Compute 7 * 8 briefly and end with \\\\boxed{56}."}], tokenize=False, add_generation_prompt=True, enable_thinking=True)\n'
        'llm = LLM(model=mdir, tensor_parallel_size=1, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.58, trust_remote_code=False)\n'
        'out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=80), use_tqdm=False)[0].outputs[0]\n'
        'print("VLLM_SMOKE_JSON:" + json.dumps({"loaded": True, "generated_tokens": len(out.token_ids), "has_close_think": "</think>" in out.text}))\n'
    )
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu_id),
        CUDA_DEVICE_ORDER='PCI_BUS_ID',
        HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',
        VLLM_NO_USAGE_STATS='1',
        DO_NOT_TRACK='1',
    )
    proc = subprocess.run(
        [sys.executable, '-B', '-c', script, str(export_dir)],
        env=env, capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f'vLLM 单卡加载失败 rc={proc.returncode}: {proc.stderr[-600:]}')
    for ln in proc.stdout.splitlines():
        if ln.startswith('VLLM_SMOKE_JSON:'):
            return json.loads(ln[len('VLLM_SMOKE_JSON:'):])
    raise RuntimeError('vLLM 单卡加载未返回 VLLM_SMOKE_JSON')


def run_verify_export(base: Path, run_dir: Path, base_model: Path, gpu_id: int | None) -> int:
    base = base.resolve()
    run_dir = run_dir.resolve()
    export_dir = run_dir / 'exports' / 'step_100'

    hashes = compute_export_sha256(export_dir)
    sha_txt = run_dir / 'export_step100_sha256.txt'
    sha_txt.write_text(''.join(f'{h}  {rel}\n' for rel, h in sorted(hashes.items())), encoding='utf-8')

    if not (base_model / 'config.json').is_file():
        raise RuntimeError(f'找不到 base 模型: {base_model}')
    cmp_res = compare_tokenizer_and_template(export_dir, base_model)

    chosen_gpu = pick_idle_gpu() if gpu_id is None else gpu_id
    vllm_res = verify_vllm_single_gpu(export_dir, chosen_gpu)

    passed = (
        bool(vllm_res.get('loaded'))
        and int(vllm_res.get('generated_tokens', 0)) > 0
        and bool(cmp_res.get('chat_template_same'))
        and bool(cmp_res.get('tokenizer_same'))
    )
    receipt = {
        'export_dir': str(export_dir),
        'sha256_file': str(sha_txt),
        'files_sha256': hashes,
        'gpu_used': chosen_gpu,
        'vllm': vllm_res,
        'base_comparison': cmp_res,
        'passed': passed,
        'verified_at': time.time(),
    }
    receipt_path = run_dir / 'export_step100_verified.json'
    write_json(receipt_path, receipt)

    print(f'sha256_archived: {sha_txt} ({len(hashes)} files)')
    print(f'vllm_loaded: {vllm_res.get("loaded")} | generated_tokens: {vllm_res.get("generated_tokens")} | close_think_present: {vllm_res.get("has_close_think")}')
    print(f'chat_template_vs_base: {cmp_res["chat_template"]} | tokenizer_behavior_vs_base: {cmp_res["tokenizer_behavior"]} | tokenizer_raw_files: {cmp_res["tokenizer_files"]}')
    print(f'STEP2_EXPORT_VERIFY_PASSED={passed} receipt={receipt_path}')
    return 0 if passed else 2


def validate_cleanup_target(runs_root: Path, candidate: Path) -> Path:
    runs_resolved = runs_root.resolve(strict=True)
    if candidate.is_symlink():
        raise RuntimeError(f'拒绝软链接目标: {candidate}')
    if not STEP_DIR_RE.fullmatch(candidate.name):
        raise RuntimeError(f'拒绝非 global_step_<数字> 目录名: {candidate}')
    if not candidate.is_dir():
        raise RuntimeError(f'目标不是目录: {candidate}')

    # Walk parents up to runs_root to ensure no symlink lies along the path.
    cur = candidate
    while cur != runs_root:
        if cur.is_symlink():
            raise RuntimeError(f'拒绝路径含软链接: {cur}')
        if cur.parent == cur:
            raise RuntimeError(f'目标不在 {runs_root} 下: {candidate}')
        cur = cur.parent

    resolved = candidate.resolve(strict=True)
    if runs_resolved not in resolved.parents:
        raise RuntimeError(f'目标解析后不在 {runs_resolved} 下: {resolved}')
    return candidate


def collect_checkpoint_targets(runs_root: Path, include_step100: bool, formal_run_name: str = FORMAL_RUN_DEFAULT) -> tuple[list[tuple[Path, int]], list[Path]]:
    if not runs_root.is_dir() or runs_root.is_symlink():
        raise RuntimeError(f'runs 目录无效: {runs_root}')
    formal_step100 = runs_root / formal_run_name / 'checkpoints' / 'global_step_100'
    selected: list[tuple[Path, int]] = []
    skipped_step100: list[Path] = []

    for root, dirs, _ in os.walk(runs_root, followlinks=False):
        rpath = Path(root)
        if rpath.is_symlink():
            raise RuntimeError(f'拒绝软链接目录: {rpath}')
        for d in sorted(dirs):
            p = rpath / d
            if d.startswith('global_step_'):
                validate_cleanup_target(runs_root, p)
                if p == formal_step100 and not include_step100:
                    skipped_step100.append(p)
                    continue
                selected.append((p, dir_size_bytes(p)))
    return selected, skipped_step100


def verify_step2_receipt_for_step100(formal_run: Path) -> dict:
    receipt_path = formal_run / 'export_step100_verified.json'
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise RuntimeError(f'缺少第 2 步核对通过标记，拒绝删除 global_step_100: {receipt_path}')
    data = json.loads(receipt_path.read_text(encoding='utf-8'))
    if data.get('passed') is not True:
        raise RuntimeError('第 2 步导出核对未通过，拒绝删除 global_step_100')
    export_dir = formal_run / 'exports' / 'step_100'
    expected = data.get('files_sha256') or {}
    if not expected:
        raise RuntimeError('第 2 步回执缺少文件 sha256 清单')
    for rel, want_sha in expected.items():
        fp = export_dir / rel
        if not fp.is_file() or fp.is_symlink() or sha256_file(fp) != want_sha:
            raise RuntimeError(f'导出文件校验异常，拒绝删除 global_step_100: {fp}')
    return data


def run_prune(base: Path, *, confirm: bool, include_step100: bool, formal_run_name: str = FORMAL_RUN_DEFAULT) -> int:
    base = base.resolve()
    runs_root = base / 'runs'
    formal_run = runs_root / formal_run_name
    if include_step100:
        verify_step2_receipt_for_step100(formal_run)

    selected, skipped = collect_checkpoint_targets(runs_root, include_step100=include_step100, formal_run_name=formal_run_name)
    total_bytes = sum(sz for _, sz in selected)

    mode = 'EXECUTE (--confirm)' if confirm else 'DRY-RUN (演练，未删除)'
    print(f'=== {mode} | include_step100={include_step100} ===')
    for p, sz in selected:
        print(f'  {format_bytes(sz):>10}  {p}')
    print(f'待删目录数: {len(selected)} | 预计释放合计: {format_bytes(total_bytes)} ({total_bytes} bytes)')
    if skipped:
        print('保留（需加 --include-step100 才删）:')
        for p in skipped:
            print(f'  {format_bytes(dir_size_bytes(p)):>10}  {p}')

    if not confirm:
        return 0

    for p, _ in selected:
        validate_cleanup_target(runs_root, p)
        shutil.rmtree(p)
        # Keep parent checkpoints/latest_checkpointed_iteration.txt clean if global_step_100 was removed.
        if p.name == 'global_step_100' and p.parent.name == 'checkpoints':
            ptr = p.parent / 'latest_checkpointed_iteration.txt'
            if ptr.is_file() and not ptr.is_symlink():
                ptr.unlink()

    du_root = Path('/data') if Path('/data').exists() else base
    du = shutil.disk_usage(du_root)
    print(f'已删除 {len(selected)} 个目录，释放约 {format_bytes(total_bytes)}。')
    print(f'DF {du_root}: free={du.free / 1e9:.2f} GB ({du.free / 1024**3:.2f} GiB)')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_inv = sub.add_parser('inventory')
    p_inv.add_argument('--base', type=Path, default=default_base())

    p_ver = sub.add_parser('verify-export')
    p_ver.add_argument('--base', type=Path, default=default_base())
    p_ver.add_argument('--run-dir', type=Path, default=None)
    p_ver.add_argument('--base-model', type=Path, default=pcfg.TEACHER_MODEL_DEFAULT)
    p_ver.add_argument('--gpu', type=int, default=None)

    p_prn = sub.add_parser('prune')
    p_prn.add_argument('--base', type=Path, default=default_base())
    p_prn.add_argument('--formal-run', default=FORMAL_RUN_DEFAULT)
    p_prn.add_argument('--confirm', action='store_true')
    p_prn.add_argument('--include-step100', action='store_true')

    args = parser.parse_args()
    if args.cmd == 'inventory':
        return run_inventory(args.base)
    if args.cmd == 'verify-export':
        run_dir = args.run_dir or (args.base.resolve() / 'runs' / FORMAL_RUN_DEFAULT)
        return run_verify_export(args.base, run_dir, args.base_model, args.gpu)
    if args.cmd == 'prune':
        return run_prune(
            args.base,
            confirm=args.confirm,
            include_step100=args.include_step100,
            formal_run_name=args.formal_run,
        )
    return 1


if __name__ == '__main__':
    sys.exit(main())
