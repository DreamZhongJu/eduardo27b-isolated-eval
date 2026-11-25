#!/usr/bin/env bash
# T-023 Stage 2 Preflight for Pro 5000 (bf16 student + Qwen3.6-27B TP=2 + math-verify==0.8.0).
set -euo pipefail

HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${HERE_DIR}/.." && pwd)"
BASE="${PRO5000_BASE:-${HOME}/bzhang/a_group_pro5000}"
ENV_DIR="${PRO5000_ENV_DIR:-${BASE}/envs/pro5000-fast}"
EDUARDO_REPO="${EDUARDO_REPO:-${BASE}/source/EduardoRL}"
STUDENT_MODEL="${STUDENT_MODEL:-/dev/shm/zhujie_a_group/Meta-Llama-3.1-8B-Instruct}"
JUDGE_MODEL="${JUDGE_MODEL:-/dev/shm/zhujie_a_group/Qwen3.6-27B}"
DATA_JSONL="${1:?Usage: preflight_stage2_pro5000.sh <DATA_JSONL> [EXPECTED_DATA_SHA256]}"
EXPECTED_DATA_SHA256="${2:-3bab0292bb6d2393a4f64cf8bf4dad282499b0bf395905c52dac9559b5f4f333}"

PYTHON_BIN="${ENV_DIR}/bin/python"
[ -x "${PYTHON_BIN}" ] || { echo "ERROR: missing python at ${PYTHON_BIN}" >&2; exit 2; }
[ -d "${EDUARDO_REPO}/eduardo" ] || { echo "ERROR: missing EduardoRL repo at ${EDUARDO_REPO}" >&2; exit 2; }

export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${CODE_ROOT}:${EDUARDO_REPO}"

"${PYTHON_BIN}" -B - "${CODE_ROOT}" "${EDUARDO_REPO}" "${STUDENT_MODEL}" "${JUDGE_MODEL}" "${DATA_JSONL}" "${EXPECTED_DATA_SHA256}" "${BASE}" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path

code_root = Path(sys.argv[1])
eduardo_repo = Path(sys.argv[2])
student_arg = Path(sys.argv[3])
judge_arg = Path(sys.argv[4])
data_jsonl = Path(sys.argv[5])
exp_data_sha = sys.argv[6].strip().lower()
base_dir = Path(sys.argv[7])

hard_errors = []


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


print("=== 1. Package Pins (Author e8c89b7 + vLLM) ===")
PINS = {
    "math-verify": "0.8.0",
    "latex2sympy2_extended": "1.10.2",
}
for pkg in ("vllm", "torch", "openai", "httpx", "math-verify", "latex2sympy2_extended", "latex2sympy2"):
    try:
        ver = importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        ver = "NOT_INSTALLED"
    print(f"  {pkg}={ver}")
    if pkg in PINS and ver != PINS[pkg]:
        hard_errors.append(f"PIN_MISMATCH:{pkg}={ver}!={PINS[pkg]}")
    elif pkg == "latex2sympy2" and ver not in ("NOT_INSTALLED", "1.5.4"):
        hard_errors.append(f"PIN_MISMATCH:{pkg}={ver}!=1.5.4")

print("=== 2. Canonical hybrid_grader.py & Agnes Prompt Reference ===")
hg_path = code_root / "routeA_data" / "hybrid_grader.py"
EXP_HG_SHA = "7d2a5f46619218e92ed11efc4e7e07f190027ac0f1cca9336905c090ff62709a"
if not hg_path.is_file():
    hard_errors.append("HYBRID_GRADER_MISSING")
else:
    got_hg = sha256_file(hg_path)
    print(f"  hybrid_grader.py sha256={got_hg} ok={got_hg == EXP_HG_SHA}")
    if got_hg != EXP_HG_SHA:
        hard_errors.append(f"HYBRID_GRADER_SHA_MISMATCH:{got_hg}")

print("=== 3. Input Dataset JSONL SHA-256 ===")
if not data_jsonl.is_file():
    print(f"  DATA_JSONL MISSING: {data_jsonl}")
    hard_errors.append("DATA_JSONL_MISSING")
else:
    got_data = sha256_file(data_jsonl)
    n_lines = 0
    n_train = 0
    n_dev = 0
    n_vars = 0
    with data_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n_lines += 1
            if r.get("split") == "train":
                n_train += 1
            elif r.get("split") == "dev":
                n_dev += 1
            n_vars += len(r.get("variants", []))
    print(f"  data_jsonl={data_jsonl} sha256={got_data} ok={got_data == exp_data_sha} lines={n_lines} (train={n_train}, dev={n_dev}) variants={n_vars}")
    if got_data != exp_data_sha:
        hard_errors.append(f"DATA_SHA_MISMATCH:{got_data}")

print("=== 4. Student Meta-Llama-3.1-8B-Instruct SHA-256 Lock ===")
STUDENT_LOCK = {
    "config.json": "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e",
    "tokenizer.json": "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
    "tokenizer_config.json": "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424",
    "model-00001-of-00004.safetensors": "2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668",
    "model-00002-of-00004.safetensors": "09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15",
    "model-00003-of-00004.safetensors": "fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa",
    "model-00004-of-00004.safetensors": "92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b",
}
student_dir = None
for cand in (student_arg, base_dir / "models" / "Meta-Llama-3.1-8B-Instruct"):
    if cand.exists() and (cand / "config.json").is_file():
        student_dir = cand.resolve(strict=True)
        break
if student_dir is None:
    print("  student_dir: MISSING")
    hard_errors.append("STUDENT_MISSING")
else:
    s_bad = 0
    for fn, exp in STUDENT_LOCK.items():
        p = student_dir / fn
        if not p.is_file() or sha256_file(p) != exp:
            print(f"  student/{fn}: MISMATCH_OR_MISSING")
            s_bad += 1
    print(f"  student_dir={student_dir} checked={len(STUDENT_LOCK)} mismatches={s_bad}")
    if s_bad:
        hard_errors.append(f"STUDENT_MISMATCH:{s_bad}")

print("=== 5. Judge Qwen3.6-27B (HF 8f33b989) SHA-256 Lock ===")
JUDGE_LOCK = {
    "model-00001-of-00015.safetensors": "5f21d4e349aef6c74bedef7b3835dc8c11a16dd5ce72f4437e2284f1e83736e9",
    "model-00002-of-00015.safetensors": "03de44dc7e933025498d72f8d7ea32d5cde16eb7a85b35ff1fa49a737f4b2242",
    "model-00003-of-00015.safetensors": "5c3a68304dabeaa5a0eb70a5e383d2b8134997ef6d5c1e8afdcd236091c847f2",
    "model-00004-of-00015.safetensors": "ba8b0849cb4c4c97e674709bfe56a9c008d7563a68ac57120ceb2855a73a9944",
    "model-00005-of-00015.safetensors": "a5abc1d5e9583409193e2cd58a671a4ed98467a6eaf883cd4377e3cc02021c22",
    "model-00006-of-00015.safetensors": "160d914e2e4704a401a3eab9e9eec3380e5723a35981c9169bedfb82ea32e6d6",
    "model-00007-of-00015.safetensors": "0bcd0ce28c7d2cc6f5ab2c21902cd032fec0cc5f29e0897e3bb0967add388e47",
    "model-00008-of-00015.safetensors": "584a0ed8018d3b19ce8e533d08a460454650027901be6323b50b591da143dca6",
    "model-00009-of-00015.safetensors": "e7e3e1a17a2673340eb6bba95c3c6f07b12e7746ee738dde56551cbde27656f5",
    "model-00010-of-00015.safetensors": "e8934789f4742c11da88f937da84bdbb8764c8a7f7ebe9e7b056bcbd41b14285",
    "model-00011-of-00015.safetensors": "44e8fe06d2d609bf20b16b1d2f42348ca2dc99179721f76035ff50db678f9a8a",
    "model-00012-of-00015.safetensors": "33c5d7d18e1b3f661334dff736fc7d00561f1f1bb9d2970c669b40d2e582974e",
    "model-00013-of-00015.safetensors": "68db2ebb03231238c2a114f3642df967a970a1d96b029adc427838cb5cf7f27a",
    "model-00014-of-00015.safetensors": "26c114fb6d5d4131ab227552d489d7a97c8c958a9ed94f6d3d1584f6cfe4b9fb",
    "model-00015-of-00015.safetensors": "b84b5b1315e865c9a19a444045d422a73e3e2e31ce3766797cffd3507c68c9c9",
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
}
judge_dir = None
for cand in (judge_arg, base_dir / "models" / "Qwen3.6-27B"):
    if cand.exists() and (cand / "config.json").is_file():
        judge_dir = cand.resolve(strict=True)
        break
if judge_dir is None:
    print("  judge_dir: MISSING")
    hard_errors.append("JUDGE_MISSING")
else:
    j_bad = 0
    for fn, exp in JUDGE_LOCK.items():
        p = judge_dir / fn
        if not p.is_file() or sha256_file(p) != exp:
            print(f"  judge/{fn}: MISMATCH_OR_MISSING")
            j_bad += 1
    jcfg = json.loads((judge_dir / "config.json").read_text(encoding="utf-8"))
    jdtype = jcfg.get("torch_dtype") or jcfg.get("dtype")
    print(f"  judge_dir={judge_dir} torch_dtype={jdtype} checked={len(JUDGE_LOCK)} mismatches={j_bad}")
    if j_bad:
        hard_errors.append(f"JUDGE_MISMATCH:{j_bad}")

print("=== 6. Disk & Agnes Key ===")
free_gb = shutil.disk_usage(base_dir if base_dir.exists() else Path.home()).free / 1e9
print(f"  base_free_gb={free_gb:.2f}")
if free_gb < 15.0:
    hard_errors.append(f"DISK_LOW:{free_gb:.2f}GB")
has_key = bool(os.environ.get("AGNES_API_KEY", "").strip())
print(f"  AGNES_API_KEY_SET={has_key}")
if not has_key:
    hard_errors.append("AGNES_API_KEY_MISSING")

if hard_errors:
    print("PREFLIGHT_FAILED:", hard_errors, file=sys.stderr)
    raise SystemExit(1)
print("PREFLIGHT_LOCK_CHECKS=OK")
PY

echo "=== 7. Run Unit Tests Inside pro5000-fast (with real math-verify==0.8.0) ==="
(cd "${CODE_ROOT}" && "${PYTHON_BIN}" -B -m unittest discover -s tests -p "test_*.py" -v)
echo "PREFLIGHT_ALL_OK=1"
