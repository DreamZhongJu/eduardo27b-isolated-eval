#!/usr/bin/env bash
# T-031 Stage 2 Preflight for Pro 5000 (bf16 student + Qwen3.6-27B TP=2 + math-verify==0.8.0).
# Also verifies all 8 Pro 5000 GPUs are idle (T-026 finished) before Stage 2 starts.
set -euo pipefail

HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${HERE_DIR}/.." && pwd)"
BASE="${PRO5000_BASE:-${HOME}/bzhang/a_group_pro5000}"
ENV_DIR="${PRO5000_ENV_DIR:-${BASE}/envs/pro5000-fast}"
EDUARDO_REPO="${EDUARDO_REPO:-${BASE}/source/EduardoRL}"
STUDENT_MODEL="${STUDENT_MODEL:-/dev/shm/zhujie_a_group/Meta-Llama-3.1-8B-Instruct}"
JUDGE_MODEL="${JUDGE_MODEL:-/dev/shm/zhujie_a_group/Qwen3.6-27B}"
MEASURED_JSONL="${1:?Usage: preflight_stage2_pro5000.sh <MEASURED_JSONL> [EXPECTED_SHA256]}"
EXPECTED_SHA256="${2:-9ff8449b781906290f9b671bc5be1982ca31e5401ebfba485474d475a60752ce}"

PYTHON_BIN="${ENV_DIR}/bin/python"
[ -x "${PYTHON_BIN}" ] || { echo "ERROR: missing python at ${PYTHON_BIN}" >&2; exit 2; }
[ -d "${EDUARDO_REPO}/eduardo" ] || { echo "ERROR: missing EduardoRL repo at ${EDUARDO_REPO}" >&2; exit 2; }

export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${CODE_ROOT}:${EDUARDO_REPO}"

"${PYTHON_BIN}" -B - "${CODE_ROOT}" "${STUDENT_MODEL}" "${JUDGE_MODEL}" "${MEASURED_JSONL}" "${EXPECTED_SHA256}" "${BASE}" <<'PY'
import hashlib, importlib.metadata, json, shutil, subprocess, sys
from pathlib import Path
from routeC_data.stage2_source_runner import is_pro5000_gpu_idle, parse_pro5000_gpu_csv_with_used

code_root = Path(sys.argv[1])
student_arg = Path(sys.argv[2])
judge_arg = Path(sys.argv[3])
measured_jsonl = Path(sys.argv[4])
exp_sha = sys.argv[5].strip().lower()
base_dir = Path(sys.argv[6])

hard_errors = []

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

print("=== 1. Package Pins (Author e8c89b7 + vLLM) ===")
PINS = {"math-verify": "0.8.0", "latex2sympy2_extended": "1.10.2"}
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

print("=== 2. Canonical hybrid_grader.py ===")
hg_path = code_root / "routeA_data" / "hybrid_grader.py"
EXP_HG_SHA = "7d2a5f46619218e92ed11efc4e7e07f190027ac0f1cca9336905c090ff62709a"
if not hg_path.is_file():
    hard_errors.append("HYBRID_GRADER_MISSING")
else:
    got_hg = sha256_file(hg_path)
    print(f"  hybrid_grader.py sha256={got_hg} ok={got_hg == EXP_HG_SHA}")
    if got_hg != EXP_HG_SHA:
        hard_errors.append(f"HYBRID_GRADER_SHA_MISMATCH:{got_hg}")

print("=== 3. Input T-023 Measured JSONL SHA-256 ===")
if not measured_jsonl.is_file():
    print(f"  MEASURED_JSONL MISSING: {measured_jsonl}")
    hard_errors.append("MEASURED_JSONL_MISSING")
else:
    got_sha = sha256_file(measured_jsonl)
    n_lines = 0
    with measured_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n_lines += 1
    print(f"  measured_jsonl={measured_jsonl} sha256={got_sha} ok={got_sha == exp_sha} lines={n_lines}")
    if got_sha != exp_sha:
        hard_errors.append(f"MEASURED_SHA_MISMATCH:{got_sha}")

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
    hard_errors.append("STUDENT_MISSING")
else:
    s_bad = sum(1 for fn, exp in STUDENT_LOCK.items() if not (student_dir / fn).is_file() or sha256_file(student_dir / fn) != exp)
    print(f"  student_dir={student_dir} checked={len(STUDENT_LOCK)} mismatches={s_bad}")
    if s_bad:
        hard_errors.append(f"STUDENT_MISMATCH:{s_bad}")

print("=== 5. Judge Qwen3.6-27B SHA-256 Lock ===")
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
    hard_errors.append("JUDGE_MISSING")
else:
    j_bad = sum(1 for fn, exp in JUDGE_LOCK.items() if not (judge_dir / fn).is_file() or sha256_file(judge_dir / fn) != exp)
    print(f"  judge_dir={judge_dir} checked={len(JUDGE_LOCK)} mismatches={j_bad}")
    if j_bad:
        hard_errors.append(f"JUDGE_MISMATCH:{j_bad}")

print("=== 6. All-8-GPU Idle Gate (T-026 Must Be Finished) & Disk ===")
gpu_csv = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
    text=True,
)
apps_csv = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
    text=True,
)
gpus = parse_pro5000_gpu_csv_with_used(gpu_csv, apps_csv)
idle_idxs = [g["index"] for g in gpus if is_pro5000_gpu_idle(g, min_free_mib=70000, max_used_mib=500)]
busy_idxs = [g["index"] for g in gpus if not is_pro5000_gpu_idle(g, min_free_mib=70000, max_used_mib=500)]
gpu_brief = ", ".join(f"{g['index']}:used={g['used_mib']}MiB/free={g['free_mib']}MiB/util={g['util_pct']}%" for g in gpus)
print(f"  total_gpus={len(gpus)} idle_gpus={idle_idxs} busy_gpus={busy_idxs}")
print(f"  gpu_detail=[{gpu_brief}]")
if len(gpus) < 8 or len(idle_idxs) < len(gpus):
    hard_errors.append(f"GPUS_NOT_ALL_IDLE:idle={len(idle_idxs)}/{len(gpus)}")

free_gb = shutil.disk_usage(base_dir if base_dir.exists() else Path.home()).free / 1e9
print(f"  base_free_gb={free_gb:.2f}")
if free_gb < 15.0:
    hard_errors.append(f"DISK_LOW:{free_gb:.2f}GB")

if hard_errors:
    print("PREFLIGHT_FAILED:", hard_errors, file=sys.stderr)
    raise SystemExit(1)
print("PREFLIGHT_LOCK_CHECKS=OK")
PY

echo "=== 7. Run Unit Tests Inside pro5000-fast (with real math-verify==0.8.0) ==="
(cd "${CODE_ROOT}" && CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" -B -m unittest discover -s tests -p "test_routeC*.py" -v)
echo "PREFLIGHT_ALL_OK=1"
