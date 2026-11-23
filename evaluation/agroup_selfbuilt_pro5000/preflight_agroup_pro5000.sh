#!/usr/bin/env bash
# Read-only preflight for T-021 on Pro 5000 (A-group step_100 x self-built 3-tier).
# Does NOT allocate GPUs and does NOT call Agnes.
set -euo pipefail

HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE_DIR}/../.." && pwd)"

BASE="${PRO5000_BASE:-${HOME}/bzhang/a_group_pro5000}"
TUTOR_ENV_DIR="${PRO5000_ENV_DIR:-${BASE}/envs/pro5000-fast}"
STUDENT_ENV_DIR="${PRO5000_STUDENT_ENV_DIR:-${BASE}/envs/pro5000-student-vllm018}"
TUTOR_PY="${TUTOR_PY:-${TUTOR_ENV_DIR}/bin/python}"
STUDENT_PY="${STUDENT_PY:-${STUDENT_ENV_DIR}/bin/python}"

A_GROUP_RUN_DIR="${A_GROUP_RUN_DIR:-${BASE}/runs/stage3_formal_20261121_144311}"
TUTOR_MODEL="${TUTOR_MODEL:-${A_GROUP_RUN_DIR}/exports/step_100}"
A_GROUP_SHA_FILE="${A_GROUP_SHA_FILE:-${A_GROUP_RUN_DIR}/export_step100_sha256.txt}"
A_GROUP_VERIFIED_JSON="${A_GROUP_VERIFIED_JSON:-${A_GROUP_RUN_DIR}/export_step100_verified.json}"
STUDENT_MODEL="${STUDENT_MODEL:-/dev/shm/zhujie_a_group/Meta-Llama-3.1-8B-Instruct}"
STUDENT_SYMLINK="${STUDENT_SYMLINK:-${BASE}/models/Meta-Llama-3.1-8B-Instruct}"
BUNDLE_DIR="${BUNDLE_DIR:-${BASE}/t021_bundle}"
AGNES_KEY_ENV="${AGNES_KEY_ENV:-AGNES_API_KEY}"
AGNES_KEY_FILE="${AGNES_KEY_FILE:-${HOME}/.secrets/agnes_api_key}"
MIN_IDLE_GPUS="${MIN_IDLE_GPUS:-2}"

unset LD_LIBRARY_PATH HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true
export LD_LIBRARY_PATH="${TUTOR_ENV_DIR}/lib"
export PATH="${TUTOR_ENV_DIR}/bin:${PATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1
export TMPDIR="${BASE}/tmp" HF_HOME="${BASE}/cache/hf" TRITON_CACHE_DIR="${BASE}/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${BASE}/cache/inductor" VLLM_CACHE_ROOT="${BASE}/cache/vllm"
export XDG_CACHE_HOME="${BASE}/cache/xdg"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${XDG_CACHE_HOME}"

echo "=== 1. Env & Secrets Check (values never printed) ==="
if [ -n "${!AGNES_KEY_ENV:-}" ]; then echo "AGNES_KEY_ENV(${AGNES_KEY_ENV}): 已设"; else echo "AGNES_KEY_ENV(${AGNES_KEY_ENV}): 未设"; fi
if [ -s "${AGNES_KEY_FILE}" ]; then echo "AGNES_KEY_FILE(${AGNES_KEY_FILE}): 存在且非空"; else echo "AGNES_KEY_FILE(${AGNES_KEY_FILE}): 不存在或为空"; fi
if [ -n "${GEMINI_API_KEY:-}" ]; then echo "GEMINI_API_KEY: 已设 (警告: 本任务不应使用 Gemini)"; else echo "GEMINI_API_KEY: 未设"; fi
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "CUDA_VISIBLE_DEVICES: 已设 (建议 unset)"; else echo "CUDA_VISIBLE_DEVICES: 未设"; fi

echo "=== 2. Disk & Ports ==="
df -BG "${BASE}" /data / | awk 'NR==1 || !seen[$1]++'
"${TUTOR_PY}" -B - <<'PY'
import socket
for port in (18701, 18702, 18861, 18961, 18962):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
        state = "FREE"
    except OSError:
        state = "BUSY"
    finally:
        s.close()
    print(f"port 127.0.0.1:{port} -> {state}")
PY

echo "=== 3. Pro 5000 GPUs (2-snap check, >=70000 MiB, GPU 6 sensor quirk) ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
"${TUTOR_PY}" -B - "${HERE_DIR}" "${MIN_IDLE_GPUS}" <<'PY'
import sys, time
sys.path.insert(0, sys.argv[1])
import wait_gpus_agroup_pro5000 as wg
min_idle = int(sys.argv[2])
s1 = wg.snapshot_nvidia_smi()
time.sleep(2)
s2 = wg.snapshot_nvidia_smi()
elig = [g["index"] for g in s2 if wg.is_gpu_eligible(g, 70000)]
for g in s2:
    if g["apps"]:
        for pid, mem in g["apps"]:
            print(f"GPU {g['index']}: pid={pid} mem={mem}MiB")
print(f"idle_gpus_2snap={elig} count={len(elig)} required={min_idle}")
PY

echo "=== 4. Tutor & Student Environments ==="
"${TUTOR_PY}" -B - <<'PY'
import importlib.metadata as md, sys
import torch, transformers, vllm, openai
print(f"tutor_env/python={sys.version.split()[0]} torch={torch.__version__} vllm={vllm.__version__} transformers={transformers.__version__} openai={openai.__version__}")
try:
    print(f"tutor_env/math-verify={md.version('math-verify')}")
except md.PackageNotFoundError:
    print("tutor_env/math-verify=NOT_INSTALLED")
PY

if [ ! -x "${STUDENT_PY}" ]; then
  echo "ERROR: student python missing: ${STUDENT_PY}" >&2
  exit 2
fi
LD_LIBRARY_PATH="${STUDENT_ENV_DIR}/lib:${LD_LIBRARY_PATH}" "${STUDENT_PY}" -B - <<'PY'
import importlib.metadata as md, sys
import torch, transformers, vllm, openai
mv = md.version("math-verify")
print(f"student_env/python={sys.version.split()[0]} torch={torch.__version__} vllm={vllm.__version__} transformers={transformers.__version__} openai={openai.__version__} math-verify={mv}")
if not vllm.__version__.startswith("0.18.0") or not torch.__version__.startswith("2.10.0") or not transformers.__version__.startswith("4.57.6"):
    raise SystemExit("ERROR: Student environment version mismatch (must be vllm 0.18.0, torch 2.10.0, transformers 4.57.6)")
PY

echo "=== 5. Models & Imported A100 Bundle Check ==="
"${TUTOR_PY}" -B - "${TUTOR_MODEL}" "${A_GROUP_SHA_FILE}" "${A_GROUP_VERIFIED_JSON}" "${STUDENT_MODEL}" "${STUDENT_SYMLINK}" <<'PY'
import hashlib, json, sys
from pathlib import Path

ag_dir, ag_sha, ag_ver, stu_shm, stu_sym = [Path(x) for x in sys.argv[1:6]]

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

STEP100_LOCK = {
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    "config.json": "6d320dedc3a50ea3eb4bf7ab503fed018f21784f001c61ff9de9a7ed411e69f0",
    "generation_config.json": "944b9abc4ec388933cc00c757a7b794683929e5a02e5123853892a750645689",
    "model.safetensors": "35435951b689199681a59cc9667cda29e75025f575cd8758bb8320e2f11d88e8",
    "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "7488550a894dcf741616126e322ea31d66cf9457b59ec5825f4c7efdced421f8",
}
t_bad = 0
if not ag_ver.is_file() or not json.loads(ag_ver.read_text()).get("passed"):
    print("A_GROUP_STEP100_VERIFIED_MISSING_OR_FAILED")
    t_bad += 1
for fname, exp in STEP100_LOCK.items():
    p = ag_dir / fname
    if not p.is_file() or p.is_symlink() or sha256_file(p) != exp:
        print(f"A_GROUP_MISMATCH: {fname}")
        t_bad += 1
print(f"a_group_step100_checked={len(STEP100_LOCK)} mismatches={t_bad}")
if t_bad:
    raise SystemExit(2)

STUDENT_LOCK = {
    "config.json": "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e",
    "tokenizer.json": "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
    "tokenizer_config.json": "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424",
    "model-00001-of-00004.safetensors": "2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668",
    "model-00002-of-00004.safetensors": "09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15",
    "model-00003-of-00004.safetensors": "fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa",
    "model-00004-of-00004.safetensors": "92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b",
}
sdir = None
for cand in (stu_shm, stu_sym):
    if cand.exists() and (cand / "config.json").is_file():
        sdir = cand.resolve(strict=True)
        break
s_bad = 0 if sdir else len(STUDENT_LOCK)
if sdir:
    for fname, exp in STUDENT_LOCK.items():
        p = sdir / fname
        if not p.is_file() or sha256_file(p) != exp:
            print(f"STUDENT_MISMATCH: {fname}")
            s_bad += 1
print(f"student_llama31_8b_checked={len(STUDENT_LOCK)} mismatches={s_bad}")
if s_bad:
    raise SystemExit(2)
PY

"${TUTOR_PY}" -B "${HERE_DIR}/import_bundle.py" --bundle-dir "${BUNDLE_DIR}"
echo "PRO5000_T021_PREFLIGHT_OK"
