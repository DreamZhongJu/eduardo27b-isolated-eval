#!/usr/bin/env bash
# Read-only preflight for T-032 Route E checks on Pro 5000 (0 GPUs allocated).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${PRO5000_BASE:-${HOME}/bzhang/a_group_pro5000}"
ENV_DIR="${PRO5000_ENV_DIR:-${BASE}/envs/pro5000-fast}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python}"
BASE_4B="${BASE_4B_MODEL:-/data/models/Qwen3.5-4B}"
STEP100_DIR="${A_GROUP_STEP100:-${BASE}/runs/stage3_formal_20261121_144311/exports/step_100}"
STUDENT_DIR="${STUDENT_MODEL:-/dev/shm/zhujie_a_group/Meta-Llama-3.1-8B-Instruct}"
JUDGE_DIR="${JUDGE_MODEL:-/dev/shm/zhujie_a_group/Qwen3.6-27B}"
HERETIC_TAR="${HERETIC_TAR:-${BASE}/tmp/heretic-d89c7a2d.tar.gz}"
REQUIRE_HERETIC="${REQUIRE_HERETIC:-0}"

unset LD_LIBRARY_PATH HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy CUDA_VISIBLE_DEVICES || true
export LD_LIBRARY_PATH="${ENV_DIR}/lib"
export PATH="${ENV_DIR}/bin:${PATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="${HERE}"

echo "=== T-032 Pro 5000 Preflight ==="
df -BG "${BASE}" /data | awk 'NR==1 || !seen[$1]++'
nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

"${PYTHON_BIN}" -B - "${BASE_4B}" "${STEP100_DIR}" "${STUDENT_DIR}" "${HERETIC_TAR}" "${REQUIRE_HERETIC}" <<'PY'
import importlib.metadata
import json
import sys
from pathlib import Path
from common_utils import (
    BASE_4B_LOCK,
    HERETIC_TAR_SHA256,
    STEP100_LOCK,
    STUDENT_8B_LOCK,
    check_grading_packages,
    detect_idle_gpus,
    print_machine_summary,
    sha256_file,
    verify_directory_lock,
    verify_model_bfloat16,
)

base4b, step100, student, heretic_tar, req_heretic = [Path(x) if i < 4 else x == "1" for i, x in enumerate(sys.argv[1:6])]
vers = check_grading_packages(require_exact=True)
for pkg in ("torch", "vllm", "transformers", "peft"):
    print(f"{pkg}={importlib.metadata.version(pkg)}")
print(f"grading_packages={json.dumps(vers)}")

errs = []
for name, path, lock in (
    ("base_4b", base4b, BASE_4B_LOCK),
    ("a_group_step100", step100, STEP100_LOCK),
    ("student_8b", student, STUDENT_8B_LOCK),
):
    ok, e = verify_directory_lock(path, lock)
    print(f"{name}_lock_ok={ok}")
    if not ok:
        errs.extend([f"{name}:{x}" for x in e])

for name, path in (("base_4b", base4b), ("a_group_step100", step100), ("student_8b", student)):
    try:
        dt = verify_model_bfloat16(path)
        print(f"{name}_dtype={dt}")
    except Exception as exc:
        errs.append(f"{name}_dtype_err:{exc}")

if heretic_tar.is_file():
    h_sha = sha256_file(heretic_tar)
    h_ok = h_sha == HERETIC_TAR_SHA256
    print(f"heretic_tar_ok={h_ok} sha256={h_sha}")
    if not h_ok:
        errs.append("HERETIC_TAR_SHA256_MISMATCH")
elif req_heretic:
    errs.append("HERETIC_TAR_MISSING")

try:
    gpu_info = detect_idle_gpus(snap_interval_s=1.0)
    print(f"idle_gpus={gpu_info['idle_gpus']} busy_gpus={gpu_info['busy_gpus']}")
except Exception as exc:
    gpu_info = {"idle_gpus": [], "busy_gpus": []}
    print(f"gpu_detect_note={exc}")

rc = 1 if errs else 0
if errs:
    print("errors=" + "; ".join(errs))
print_machine_summary(
    {
        "status": "PREFLIGHT_OK" if rc == 0 else "PREFLIGHT_FAIL",
        "rc": rc,
        "calls": 0,
        "failures": len(errs),
        "idle_gpu_count": len(gpu_info["idle_gpus"]),
    }
)
sys.exit(rc)
PY
