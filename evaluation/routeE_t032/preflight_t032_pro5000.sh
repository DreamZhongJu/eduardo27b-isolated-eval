#!/usr/bin/env bash
# Read-only preflight for T-032 Route E checks on Pro 5000 (0 GPUs allocated).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${PRO5000_BASE:-${HOME}/bzhang/a_group_pro5000}"
ENV_DIR="${PRO5000_ENV_DIR:-${BASE}/envs/pro5000-fast}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python}"
BASE_4B="${BASE_4B_MODEL:-/data/models/Qwen3.5-4B}"
STEP100_DIR="${A_GROUP_STEP100:-${BASE}/runs/stage3_formal_20261121_144311/exports/step_100}"
ROUTE_A_STEP100_DIR="${ROUTE_A_STEP100:-${BASE}/runs/t026_stage3_formal_20261126_140508/exports/step_100}"
ROUTE_A_SHA_FILE="${ROUTE_A_STEP100_SHA_FILE:-${BASE}/runs/t026_stage3_formal_20261126_140508/export_step100_sha256.txt}"
TEST_494_PARQUET="${TEST_494_PARQUET:-${BASE}/data/author494_eval/test.parquet}"
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

"${PYTHON_BIN}" -B - "${BASE_4B}" "${STEP100_DIR}" "${ROUTE_A_STEP100_DIR}" "${ROUTE_A_SHA_FILE}" "${TEST_494_PARQUET}" "${STUDENT_DIR}" "${HERETIC_TAR}" "${REQUIRE_HERETIC}" <<'PY'
import importlib.metadata
import json
import sys
from pathlib import Path
from common_utils import (
    BASE_4B_LOCK,
    HERETIC_TAR_SHA256,
    ROUTE_A_STEP100_LOCK,
    STEP100_LOCK,
    STUDENT_8B_LOCK,
    TEST_494_PARQUET_SHA256,
    check_grading_packages,
    detect_idle_gpus,
    print_machine_summary,
    sha256_file,
    verify_directory_lock,
    verify_model_bfloat16,
)

base4b, step100, route_a_step100, route_a_sha_file, test_494_pq, student, heretic_tar = [
    Path(x) for x in sys.argv[1:8]
]
req_heretic = sys.argv[8] == "1"
vers = check_grading_packages(require_exact=True)
for pkg in ("torch", "vllm", "transformers", "peft"):
    print(f"{pkg}={importlib.metadata.version(pkg)}")
print(f"grading_packages={json.dumps(vers)}")

errs = []
for name, path, lock in (
    ("base_4b", base4b, BASE_4B_LOCK),
    ("a_group_step100", step100, STEP100_LOCK),
    ("route_a_step100", route_a_step100, ROUTE_A_STEP100_LOCK),
    ("student_8b", student, STUDENT_8B_LOCK),
):
    ok, e = verify_directory_lock(path, lock)
    print(f"{name}_lock_ok={ok}")
    if not ok:
        errs.extend([f"{name}:{x}" for x in e])

if route_a_sha_file.is_file():
    manifest_lock = {}
    for line in route_a_sha_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            manifest_lock[parts[1]] = parts[0]
    m_ok, m_errs = verify_directory_lock(route_a_step100, manifest_lock)
    print(f"route_a_step100_manifest_ok={m_ok}")
    if not m_ok:
        errs.extend([f"route_a_manifest:{x}" for x in m_errs])
else:
    print("route_a_step100_manifest_ok=False")
    errs.append(f"ROUTE_A_SHA_FILE_MISSING:{route_a_sha_file}")

if test_494_pq.is_file():
    t_sha = sha256_file(test_494_pq)
    t_ok = t_sha == TEST_494_PARQUET_SHA256
    print(f"test_494_parquet_lock_ok={t_ok}")
    if not t_ok:
        errs.append(f"TEST_494_PARQUET_SHA256_MISMATCH:{t_sha}")
else:
    print("test_494_parquet_lock_ok=False")
    errs.append(f"TEST_494_PARQUET_MISSING:{test_494_pq}")

for name, path in (
    ("base_4b", base4b),
    ("a_group_step100", step100),
    ("route_a_step100", route_a_step100),
    ("student_8b", student),
):
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
