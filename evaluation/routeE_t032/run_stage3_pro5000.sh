#!/usr/bin/env bash
# Stage 3 runner for T-032 P4 E0 on Pro 5000:
#   --step {all|contrastive|fit_methods|eval_val100|status}
# Prefers sharing existing Route A Student + 27B vLLM services; otherwise starts
# missing services only if enough idle GPUs exist (total occupied <= 8).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${PRO5000_BASE:-${HOME}/bzhang/a_group_pro5000}"
ENV_DIR="${PRO5000_ENV_DIR:-${BASE}/envs/pro5000-fast}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python}"
VLLM_BIN="${VLLM_BIN:-${ENV_DIR}/bin/vllm}"

BASE_4B="${BASE_4B_MODEL:-/data/models/Qwen3.5-4B}"
STEP100_DIR="${A_GROUP_STEP100:-${BASE}/runs/stage3_formal_20261121_144311/exports/step_100}"
STUDENT_MODEL_DIR="${STUDENT_MODEL:-/dev/shm/zhujie_a_group/Meta-Llama-3.1-8B-Instruct}"
JUDGE_MODEL_DIR="${JUDGE_MODEL_DIR:-/dev/shm/zhujie_a_group/Qwen3.6-27B}"
HERETIC_ROOT="${HERETIC_ROOT:-${BASE}/external/heretic-d89c7a2d}"

STEP="all"
RUN_DIR="${BASE}/runs/t032_stage3_e0"
SHARED_STUDENT_URL=""
SHARED_JUDGE_URL=""
STUDENT_PORT="18992"
JUDGE_PORT="18993"
ARA_SMALL_RANK="16"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --step) STEP="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --heretic-root) HERETIC_ROOT="$2"; shift 2 ;;
    --shared-student-api-base) SHARED_STUDENT_URL="$2"; shift 2 ;;
    --shared-judge-api-base) SHARED_JUDGE_URL="$2"; shift 2 ;;
    --ara-small-rank) ARA_SMALL_RANK="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

unset LD_LIBRARY_PATH HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy CUDA_VISIBLE_DEVICES AGNES_API_KEY GEMINI_API_KEY || true
export LD_LIBRARY_PATH="${ENV_DIR}/lib"
export PATH="${ENV_DIR}/bin:${PATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="${HERE}"
export TMPDIR="${BASE}/tmp" HF_HOME="${BASE}/cache/hf" TRITON_CACHE_DIR="${BASE}/cache/triton" \
       TORCHINDUCTOR_CACHE_DIR="${BASE}/cache/inductor" VLLM_CACHE_ROOT="${BASE}/cache/vllm" XDG_CACHE_HOME="${BASE}/cache/xdg"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${XDG_CACHE_HOME}" "${RUN_DIR}"

STUDENT_PID=""
JUDGE_PID=""
cleanup() {
  for p in "${STUDENT_PID}" "${JUDGE_PID}"; do
    if [ -n "${p}" ] && kill -0 "${p}" 2>/dev/null; then
      kill -TERM "${p}" 2>/dev/null || true
      wait "${p}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

wait_health() {
  local pid="$1" port="$2" name="$3"
  "${PYTHON_BIN}" -B - "${pid}" "${port}" "${name}" <<'PY'
import os, sys, time, urllib.request
pid, port, name = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
deadline = time.monotonic() + 600
while time.monotonic() < deadline:
    os.kill(pid, 0)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
            if r.status == 200:
                sys.exit(0)
    except OSError:
        pass
    time.sleep(5)
raise SystemExit(f"ERROR: {name} health timeout")
PY
}

allocate_gpus() {
  local step_name="$1"
  local need_4b=1 need_stu=0 need_27b=0
  if [ "${step_name}" != "fit_methods" ]; then
    [ -z "${SHARED_STUDENT_URL}" ] && need_stu=1
    if [ "${step_name}" = "contrastive" ] || [ "${step_name}" = "eval_val100" ] || [ "${step_name}" = "all" ]; then
      [ -z "${SHARED_JUDGE_URL}" ] && need_27b=2
    fi
  fi
  "${PYTHON_BIN}" -B - "${need_4b}" "${need_stu}" "${need_27b}" <<'PY'
import sys
from common_utils import detect_idle_gpus
n4, ns, n27 = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
req = n4 + ns + n27
info = detect_idle_gpus()
idle = info["idle_gpus"]
if len(idle) < req or len(info["busy_gpus"]) + req > 8:
    raise SystemExit(
        f"ERROR: need {req} idle GPUs and total occupied <=8 (4B={n4}, Stu={ns}, 27B={n27}), "
        f"got idle={idle} busy={info['busy_gpus']}"
    )
g4 = str(idle[0])
idx = 1
gs = str(idle[idx]) if ns else ""
idx += ns
g27 = f"{idle[idx]},{idle[idx+1]}" if n27 else ""
print(f"{g4}|{gs}|{g27}")
PY
}

if [ "${STEP}" = "status" ]; then
  for f in "${RUN_DIR}/contrastive/contrastive_stats.json" "${RUN_DIR}/artifacts/base_4b/fit_summary.json" "${RUN_DIR}/artifacts/a_group_step100/fit_summary.json" "${RUN_DIR}/val100/p4_val100_table.json"; do
    [ -f "${f}" ] && echo "FOUND=${f}" || echo "PENDING=${f}"
  done
  exit 0
fi

ALLOC="$(allocate_gpus "${STEP}")"
GPU_4B="${ALLOC%%|*}"
REST="${ALLOC#*|}"
GPU_STU="${REST%%|*}"
GPU_27B="${REST#*|}"

STU_URL="${SHARED_STUDENT_URL}"
if [ -z "${STU_URL}" ] && [ -n "${GPU_STU}" ]; then
  CUDA_VISIBLE_DEVICES="${GPU_STU}" "${VLLM_BIN}" serve "${STUDENT_MODEL_DIR}" \
    --host 127.0.0.1 --port "${STUDENT_PORT}" --served-model-name "Llama-3.1-8B-Instruct" \
    --dtype bfloat16 --max-model-len 32768 --max-num-seqs 256 --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill --enable-prefix-caching \
    > "${RUN_DIR}/student_8b_server.log" 2>&1 &
  STUDENT_PID=$!
  wait_health "${STUDENT_PID}" "${STUDENT_PORT}" "student_8b"
  STU_URL="http://127.0.0.1:${STUDENT_PORT}/v1"
fi

J_URL="${SHARED_JUDGE_URL}"
if [ -z "${J_URL}" ] && [ -n "${GPU_27B}" ]; then
  CUDA_VISIBLE_DEVICES="${GPU_27B}" "${VLLM_BIN}" serve "${JUDGE_MODEL_DIR}" \
    --host 127.0.0.1 --port "${JUDGE_PORT}" --served-model-name "Qwen3.6-27B" \
    --dtype bfloat16 --tensor-parallel-size 2 --max-model-len 32768 --max-num-seqs 256 \
    --gpu-memory-utilization 0.90 --enable-chunked-prefill --enable-prefix-caching \
    > "${RUN_DIR}/judge_27b_server.log" 2>&1 &
  JUDGE_PID=$!
  wait_health "${JUDGE_PID}" "${JUDGE_PORT}" "judge_27b"
  J_URL="http://127.0.0.1:${JUDGE_PORT}/v1"
fi

run_contrastive() {
  CUDA_VISIBLE_DEVICES="${GPU_4B}" "${PYTHON_BIN}" -B "${HERE}/p4_contrastive_data.py" \
    --data-dir "${BASE}/data/author_prepared" \
    --out-dir "${RUN_DIR}/contrastive" \
    --rewriter-api-base "${J_URL}" \
    --tutor-model "${STEP100_DIR}" \
    --student-api-base "${STU_URL}" \
    --require-math-verify
}

run_fit_methods() {
  for blabel_path in "base_4b=${BASE_4B}" "a_group_step100=${STEP100_DIR}"; do
    blabel="${blabel_path%%=*}"
    bpath="${blabel_path#*=}"
    CUDA_VISIBLE_DEVICES="${GPU_4B}" "${PYTHON_BIN}" -B "${HERE}/p4_e0_methods.py" \
      --base-label "${blabel}" \
      --base-model "${bpath}" \
      --fit-jsonl "${RUN_DIR}/contrastive/contrastive_fit.jsonl" \
      --heldout-jsonl "${RUN_DIR}/contrastive/contrastive_heldout.jsonl" \
      --heretic-root "${HERETIC_ROOT}" \
      --out-dir "${RUN_DIR}/artifacts" \
      --ara-small-rank "${ARA_SMALL_RANK}"
  done
}

run_eval_val100() {
  CUDA_VISIBLE_DEVICES="${GPU_4B}" "${PYTHON_BIN}" -B "${HERE}/p4_eval_val100.py" \
    --val-parquet "${BASE}/data/author_prepared/val.parquet" \
    --data-dir "${BASE}/data/routeE_benchmarks" \
    --base4b-path "${BASE_4B}" \
    --agroup-path "${STEP100_DIR}" \
    --artifacts-dir "${RUN_DIR}/artifacts" \
    --out-dir "${RUN_DIR}/val100" \
    --student-api-base "${STU_URL}" \
    --judge-api-base "${J_URL}" \
    --require-math-verify
}

case "${STEP}" in
  contrastive) run_contrastive ;;
  fit_methods) run_fit_methods ;;
  eval_val100) run_eval_val100 ;;
  all)
    run_contrastive
    run_fit_methods
    run_eval_val100
    ;;
  *)
    echo "Usage: $0 --step {all|contrastive|fit_methods|eval_val100|status} [options]" >&2
    exit 2
    ;;
esac
