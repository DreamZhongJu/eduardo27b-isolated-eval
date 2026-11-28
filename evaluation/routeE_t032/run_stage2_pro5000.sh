#!/usr/bin/env bash
# Stage 2 runner for T-032 on Pro 5000:
#   --step p1                 (1 idle GPU, bf16, MATH500 + GSM8K)
#   --step p2_sample          (0 GPUs, stratified 50+50 blind sampling)
#   --step p2_export_claude30 (0 GPUs, export 30 blind dialogues for Claude)
#   --step p2_calibrate       (reuses shared 27B or starts TP=2 on 2 idle GPUs)
#   --step p3                 (reuses shared 27B or starts TP=2 on 2 idle GPUs)
#   --step status             (print latest run status & aggregate summary)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${PRO5000_BASE:-${HOME}/bzhang/a_group_pro5000}"
ENV_DIR="${PRO5000_ENV_DIR:-${BASE}/envs/pro5000-fast}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python}"
VLLM_BIN="${VLLM_BIN:-${ENV_DIR}/bin/vllm}"

BASE_4B="${BASE_4B_MODEL:-/data/models/Qwen3.5-4B}"
STEP100_DIR="${A_GROUP_STEP100:-${BASE}/runs/stage3_formal_20261121_144311/exports/step_100}"
JUDGE_MODEL_DIR="${JUDGE_MODEL_DIR:-/dev/shm/zhujie_a_group/Qwen3.6-27B}"
AGROUP_494_JSON="${AGROUP_494_JSON:-${BASE}/runs/eval_author494_agroup_full_20261123_134421/a_group_step100/transfer_eval_a_group_step100_author494_pro5000_20261123_134533.json}"
M27B_494_JSON="${M27B_494_JSON:-${BASE}/runs/t022_27b_pro5000_20261125_102341/full494/transfer_eval_eduardo_27b_author494_pro5000_full494_20261125_103810.json}"

STEP=""
RUN_DIR="${BASE}/runs/t032_stage2"
SHARED_JUDGE_URL=""
JUDGE_PORT="18991"
USER_CSV=""
CLAUDE_CSV=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --step) STEP="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --shared-judge-api-base) SHARED_JUDGE_URL="$2"; shift 2 ;;
    --judge-port) JUDGE_PORT="$2"; shift 2 ;;
    --user-csv) USER_CSV="$2"; shift 2 ;;
    --claude-csv) CLAUDE_CSV="$2"; shift 2 ;;
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

JUDGE_PID=""
cleanup() {
  if [ -n "${JUDGE_PID}" ] && kill -0 "${JUDGE_PID}" 2>/dev/null; then
    kill -TERM "${JUDGE_PID}" 2>/dev/null || true
    wait "${JUDGE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

ensure_judge_server() {
  if [ -n "${SHARED_JUDGE_URL}" ]; then
    echo "${SHARED_JUDGE_URL}"
    return 0
  fi
  local gpus
  gpus="$("${PYTHON_BIN}" -B - <<'PY'
import sys
from common_utils import detect_idle_gpus
info = detect_idle_gpus()
idle = info["idle_gpus"]
if len(idle) < 2 or len(info["busy_gpus"]) + 2 > 8:
    raise SystemExit(
        f"ERROR: need >=2 idle GPUs and total occupied <=8 for 27B TP=2, got idle={idle} busy={info['busy_gpus']}"
    )
print(f"{idle[0]},{idle[1]}")
PY
)"
  CUDA_VISIBLE_DEVICES="${gpus}" "${VLLM_BIN}" serve "${JUDGE_MODEL_DIR}" \
    --host 127.0.0.1 --port "${JUDGE_PORT}" --served-model-name "Qwen3.6-27B" \
    --dtype bfloat16 --tensor-parallel-size 2 --max-model-len 32768 --max-num-seqs 256 \
    --gpu-memory-utilization 0.90 --enable-chunked-prefill --enable-prefix-caching \
    > "${RUN_DIR}/judge_27b_server.log" 2>&1 &
  JUDGE_PID=$!
  "${PYTHON_BIN}" -B - "${JUDGE_PID}" "${JUDGE_PORT}" <<'PY'
import os, sys, time, urllib.request
pid, port = int(sys.argv[1]), int(sys.argv[2])
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
raise SystemExit("ERROR: 27B server startup timeout")
PY
  echo "http://127.0.0.1:${JUDGE_PORT}/v1"
}

case "${STEP}" in
  p1)
    GPU_ID="$("${PYTHON_BIN}" -B - <<'PY'
from common_utils import detect_idle_gpus
info = detect_idle_gpus()
idle = info["idle_gpus"]
if not idle or len(info["busy_gpus"]) + 1 > 8:
    raise SystemExit(
        f"ERROR: need >=1 idle GPU and total occupied <=8 for P1, got idle={idle} busy={info['busy_gpus']}"
    )
print(idle[0])
PY
)"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -B "${HERE}/p1_math_checkup.py" \
      --models "base_4b=${BASE_4B},a_group_step100=${STEP100_DIR}" \
      --benchmarks math500,gsm8k \
      --data-dir "${BASE}/data/routeE_benchmarks" \
      --out-dir "${RUN_DIR}/p1" \
      --require-math-verify
    ;;
  p2_sample)
    "${PYTHON_BIN}" -B "${HERE}/p2_gold_sampling.py" \
      --agroup-json "${AGROUP_494_JSON}" \
      --model27b-json "${M27B_494_JSON}" \
      --out-dir "${RUN_DIR}/p2_gold" \
      --per-model 50 --claude-n 30 --seed 42
    ;;
  p2_export_claude30)
    "${PYTHON_BIN}" -B "${HERE}/p2_export_claude30.py" \
      --blind-jsonl "${RUN_DIR}/p2_gold/blind_dialogues_100.jsonl" \
      --claude-ids-json "${RUN_DIR}/p2_gold/claude_30_ids.json" \
      --out-dir "${RUN_DIR}/p2_claude30_export"
    ;;
  p2_calibrate)
    J_URL="$(ensure_judge_server)"
    U_CSV="${USER_CSV:-${RUN_DIR}/p2_gold/annotations_user_100.csv}"
    C_CSV="${CLAUDE_CSV:-${RUN_DIR}/p2_gold/annotations_claude_30.csv}"
    "${PYTHON_BIN}" -B "${HERE}/p2_calibrate_consistency.py" \
      --gold-dir "${RUN_DIR}/p2_gold" \
      --user-csv "${U_CSV}" \
      --claude-csv "${C_CSV}" \
      --out-dir "${RUN_DIR}/p2_calibrated" \
      --judge-api-base "${J_URL}" \
      --require-math-verify
    ;;
  p3)
    J_URL="$(ensure_judge_server)"
    "${PYTHON_BIN}" -B "${HERE}/p3_rejudge_494.py" \
      --agroup-json "${AGROUP_494_JSON}" \
      --model27b-json "${M27B_494_JSON}" \
      --out-dir "${RUN_DIR}/p3" \
      --judge-api-base "${J_URL}" \
      --require-math-verify
    ;;
  status)
    for f in "${RUN_DIR}/p1/p1_checkup_summary.json" "${RUN_DIR}/p2_calibrated/p2_calibration_report.json" "${RUN_DIR}/p3/p3_rejudge_summary.json"; do
      if [ -f "${f}" ]; then
        echo "FOUND=${f}"
      else
        echo "PENDING=${f}"
      fi
    done
    ;;
  *)
    echo "Usage: $0 --step {p1|p2_sample|p2_export_claude30|p2_calibrate|p3|status} [options]" >&2
    exit 2
    ;;
esac
