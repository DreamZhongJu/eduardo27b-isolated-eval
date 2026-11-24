#!/usr/bin/env bash
# End-to-end Pro 5000 pipeline for T-022 Eduardo-27B author-494 eval:
#   1) Download & SHA-256 verify Eduardo-27b-3bbf17e to /dev/shm/zhujie_a_group/
#   2) Read-only preflight + Agnes single-call ping
#   3) 20-problem smoke (TP=2 Tutor + DP=3 or fallback DP=2 Student, bf16)
#   4) Enforce smoke gate (0 student fails, 100% transfer tested, <=1% Agnes tech fail, <=5% empty)
#   5) Auto-launch full 494 eval and summarize against paper Eduardo-27B baseline.
# Supports --resume on an interrupted --pipe-dir and --status <pipe_dir>.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_27b_pipeline_pro5000.sh [--pipe-dir <dir>] [--student-dp auto|2|3] \
    [--student-port 18971] [--poll-seconds 60] \
    [--judge-concurrency 12] [--eval-judge-concurrency 4] \
    [--skip-download] [--resume]

  run_27b_pipeline_pro5000.sh --status <pipe_dir>
EOF
}

HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${PRO5000_BASE:-${HOME}/bzhang/a_group_pro5000}"
ENV_DIR="${PRO5000_ENV_DIR:-${BASE}/envs/pro5000-fast}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python}"

if [ "${1:-}" = "--status" ]; then
  PDIR="${2:?missing pipe_dir}"
  if [ -f "${PDIR}/pipeline.pid" ]; then
    PPID_VAL="$(cat "${PDIR}/pipeline.pid")"
    if kill -0 "${PPID_VAL}" 2>/dev/null; then
      echo "pipeline_pid=${PPID_VAL} state=RUNNING"
    else
      echo "pipeline_pid=${PPID_VAL} state=EXITED"
    fi
  else
    echo "pipeline_pid=MISSING"
  fi
  if [ -f "${PDIR}/pipeline_state.txt" ]; then
    cat "${PDIR}/pipeline_state.txt"
  fi
  if [ -d "${PDIR}/smoke20" ]; then
    echo "--- smoke20 ---"
    bash "${HERE_DIR}/run_27b_pro5000.sh" --status "${PDIR}/smoke20" || true
  fi
  if [ -f "${PDIR}/smoke20_summary.json" ]; then
    python3 -B - "${PDIR}/smoke20_summary.json" <<'PY'
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
g = d.get("smoke_gate") or {}
print(f"smoke_gate passed={g.get('passed')} reasons={g.get('reasons')}")
PY
  fi
  if [ -d "${PDIR}/full494" ]; then
    echo "--- full494 ---"
    bash "${HERE_DIR}/run_27b_pro5000.sh" --status "${PDIR}/full494" || true
  fi
  if [ -f "${PDIR}/full494_report.txt" ]; then
    echo "--- full494_report ---"
    cat "${PDIR}/full494_report.txt"
  fi
  exit 0
fi

PIPE_DIR=""
STUDENT_DP="auto"
STUDENT_PORT="18971"
POLL_SECONDS="60"
JUDGE_CONCURRENCY="12"
EVAL_JUDGE_CONCURRENCY="4"
SKIP_DOWNLOAD="0"
RESUME="0"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --pipe-dir) PIPE_DIR="$2"; shift 2 ;;
    --student-dp) STUDENT_DP="$2"; shift 2 ;;
    --student-port) STUDENT_PORT="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --judge-concurrency) JUDGE_CONCURRENCY="$2"; shift 2 ;;
    --eval-judge-concurrency) EVAL_JUDGE_CONCURRENCY="$2"; shift 2 ;;
    --skip-download) SKIP_DOWNLOAD="1"; shift ;;
    --resume) RESUME="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [ -z "${PIPE_DIR}" ]; then
  PIPE_DIR="${BASE}/runs/t022_27b_pro5000_$(date +%Y%m%d_%H%M%S)"
fi

unset LD_LIBRARY_PATH HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true
export LD_LIBRARY_PATH="${ENV_DIR}/lib"
export PATH="${ENV_DIR}/bin:${PATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TMPDIR="${BASE}/tmp"
export HF_HOME="${BASE}/cache/hf"
export TRITON_CACHE_DIR="${BASE}/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${BASE}/cache/inductor"
export VLLM_CACHE_ROOT="${BASE}/cache/vllm"
export XDG_CACHE_HOME="${BASE}/cache/xdg"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${XDG_CACHE_HOME}" "${PIPE_DIR}"

if [ -z "${AGNES_API_KEY:-}" ] && [ -s "${HOME}/.secrets/agnes_api_key" ]; then
  export AGNES_API_KEY="$(tr -d '\r\n' < "${HOME}/.secrets/agnes_api_key")"
fi
[ -n "${AGNES_API_KEY:-}" ] || { echo "STOP: AGNES_API_KEY not set" >&2; exit 2; }
[ -z "${GEMINI_API_KEY:-}" ] || { echo "STOP: GEMINI_API_KEY must NOT be set" >&2; exit 2; }

echo "$$" > "${PIPE_DIR}/pipeline.pid"

set_state() {
  printf "stage=%s\nupdated_at=%s\n" "$1" "$(date '+%Y-%m-%d %H:%M:%S')" > "${PIPE_DIR}/pipeline_state.txt"
}

echo "[pipeline] PIPE_DIR=${PIPE_DIR} resume=${RESUME} student_dp=${STUDENT_DP} port=${STUDENT_PORT}"

set_state "1_download_weights"
if [ "${SKIP_DOWNLOAD}" = "1" ]; then
  "${PYTHON_BIN}" -B "${HERE_DIR}/download_27b_pro5000.py" --verify-only
else
  "${PYTHON_BIN}" -B "${HERE_DIR}/download_27b_pro5000.py"
fi

set_state "2_preflight_and_ping"
bash "${HERE_DIR}/preflight_27b_pro5000.sh" | tee "${PIPE_DIR}/preflight.log"
"${PYTHON_BIN}" -B "${HERE_DIR}/agnes_ping_author494.py" | tee "${PIPE_DIR}/agnes_ping.log"

SMOKE_DIR="${PIPE_DIR}/smoke20"
SMOKE_JSON="$(ls -t "${SMOKE_DIR}"/transfer_eval_*.json 2>/dev/null | head -n 1 || true)"
if [ "${RESUME}" = "1" ] && [ -n "${SMOKE_JSON}" ] && [ -f "${PIPE_DIR}/smoke20_summary.json" ]; then
  echo "[pipeline] 3/5 Smoke already completed; checking saved smoke gate..."
else
  set_state "3_smoke20"
  SMOKE_RESUME=()
  if [ "${RESUME}" = "1" ] && [ -f "${SMOKE_DIR}/replay_cache/replay_manifest.json" ] && [ -z "${SMOKE_JSON}" ]; then
    SMOKE_RESUME=(--resume)
  fi
  bash "${HERE_DIR}/run_27b_pro5000.sh" \
    --auto-wait --poll-seconds "${POLL_SECONDS}" \
    --tutor-tp 2 --student-dp "${STUDENT_DP}" --student-port "${STUDENT_PORT}" \
    --num-problems 20 \
    --judge-concurrency "${JUDGE_CONCURRENCY}" --eval-judge-concurrency "${EVAL_JUDGE_CONCURRENCY}" \
    --tag eduardo_27b_author494_pro5000_smoke20 \
    --output-dir "${SMOKE_DIR}" \
    "${SMOKE_RESUME[@]}"
  SMOKE_JSON="$(ls -t "${SMOKE_DIR}"/transfer_eval_*.json 2>/dev/null | head -n 1 || true)"
fi

[ -n "${SMOKE_JSON}" ] && [ -f "${SMOKE_JSON}" ] || { echo "STOP: smoke20 transfer_eval JSON missing" >&2; exit 11; }

set_state "4_smoke_gate"
SMOKE_STATS=()
if [ -f "${SMOKE_DIR}/agnes_window_stats.json" ]; then
  SMOKE_STATS=(--agnes-stats "${SMOKE_DIR}/agnes_window_stats.json")
fi
set +e
"${PYTHON_BIN}" -B "${HERE_DIR}/summarize_author494.py" \
  --paper-baseline eduardo-27b \
  --ours-label "Ours (Pro 5000 + Agnes)" \
  --smoke-gate \
  "${SMOKE_STATS[@]}" \
  --json-out "${PIPE_DIR}/smoke20_summary.json" \
  "${SMOKE_JSON}" | tee "${PIPE_DIR}/smoke20_report.txt"
GATE_RC=$?
set -e
if [ "${GATE_RC}" -ne 0 ]; then
  set_state "FAILED_SMOKE_GATE(rc=${GATE_RC})"
  echo "[pipeline] STOP: smoke gate failed (rc=${GATE_RC}); full 494 will NOT run." >&2
  exit "${GATE_RC}"
fi

FULL_DIR="${PIPE_DIR}/full494"
FULL_JSON="$(ls -t "${FULL_DIR}"/transfer_eval_*.json 2>/dev/null | head -n 1 || true)"
if [ "${RESUME}" = "1" ] && [ -n "${FULL_JSON}" ]; then
  echo "[pipeline] 5/5 Full 494 already has ${FULL_JSON}; summarizing..."
else
  set_state "5_full494"
  FULL_RESUME=()
  if [ "${RESUME}" = "1" ] && [ -f "${FULL_DIR}/replay_cache/replay_manifest.json" ] && [ -z "${FULL_JSON}" ]; then
    FULL_RESUME=(--resume)
  fi
  bash "${HERE_DIR}/run_27b_pro5000.sh" \
    --auto-wait --poll-seconds "${POLL_SECONDS}" \
    --tutor-tp 2 --student-dp "${STUDENT_DP}" --student-port "${STUDENT_PORT}" \
    --num-problems -1 \
    --judge-concurrency "${JUDGE_CONCURRENCY}" --eval-judge-concurrency "${EVAL_JUDGE_CONCURRENCY}" \
    --tag eduardo_27b_author494_pro5000_full494 \
    --output-dir "${FULL_DIR}" \
    "${FULL_RESUME[@]}"
  FULL_JSON="$(ls -t "${FULL_DIR}"/transfer_eval_*.json 2>/dev/null | head -n 1 || true)"
fi

[ -n "${FULL_JSON}" ] && [ -f "${FULL_JSON}" ] || { echo "STOP: full494 transfer_eval JSON missing" >&2; exit 12; }

FULL_STATS=()
if [ -f "${FULL_DIR}/agnes_window_stats.json" ]; then
  FULL_STATS=(--agnes-stats "${FULL_DIR}/agnes_window_stats.json")
fi
"${PYTHON_BIN}" -B "${HERE_DIR}/summarize_author494.py" \
  --paper-baseline eduardo-27b \
  --ours-label "Ours (Pro 5000 + Agnes)" \
  "${FULL_STATS[@]}" \
  --json-out "${PIPE_DIR}/full494_summary.json" \
  "${FULL_JSON}" | tee "${PIPE_DIR}/full494_report.txt"

set_state "PIPELINE_DONE=0"
echo "PIPELINE_DONE=0 pipe_dir=${PIPE_DIR}"
