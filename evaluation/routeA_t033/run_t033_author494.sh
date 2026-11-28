#!/usr/bin/env bash
# T-033 Pipeline 1: Run Route A Step-100 on Author 494 test.parquet (bf16, Agnes AIMD <= 16).
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_t033_author494.sh --tutor-model <path> --output-dir <dir> \
    (--tutor-gpu <id> --student-gpus <ids> | --auto-wait) \
    [--student-dp 3|2] [--num-problems 16|-1] [--max-agnes-window 16] \
    [--judge-concurrency 12] [--eval-judge-concurrency 4] \
    [--student-port 18981] [--poll-seconds 60] [--tag routeA_step100_author494] [--resume]

  run_t033_author494.sh --status <output_dir>
EOF
}

if [ "${1:-}" = "--status" ]; then
  OUT_DIR="${2:?missing output_dir}"
  if [ -f "${OUT_DIR}/runner.pid" ] && kill -0 "$(cat "${OUT_DIR}/runner.pid")" 2>/dev/null; then
    echo "runner_pid=$(cat "${OUT_DIR}/runner.pid") state=RUNNING"
  else
    echo "runner_pid=$(cat "${OUT_DIR}/runner.pid" 2>/dev/null || echo MISSING) state=EXITED"
  fi
  [ -f "${OUT_DIR}/author494_summary.json" ] && cat "${OUT_DIR}/author494_summary.json"
  exit 0
fi

HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE_DIR}/../.." && pwd)"
BASE="${PRO5000_BASE:-${HOME}/bzhang/a_group_pro5000}"
ENV_DIR="${PRO5000_ENV_DIR:-${BASE}/envs/pro5000-fast}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python}"
VLLM_BIN="${VLLM_BIN:-${ENV_DIR}/bin/vllm}"
REPO="${EDUARDO_REPO:-${BASE}/source/EduardoRL}"
TUTOR_MODEL="${BASE}/runs/t026_stage3_formal_20261126_140508/exports/step_100"
STUDENT_MODEL="${STUDENT_MODEL:-/dev/shm/zhujie_a_group/Meta-Llama-3.1-8B-Instruct}"
TEST_PARQUET="${BASE}/data/author494_eval/test.parquet"

TUTOR_GPU=""
STUDENT_GPUS=""
STUDENT_DP="3"
AUTO_WAIT="0"
POLL_SECONDS="60"
OUTPUT_DIR=""
NUM_PROBLEMS="-1"
STUDENT_PORT="18981"
TAG="routeA_step100_author494"
AGNES_API_BASE="https://api.agnes-ai.cn/v1"
AGNES_MODEL="agnes-3.0-flash"
AGNES_KEY_ENV="AGNES_API_KEY"
AGNES_KEY_FILE="${AGNES_KEY_FILE:-${HOME}/.secrets/agnes_api_key}"
JUDGE_CONCURRENCY="12"
EVAL_JUDGE_CONCURRENCY="4"
MAX_AGNES_WINDOW="16"
RESUME="0"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tutor-model) TUTOR_MODEL="$2"; shift 2 ;;
    --tutor-gpu) TUTOR_GPU="$2"; shift 2 ;;
    --student-gpus) STUDENT_GPUS="$2"; shift 2 ;;
    --student-dp) STUDENT_DP="$2"; shift 2 ;;
    --auto-wait) AUTO_WAIT="1"; shift ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --num-problems) NUM_PROBLEMS="$2"; shift 2 ;;
    --student-port) STUDENT_PORT="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --agnes-api-base) AGNES_API_BASE="$2"; shift 2 ;;
    --agnes-model) AGNES_MODEL="$2"; shift 2 ;;
    --agnes-key-env) AGNES_KEY_ENV="$2"; shift 2 ;;
    --judge-concurrency) JUDGE_CONCURRENCY="$2"; shift 2 ;;
    --eval-judge-concurrency) EVAL_JUDGE_CONCURRENCY="$2"; shift 2 ;;
    --max-agnes-window) MAX_AGNES_WINDOW="$2"; shift 2 ;;
    --test-parquet) TEST_PARQUET="$2"; shift 2 ;;
    --resume) RESUME="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

unset LD_LIBRARY_PATH HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy GEMINI_API_KEY || true
export LD_LIBRARY_PATH="${ENV_DIR}/lib"
export PATH="${ENV_DIR}/bin:${PATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 WANDB_MODE=disabled DO_NOT_TRACK=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
export NO_PROXY="127.0.0.1,localhost,::1" no_proxy="127.0.0.1,localhost,::1"
export TMPDIR="${BASE}/tmp" HF_HOME="${BASE}/cache/hf" TRITON_CACHE_DIR="${BASE}/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${BASE}/cache/inductor" VLLM_CACHE_ROOT="${BASE}/cache/vllm" XDG_CACHE_HOME="${BASE}/cache/xdg"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${XDG_CACHE_HOME}"

if [ -z "${!AGNES_KEY_ENV:-}" ] && [ -s "${AGNES_KEY_FILE}" ]; then
  export "${AGNES_KEY_ENV}=$(tr -d '\r\n' < "${AGNES_KEY_FILE}")"
fi
[ -n "${!AGNES_KEY_ENV:-}" ] || { echo "ERROR: ${AGNES_KEY_ENV} missing" >&2; exit 2; }
[ -n "${OUTPUT_DIR}" ] || { echo "ERROR: --output-dir required" >&2; exit 2; }
[ -d "${TUTOR_MODEL}" ] || { echo "ERROR: --tutor-model missing: ${TUTOR_MODEL}" >&2; exit 2; }

if [ "${RESUME}" = "0" ] && [ -e "${OUTPUT_DIR}" ] && [ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null || true)" ]; then
  echo "ERROR: output-dir ${OUTPUT_DIR} non-empty; use --resume or new dir" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"
echo "$$" > "${OUTPUT_DIR}/runner.pid"

TOTAL_GPUS=$((1 + STUDENT_DP))
if [ "${AUTO_WAIT}" = "1" ]; then
  READ_GPUS="$("${PYTHON_BIN}" -B - "${HERE_DIR}" "${TOTAL_GPUS}" "${POLL_SECONDS}" <<'PY' | awk 'NF{line=$0} END{print line}'
import sys
sys.path.insert(0, sys.argv[1])
from common import wait_for_free_gpus
gpus = wait_for_free_gpus(int(sys.argv[2]), poll_seconds=int(sys.argv[3]))
print(f"{gpus[0]} {','.join(str(g) for g in gpus[1:])}")
PY
)"
  TUTOR_GPU="${READ_GPUS%% *}"
  STUDENT_GPUS="${READ_GPUS#* }"
fi
[[ "${TUTOR_GPU}" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid TUTOR_GPU='${TUTOR_GPU}'" >&2; exit 2; }
[[ "${STUDENT_GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "ERROR: invalid STUDENT_GPUS='${STUDENT_GPUS}'" >&2; exit 2; }
echo "[t033_author494] selected tutor_gpu=${TUTOR_GPU} student_gpus=${STUDENT_GPUS}" >&2

RESOLVED_STUDENT="$("${PYTHON_BIN}" -B - "${HERE_DIR}" "${BASE}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from common import resolve_local_model
print(resolve_local_model(Path(sys.argv[2]), "Meta-Llama-3.1-8B-Instruct", ("llama-3.1-8b", "8b-instruct")))
PY
)"

STUDENT_PORT="$("${PYTHON_BIN}" -B - "${HERE_DIR}" "${STUDENT_PORT}" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from common import pick_free_ports
print(pick_free_ports([int(sys.argv[2])])[0])
PY
)"

SERVING_API_KEY="$("${PYTHON_BIN}" -c 'import secrets; print(secrets.token_urlsafe(32))')"
export SERVING_API_KEY
STUDENT_SERVED_NAME="Llama-3.1-8B-Instruct-${USER}-t033-a494"

STUDENT_PID=""
cleanup() {
  set +e
  if [ -n "${STUDENT_PID}" ]; then
    "${PYTHON_BIN}" -B - "${HERE_DIR}" "${STUDENT_PID}" <<'PY' || true
import sys
sys.path.insert(0, sys.argv[1])
from common import kill_pid_tree
kill_pid_tree(int(sys.argv[2]))
PY
    wait "${STUDENT_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="${STUDENT_GPUS}" VLLM_API_KEY="${SERVING_API_KEY}" \
  "${VLLM_BIN}" serve "${RESOLVED_STUDENT}" \
  --host 127.0.0.1 --port "${STUDENT_PORT}" --served-model-name "${STUDENT_SERVED_NAME}" \
  --api-key "${SERVING_API_KEY}" --max-model-len 32768 --dtype bfloat16 \
  --gpu-memory-utilization 0.9 --data-parallel-size "${STUDENT_DP}" --tensor-parallel-size 1 \
  --max-num-seqs 256 --enable-chunked-prefill --enable-prefix-caching \
  > "${OUTPUT_DIR}/student_server.log" 2>&1 &
STUDENT_PID=$!

"${PYTHON_BIN}" -B - "${STUDENT_PID}" "${STUDENT_PORT}" "${OUTPUT_DIR}/student_server.log" <<'PY'
import os, sys, time, urllib.request
from pathlib import Path
pid, port, log_p = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
deadline = time.monotonic() + 600
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        tail = log_p.read_text(encoding="utf-8", errors="replace").splitlines()[-25:] if log_p.is_file() else []
        print("\n".join(tail), file=sys.stderr)
        raise SystemExit("ERROR: student vllm exited before becoming healthy")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as resp:
            if resp.status == 200:
                print(f"[t033_author494] student healthy on 127.0.0.1:{port} (dtype=bfloat16)")
                sys.exit(0)
    except OSError:
        pass
    time.sleep(5)
raise SystemExit("ERROR: student health check timeout")
PY

RESUME_FLAG=()
if [ "${RESUME}" = "1" ] && [ -f "${OUTPUT_DIR}/replay_cache/replay_manifest.json" ]; then
  RESUME_FLAG=(--resume)
fi

set +e
CUDA_VISIBLE_DEVICES="${TUTOR_GPU}" PYTHONPATH="${REPO_ROOT}:${REPO}" \
  "${PYTHON_BIN}" -B "${HERE_DIR}/run_eval_routea_author494.py" \
  --max-agnes-window "${MAX_AGNES_WINDOW}" \
  --student-model-dir "${RESOLVED_STUDENT}" \
  --replay-dir "${OUTPUT_DIR}/replay_cache" \
  --agnes-stats-out "${OUTPUT_DIR}/agnes_window_stats.json" \
  --replay-stats-out "${OUTPUT_DIR}/replay_stats.json" \
  "${RESUME_FLAG[@]}" \
  --model "${TUTOR_MODEL}" \
  --test-parquet "${TEST_PARQUET}" \
  --num-problems "${NUM_PROBLEMS}" \
  --enable-thinking \
  --student-model "${STUDENT_SERVED_NAME}" \
  --student-api-base "http://127.0.0.1:${STUDENT_PORT}/v1" \
  --gemini \
  --gemini-api-base "${AGNES_API_BASE}" \
  --gemini-model "${AGNES_MODEL}" \
  --gemini-api-key-env "${AGNES_KEY_ENV}" \
  --eval-judge-model "${AGNES_MODEL}" \
  --judge-concurrency "${JUDGE_CONCURRENCY}" \
  --eval-judge-concurrency "${EVAL_JUDGE_CONCURRENCY}" \
  --output-dir "${OUTPUT_DIR}" \
  --tag "${TAG}" \
  > "${OUTPUT_DIR}/eval_run.log" 2>&1
RC=$?
set -e

tail -n 15 "${OUTPUT_DIR}/eval_run.log" || true
exit "${RC}"
