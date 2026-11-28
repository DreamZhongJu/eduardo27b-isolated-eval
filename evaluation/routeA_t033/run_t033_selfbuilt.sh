#!/usr/bin/env bash
# T-033 Pipeline 2: Route A Step-100 on Self-built 3-tier (Tutor & Student both bfloat16).
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_t033_selfbuilt.sh --tutor-model <path> --output-dir <dir> \
    (--tutor-gpu <id> --student-gpu <id> | --auto-wait) \
    [--smoke-total 16|0] [--concurrency 8] [--agnes-workers 16] \
    [--gate-first-n 0] [--teacher-port 18711] [--student-port 18712] \
    [--poll-seconds 60] [--skip-grading] [--skip-author-rule]

  run_t033_selfbuilt.sh --grade-only <output_dir> [--smoke-total 16|0] [--agnes-workers 16]
  run_t033_selfbuilt.sh --status <output_dir>
EOF
}

if [ "${1:-}" = "--status" ]; then
  OUT_DIR="${2:?missing output_dir}"
  if [ -f "${OUT_DIR}/runner.pid" ] && kill -0 "$(cat "${OUT_DIR}/runner.pid")" 2>/dev/null; then
    echo "runner_pid=$(cat "${OUT_DIR}/runner.pid") state=RUNNING"
  else
    echo "runner_pid=$(cat "${OUT_DIR}/runner.pid" 2>/dev/null || echo MISSING) state=EXITED"
  fi
  [ -f "${OUT_DIR}/selfbuilt_summary.json" ] && cat "${OUT_DIR}/selfbuilt_summary.json"
  exit 0
fi

HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE_DIR}/../.." && pwd)"
BASE="${PRO5000_BASE:-${HOME}/bzhang/a_group_pro5000}"
ENV_DIR="${PRO5000_ENV_DIR:-${BASE}/envs/pro5000-fast}"
PY_BIN="${PY_BIN:-${ENV_DIR}/bin/python}"
VLLM_BIN="${VLLM_BIN:-${ENV_DIR}/bin/vllm}"
TUTOR_MODEL="${BASE}/runs/t026_stage3_formal_20261126_140508/exports/step_100"
BUNDLE_DIR="${BUNDLE_DIR:-${BASE}/t021_bundle}"
AGNES_KEY_FILE="${AGNES_KEY_FILE:-${HOME}/.secrets/agnes_api_key}"

TUTOR_GPU=""
STUDENT_GPU=""
AUTO_WAIT="0"
POLL_SECONDS="60"
OUTPUT_DIR=""
SMOKE_TOTAL="0"
CONCURRENCY="8"
AGNES_WORKERS="${AGNES_WORKERS:-16}"
GATE_FIRST_N="0"
TEACHER_PORT="18711"
STUDENT_PORT="18712"
SKIP_GRADING="0"
SKIP_AUTHOR_RULE="0"
GRADE_ONLY_DIR=""
SHARD_INDEX="0"
NUM_SHARDS="1"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tutor-model) TUTOR_MODEL="$2"; shift 2 ;;
    --tutor-gpu) TUTOR_GPU="$2"; shift 2 ;;
    --student-gpu) STUDENT_GPU="$2"; shift 2 ;;
    --auto-wait) AUTO_WAIT="1"; shift ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --bundle-dir) BUNDLE_DIR="$2"; shift 2 ;;
    --smoke-total) SMOKE_TOTAL="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --agnes-workers) AGNES_WORKERS="$2"; shift 2 ;;
    --gate-first-n) GATE_FIRST_N="$2"; shift 2 ;;
    --teacher-port) TEACHER_PORT="$2"; shift 2 ;;
    --student-port) STUDENT_PORT="$2"; shift 2 ;;
    --skip-grading) SKIP_GRADING="1"; shift ;;
    --skip-author-rule) SKIP_AUTHOR_RULE="1"; shift ;;
    --grade-only) GRADE_ONLY_DIR="$2"; shift 2 ;;
    --shard-index) SHARD_INDEX="$2"; shift 2 ;;
    --num-shards) NUM_SHARDS="$2"; shift 2 ;;
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
export TMPDIR="${BASE}/tmp" HF_HOME="${BASE}/cache/hf" TRITON_CACHE_DIR="${BASE}/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${BASE}/cache/inductor" VLLM_CACHE_ROOT="${BASE}/cache/vllm" XDG_CACHE_HOME="${BASE}/cache/xdg"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${XDG_CACHE_HOME}"

if [ -n "${GRADE_ONLY_DIR}" ]; then
  test -s "${AGNES_KEY_FILE}"
  "${PY_BIN}" -B "${HERE_DIR}/run_post_eval_routea.py" \
    --repo "${REPO_ROOT}" --bundle-dir "${BUNDLE_DIR}" --out-root "${GRADE_ONLY_DIR}" \
    --key-file "${AGNES_KEY_FILE}" --smoke-total "${SMOKE_TOTAL}" \
    --workers "${AGNES_WORKERS}" --run-grading \
    $( [ "${SKIP_AUTHOR_RULE}" = "0" ] && echo "--run-author-rule" )
  exit $?
fi

[ -n "${OUTPUT_DIR}" ] || { echo "ERROR: --output-dir required" >&2; exit 2; }
[ -d "${TUTOR_MODEL}" ] || { echo "ERROR: --tutor-model missing: ${TUTOR_MODEL}" >&2; exit 2; }
if [ "${SKIP_GRADING}" = "0" ] && [ ! -s "${AGNES_KEY_FILE}" ] && [ -z "${AGNES_API_KEY:-}" ]; then
  echo "ERROR: Agnes key missing" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
echo "$$" > "${OUTPUT_DIR}/runner.pid"

RESOLVED_STUDENT="$("${PY_BIN}" -B - "${HERE_DIR}" "${BASE}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from common import resolve_local_model
print(resolve_local_model(Path(sys.argv[2]), "Meta-Llama-3.1-8B-Instruct", ("llama-3.1-8b", "8b-instruct")))
PY
)"

if [ "${AUTO_WAIT}" = "1" ]; then
  READ_GPUS="$("${PY_BIN}" -B - "${HERE_DIR}" "${POLL_SECONDS}" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from common import wait_for_free_gpus
gpus = wait_for_free_gpus(2, poll_seconds=int(sys.argv[2]))
print(f"{gpus[0]} {gpus[1]}")
PY
)"
  TUTOR_GPU="${READ_GPUS%% *}"
  STUDENT_GPU="${READ_GPUS#* }"
fi

PORTS="$("${PY_BIN}" -B - "${HERE_DIR}" "${TEACHER_PORT}" "${STUDENT_PORT}" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from common import pick_free_ports
p = pick_free_ports([int(sys.argv[2]), int(sys.argv[3])])
print(f"{p[0]} {p[1]}")
PY
)"
TEACHER_PORT="${PORTS%% *}"
STUDENT_PORT="${PORTS#* }"

SERVING_API_KEY="loopback-only"
STUDENT_PID=""
TEACHER_PID=""

cleanup() {
  set +e
  for pid in "${TEACHER_PID}" "${STUDENT_PID}"; do
    if [ -n "${pid}" ]; then
      "${PY_BIN}" -B - "${HERE_DIR}" "${pid}" <<'PY' || true
import sys
sys.path.insert(0, sys.argv[1])
from common import kill_pid_tree
kill_pid_tree(int(sys.argv[2]))
PY
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

# Student MUST use --dtype bfloat16 per T-033 & 工人须知第 9 条.
CUDA_VISIBLE_DEVICES="${STUDENT_GPU}" VLLM_API_KEY="${SERVING_API_KEY}" \
  "${VLLM_BIN}" serve "${RESOLVED_STUDENT}" \
  --host 127.0.0.1 --port "${STUDENT_PORT}" --served-model-name "Llama-3.1-8B-Instruct" \
  --api-key "${SERVING_API_KEY}" --dtype bfloat16 --max-model-len 20000 \
  --gpu-memory-utilization 0.90 --seed 42 \
  > "${OUTPUT_DIR}/student_server.log" 2>&1 &
STUDENT_PID=$!

CUDA_VISIBLE_DEVICES="${TUTOR_GPU}" VLLM_API_KEY="${SERVING_API_KEY}" \
  "${VLLM_BIN}" serve "${TUTOR_MODEL}" \
  --host 127.0.0.1 --port "${TEACHER_PORT}" --served-model-name "RouteA-step100" \
  --api-key "${SERVING_API_KEY}" --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 20000 --gpu-memory-utilization 0.85 --max-num-seqs 8 --enforce-eager \
  --limit-mm-per-prompt '{"image":0,"video":0}' --seed 42 \
  > "${OUTPUT_DIR}/teacher_server.log" 2>&1 &
TEACHER_PID=$!

"${PY_BIN}" -B - "${STUDENT_PID}" "${STUDENT_PORT}" "${TEACHER_PID}" "${TEACHER_PORT}" "${OUTPUT_DIR}/student_runtime.json" <<'PY'
import importlib.metadata as md, json, os, sys, time, urllib.request
from pathlib import Path
import torch, transformers, vllm, openai

for label, pid, port in (("student", int(sys.argv[1]), int(sys.argv[2])), ("teacher", int(sys.argv[3]), int(sys.argv[4]))):
    deadline = time.monotonic() + 600
    while True:
        os.kill(pid, 0)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as resp:
                if resp.status == 200:
                    print(f"[t033_selfbuilt] {label} healthy on 127.0.0.1:{port} (dtype=bfloat16)")
                    break
        except OSError:
            pass
        if time.monotonic() >= deadline:
            raise SystemExit(f"ERROR: {label} health timeout")
        time.sleep(5)

doc = {
    "student_dtype": "bfloat16",
    "tutor_dtype": "bfloat16",
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "vllm": vllm.__version__,
    "transformers": transformers.__version__,
    "openai": openai.__version__,
    "math_verify": md.version("math-verify"),
}
Path(sys.argv[5]).write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

set +e
"${PY_BIN}" -B "${HERE_DIR}/run_post_eval_routea.py" \
  --repo "${REPO_ROOT}" --bundle-dir "${BUNDLE_DIR}" --out-root "${OUTPUT_DIR}" \
  --teacher-model "${TUTOR_MODEL}" --teacher-served-name "RouteA-step100" \
  --teacher-url "http://127.0.0.1:${TEACHER_PORT}/v1" \
  --student-url "http://127.0.0.1:${STUDENT_PORT}/v1" \
  --key-file "${AGNES_KEY_FILE}" --smoke-total "${SMOKE_TOTAL}" \
  --shard-index "${SHARD_INDEX}" --num-shards "${NUM_SHARDS}" \
  --concurrency "${CONCURRENCY}" --gate-first-n "${GATE_FIRST_N}" \
  --workers "${AGNES_WORKERS}" --run-generation \
  $( [ "${SKIP_AUTHOR_RULE}" = "0" ] && echo "--run-author-rule" ) \
  $( [ "${SKIP_GRADING}" = "0" ] && echo "--run-grading" ) \
  --export-summary \
  > "${OUTPUT_DIR}/eval_run.log" 2>&1
RC=$?
set -e

tail -n 15 "${OUTPUT_DIR}/eval_run.log" || true
exit "${RC}"
