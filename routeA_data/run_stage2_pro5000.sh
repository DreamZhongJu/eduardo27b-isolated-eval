#!/usr/bin/env bash
# T-023 Stage 2 Pro 5000 Entry Script (Card v3 / Inbox #3).
# Runs:
#   1. Preflight (package pins math-verify==0.8.0, model SHA-256 locks, input SHA-256, unit tests in pro5000-fast)
#   2. GPU Phase (wait 3-4 idle GPUs per T-020 rule -> launch bf16 Student DP=1..2 + bf16 Qwen3.6-27B TP=2 ->
#      k=16 rollouts + canonical hybrid_grader.py + dev 27B audit -> stop servers & release GPUs immediately)
#   3. Agnes Dev Phase (CPU-only, 0 GPUs -> 100 dev sources / 178 variants / <=2,848 calls, AIMD 1..16, >1% gate)
#   4. Summarize Phase (write private routeA_stage2_measured.jsonl + stage2_manifest.json + sanitized summary)
set -euo pipefail
umask 077

HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${HERE_DIR}/.." && pwd)"
BASE="${PRO5000_BASE:-${HOME}/bzhang/a_group_pro5000}"
ENV_DIR="${PRO5000_ENV_DIR:-${BASE}/envs/pro5000-fast}"
EDUARDO_REPO="${EDUARDO_REPO:-${BASE}/source/EduardoRL}"
STUDENT_MODEL="${STUDENT_MODEL:-/dev/shm/zhujie_a_group/Meta-Llama-3.1-8B-Instruct}"
JUDGE_MODEL="${JUDGE_MODEL:-/dev/shm/zhujie_a_group/Qwen3.6-27B}"

RUN_DIR=""
DATA_JSONL=""
EXPECTED_DATA_SHA256="3bab0292bb6d2393a4f64cf8bf4dad282499b0bf395905c52dac9559b5f4f333"
RESUME=0
STATUS_DIR=""
POLL_SECONDS=60
STUDENT_PORT=18981
JUDGE_PORT=18982
STUDENT_CONCURRENCY=64
VARIANT_CONCURRENCY=16

while [ $# -gt 0 ]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --data-jsonl) DATA_JSONL="$2"; shift 2 ;;
    --expected-data-sha256) EXPECTED_DATA_SHA256="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --status) STATUS_DIR="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --student-port) STUDENT_PORT="$2"; shift 2 ;;
    --judge-port) JUDGE_PORT="$2"; shift 2 ;;
    --student-concurrency) STUDENT_CONCURRENCY="$2"; shift 2 ;;
    --variant-concurrency) VARIANT_CONCURRENCY="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ -n "${STATUS_DIR}" ]; then
  echo "=== T-023 STAGE 2 STATUS (${STATUS_DIR}) ==="
  if [ -f "${STATUS_DIR}/pipeline.pid" ] && kill -0 "$(cat "${STATUS_DIR}/pipeline.pid")" 2>/dev/null; then
    echo "PIPELINE_PROCESS=RUNNING (pid=$(cat "${STATUS_DIR}/pipeline.pid"))"
  else
    echo "PIPELINE_PROCESS=NOT_RUNNING"
  fi
  [ -f "${STATUS_DIR}/preflight_ok.flag" ] && echo "PREFLIGHT=OK" || echo "PREFLIGHT=PENDING"
  [ -f "${STATUS_DIR}/gpu_phase_done.flag" ] && echo "GPU_PHASE=DONE" || echo "GPU_PHASE=PENDING_OR_RUNNING"
  [ -f "${STATUS_DIR}/agnes_phase_done.flag" ] && echo "AGNES_PHASE=DONE" || echo "AGNES_PHASE=PENDING_OR_RUNNING"
  [ -f "${STATUS_DIR}/STAGE2_DONE.flag" ] && echo "STAGE2_DONE=0" || echo "STAGE2_DONE=PENDING"
  [ -f "${STATUS_DIR}/gpu_phase_status.json" ] && echo "gpu_phase_status: $(cat "${STATUS_DIR}/gpu_phase_status.json")"
  [ -f "${STATUS_DIR}/agnes_phase_status.json" ] && echo "agnes_phase_status: $(cat "${STATUS_DIR}/agnes_phase_status.json")"
  [ -f "${STATUS_DIR}/stage2_summary.txt" ] && cat "${STATUS_DIR}/stage2_summary.txt"
  exit 0
fi

[ -n "${RUN_DIR}" ] || { echo "ERROR: --run-dir required" >&2; exit 2; }
[ -n "${DATA_JSONL}" ] || { echo "ERROR: --data-jsonl required" >&2; exit 2; }

unset GEMINI_API_KEY HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy CUDA_VISIBLE_DEVICES || true
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TMPDIR="${BASE}/tmp"
export HF_HOME="${BASE}/cache/hf"
export TRITON_CACHE_DIR="${BASE}/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${BASE}/cache/inductor"
export VLLM_CACHE_ROOT="${BASE}/cache/vllm"
export XDG_CACHE_HOME="${BASE}/cache/xdg"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${XDG_CACHE_HOME}" "${RUN_DIR}"
chmod 700 "${RUN_DIR}"

if [ -z "${AGNES_API_KEY:-}" ] && [ -s "${HOME}/.secrets/agnes_api_key" ]; then
  export AGNES_API_KEY="$(tr -d '\r\n' < "${HOME}/.secrets/agnes_api_key")"
fi
[ -n "${AGNES_API_KEY:-}" ] || { echo "STOP: AGNES_API_KEY 未设" >&2; exit 2; }

export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${CODE_ROOT}:${EDUARDO_REPO}"
PYTHON_BIN="${ENV_DIR}/bin/python"

echo "$$" > "${RUN_DIR}/pipeline.pid"

STUDENT_PID=""
JUDGE_PID=""

kill_tree() {
  local root_pid="$1"
  [ -n "${root_pid}" ] || return 0
  "${PYTHON_BIN}" -B - "${root_pid}" <<'PY' || true
import os, signal, subprocess, sys, time
root = int(sys.argv[1])
def descendants(pid):
    out = []
    try:
        ps = subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True)
    except Exception:
        return [pid]
    children = {}
    for line in ps.splitlines():
        parts = line.split()
        if len(parts) == 2:
            p, pp = int(parts[0]), int(parts[1])
            children.setdefault(pp, []).append(p)
    stack = [pid]
    while stack:
        cur = stack.pop()
        out.append(cur)
        stack.extend(children.get(cur, []))
    return list(reversed(out))
for sig in (signal.SIGTERM, signal.SIGKILL):
    pids = descendants(root)
    alive = False
    for p in pids:
        try:
            os.kill(p, sig)
            alive = True
        except OSError:
            pass
    if not alive:
        break
    time.sleep(3)
PY
}

stop_servers() {
  set +e
  if [ -n "${STUDENT_PID}" ]; then
    echo "[t023] Stopping student vLLM (pid=${STUDENT_PID})..."
    kill_tree "${STUDENT_PID}"
    wait "${STUDENT_PID}" 2>/dev/null || true
    STUDENT_PID=""
  fi
  if [ -n "${JUDGE_PID}" ]; then
    echo "[t023] Stopping judge 27B vLLM (pid=${JUDGE_PID})..."
    kill_tree "${JUDGE_PID}"
    wait "${JUDGE_PID}" 2>/dev/null || true
    JUDGE_PID=""
  fi
}

trap stop_servers EXIT INT TERM

echo "=== [1/4] Preflight & Unit Tests in pro5000-fast ==="
bash "${HERE_DIR}/preflight_stage2_pro5000.sh" "${DATA_JSONL}" "${EXPECTED_DATA_SHA256}" \
  | tee "${RUN_DIR}/preflight.log"
echo "ok" > "${RUN_DIR}/preflight_ok.flag"

if [ ! -f "${RUN_DIR}/gpu_phase_done.flag" ]; then
  echo "=== [2/4] Waiting for 3-4 Idle Pro 5000 GPUs (Judge TP=2 + Student DP=1..2) ==="
  GPU_JSON="${RUN_DIR}/selected_gpus.json"
  "${PYTHON_BIN}" -B "${HERE_DIR}/gpu_wait_pro5000.py" \
    --num-gpus 4 --fallback-min-gpus 3 --min-free-mib 70000 \
    --confirm-seconds 5 --poll-seconds "${POLL_SECONDS}" \
    --out-json "${GPU_JSON}"

  JUDGE_GPUS="$("${PYTHON_BIN}" -B -c "import json; d=json.load(open('${GPU_JSON}')); print(','.join(map(str, d['judge_gpus'])))")"
  STUDENT_GPUS="$("${PYTHON_BIN}" -B -c "import json; d=json.load(open('${GPU_JSON}')); print(','.join(map(str, d['student_gpus'])))")"
  STUDENT_DP="$("${PYTHON_BIN}" -B -c "import json; d=json.load(open('${GPU_JSON}')); print(d['student_dp'])")"
  echo "[t023] Acquired GPUs: judge=${JUDGE_GPUS} (TP=2), student=${STUDENT_GPUS} (DP=${STUDENT_DP})"

  RESOLVED_STUDENT="$("${PYTHON_BIN}" -B -c "from pathlib import Path; p=Path('${STUDENT_MODEL}'); q=Path('${BASE}/models/Meta-Llama-3.1-8B-Instruct'); print((p if (p/'config.json').is_file() else q).resolve())")"
  RESOLVED_JUDGE="$("${PYTHON_BIN}" -B -c "from pathlib import Path; p=Path('${JUDGE_MODEL}'); q=Path('${BASE}/models/Qwen3.6-27B'); print((p if (p/'config.json').is_file() else q).resolve())")"

  SERVING_API_KEY="$("${PYTHON_BIN}" -B -c 'import secrets; print(secrets.token_urlsafe(32))')"
  export SERVING_API_KEY
  STUDENT_SERVED_NAME="Llama-3.1-8B-Instruct-t023"
  JUDGE_SERVED_NAME="judge-local"

  echo "[t023] Starting Student vLLM (bf16, DP=${STUDENT_DP}, port=${STUDENT_PORT})..."
  CUDA_VISIBLE_DEVICES="${STUDENT_GPUS}" VLLM_API_KEY="${SERVING_API_KEY}" \
    "${PYTHON_BIN}" -B -m vllm.entrypoints.openai.api_server \
    --model "${RESOLVED_STUDENT}" \
    --served-model-name "${STUDENT_SERVED_NAME}" \
    --host 127.0.0.1 \
    --port "${STUDENT_PORT}" \
    --api-key "${SERVING_API_KEY}" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --max-num-seqs 256 \
    --tensor-parallel-size 1 \
    --data-parallel-size "${STUDENT_DP}" \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    > "${RUN_DIR}/student_vllm.log" 2>&1 &
  STUDENT_PID=$!

  echo "[t023] Starting Judge Qwen3.6-27B vLLM (bf16, TP=2, port=${JUDGE_PORT})..."
  CUDA_VISIBLE_DEVICES="${JUDGE_GPUS}" VLLM_API_KEY="${SERVING_API_KEY}" \
    "${PYTHON_BIN}" -B -m vllm.entrypoints.openai.api_server \
    --model "${RESOLVED_JUDGE}" \
    --served-model-name "${JUDGE_SERVED_NAME}" \
    --host 127.0.0.1 \
    --port "${JUDGE_PORT}" \
    --api-key "${SERVING_API_KEY}" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --max-num-seqs 256 \
    --tensor-parallel-size 2 \
    --data-parallel-size 1 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    > "${RUN_DIR}/judge_vllm.log" 2>&1 &
  JUDGE_PID=$!

  # Wait for both /health endpoints (up to 10 min)
  "${PYTHON_BIN}" -B - "${STUDENT_PID}" "${STUDENT_PORT}" "${JUDGE_PID}" "${JUDGE_PORT}" <<'PY'
import os, sys, time, urllib.request
s_pid, s_port, j_pid, j_port = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
deadline = time.monotonic() + 600
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
s_ok = False
j_ok = False
while time.monotonic() < deadline and not (s_ok and j_ok):
    for name, pid in (("student", s_pid), ("judge", j_pid)):
        try:
            os.kill(pid, 0)
        except OSError:
            raise SystemExit(f"ERROR: {name} vLLM exited before becoming healthy")
    if not s_ok:
        try:
            with opener.open(f"http://127.0.0.1:{s_port}/health", timeout=3) as r:
                if r.status == 200:
                    s_ok = True
                    print(f"[health] student port {s_port} READY", flush=True)
        except Exception:
            pass
    if not j_ok:
        try:
            with opener.open(f"http://127.0.0.1:{j_port}/health", timeout=3) as r:
                if r.status == 200:
                    j_ok = True
                    print(f"[health] judge port {j_port} READY", flush=True)
        except Exception:
            pass
    if not (s_ok and j_ok):
        time.sleep(3)
if not (s_ok and j_ok):
    raise SystemExit(f"ERROR: vLLM health timeout (student={s_ok}, judge={j_ok})")
PY

  echo "[t023] Running GPU Phase (Student k=16 bf16 + Canonical HybridGrader + Dev 27B Audit)..."
  "${PYTHON_BIN}" -B -m routeA_data.stage2_runner \
    --phase gpu \
    --data-jsonl "${DATA_JSONL}" \
    --expected-input-sha256 "${EXPECTED_DATA_SHA256}" \
    --run-dir "${RUN_DIR}" \
    --student-api-base "http://127.0.0.1:${STUDENT_PORT}/v1" \
    --student-model "${STUDENT_SERVED_NAME}" \
    --judge-api-base "http://127.0.0.1:${JUDGE_PORT}/v1" \
    --judge-model "${JUDGE_SERVED_NAME}" \
    --k-samples 16 \
    --student-concurrency "${STUDENT_CONCURRENCY}" \
    --judge-max-window 16 \
    --variant-concurrency "${VARIANT_CONCURRENCY}"

  stop_servers
  echo "[t023] GPU Phase complete; GPUs released."
else
  echo "=== [2/4] GPU Phase already complete (skipping on --resume) ==="
fi

if [ ! -f "${RUN_DIR}/agnes_phase_done.flag" ]; then
  echo "=== [3/4] Dev-Set Agnes Consistency Check (CPU-only, 0 GPUs, AIMD 1..16) ==="
  "${PYTHON_BIN}" -B -m routeA_data.stage2_runner \
    --phase agnes \
    --data-jsonl "${DATA_JSONL}" \
    --expected-input-sha256 "${EXPECTED_DATA_SHA256}" \
    --run-dir "${RUN_DIR}" \
    --agnes-max-window 16
else
  echo "=== [3/4] Agnes Dev Phase already complete (skipping on --resume) ==="
fi

echo "=== [4/4] Summarize, Write Enriched JSONL, and Manifest ==="
"${PYTHON_BIN}" -B -m routeA_data.stage2_runner \
  --phase summarize \
  --data-jsonl "${DATA_JSONL}" \
  --expected-input-sha256 "${EXPECTED_DATA_SHA256}" \
  --run-dir "${RUN_DIR}"

echo "T023_STAGE2_PIPELINE_DONE=0 RUN_DIR=${RUN_DIR}"
