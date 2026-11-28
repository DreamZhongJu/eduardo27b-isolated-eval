#!/usr/bin/env bash
# Launch 3-GPU student vLLM server on Pro 5000 and run 1-round untutored control (ctl)
# via unmodified eduardo.create_transfer_dataset._solve_rate. Aggregate output only.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_control_author494_pro5000.sh (--student-gpus <id1,id2,id3> | --auto-wait) \
    --output-dir <new_dir> [--num-problems -1] [--student-port 18961] \
    [--poll-seconds 60] [--tag pro5000_author494_control] \
    [--k 16] [--student-concurrency 64] [--row-concurrency 16] \
    [--test-parquet <path>]

  run_control_author494_pro5000.sh --status <output_dir>
EOF
}

if [ "${1:-}" = "--status" ]; then
  OUT_DIR="${2:?missing output_dir}"
  if [ -f "${OUT_DIR}/runner.pid" ]; then
    RPID="$(cat "${OUT_DIR}/runner.pid")"
    if kill -0 "${RPID}" 2>/dev/null; then
      echo "runner_pid=${RPID} state=RUNNING"
    else
      echo "runner_pid=${RPID} state=EXITED"
    fi
  else
    echo "runner_pid=MISSING"
  fi
  if [ -f "${OUT_DIR}/wait_status.json" ]; then
    python3 -B - "${OUT_DIR}/wait_status.json" <<'PY'
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"wait_state={d.get('state')} polls={d.get('polls')} selected={d.get('selected_gpus')} eligible={d.get('eligible_gpus')} updated_at={d.get('updated_at')}")
PY
  fi
  if [ -f "${OUT_DIR}/control_progress.json" ]; then
    python3 -B - "${OUT_DIR}/control_progress.json" <<'PY'
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"control_round={d.get('round')} done={d.get('done')}/{d.get('total')} updated_at={d.get('updated_at')}")
PY
  fi
  if [ -f "${OUT_DIR}/control_run.log" ]; then
    python3 -B - "${OUT_DIR}/control_run.log" <<'PY'
import re, sys
from pathlib import Path
safe_re = re.compile(r"^(?:\[control\] |Control results written to: )")
for line in Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines():
    if safe_re.match(line):
        print(" ", line)
PY
  fi
  if [ -f "${OUT_DIR}/gpu_watch.log" ]; then
    ALERTS=$(grep -c "NEW_HOST_PID_ALERT" "${OUT_DIR}/gpu_watch.log" || true)
    echo "gpu_watch_new_host_pid_alerts=${ALERTS}"
  fi
  exit 0
fi

BASE="${PRO5000_BASE:-${HOME}/bzhang/a_group_pro5000}"
ENV_DIR="${PRO5000_ENV_DIR:-${BASE}/envs/pro5000-fast}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python}"
VLLM_BIN="${VLLM_BIN:-${ENV_DIR}/bin/vllm}"
REPO="${EDUARDO_REPO:-${BASE}/source/EduardoRL}"
STUDENT_MODEL="${STUDENT_MODEL:-/dev/shm/zhujie_a_group/Meta-Llama-3.1-8B-Instruct}"
STUDENT_SYMLINK="${STUDENT_SYMLINK:-${BASE}/models/Meta-Llama-3.1-8B-Instruct}"
TEST_PARQUET="${BASE}/data/author494_eval/test.parquet"

STUDENT_GPUS=""
AUTO_WAIT="0"
POLL_SECONDS="60"
OUTPUT_DIR=""
NUM_PROBLEMS="-1"
STUDENT_PORT="18961"
TAG="pro5000_author494_control"
K_SAMPLES="16"
STUDENT_CONCURRENCY="64"
ROW_CONCURRENCY="16"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --student-gpus) STUDENT_GPUS="$2"; shift 2 ;;
    --auto-wait) AUTO_WAIT="1"; shift ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --num-problems) NUM_PROBLEMS="$2"; shift 2 ;;
    --student-port) STUDENT_PORT="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --k) K_SAMPLES="$2"; shift 2 ;;
    --student-concurrency) STUDENT_CONCURRENCY="$2"; shift 2 ;;
    --row-concurrency) ROW_CONCURRENCY="$2"; shift 2 ;;
    --test-parquet) TEST_PARQUET="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

unset LD_LIBRARY_PATH HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true
export LD_LIBRARY_PATH="${ENV_DIR}/lib"
export PATH="${ENV_DIR}/bin:${PATH}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export WANDB_MODE=disabled
export DO_NOT_TRACK=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export USER="${USER:-root}"
export PYTHONUNBUFFERED=1
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="127.0.0.1,localhost,::1"
export TMPDIR="${BASE}/tmp"
export HF_HOME="${BASE}/cache/hf"
export TRITON_CACHE_DIR="${BASE}/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${BASE}/cache/inductor"
export VLLM_CACHE_ROOT="${BASE}/cache/vllm"
export XDG_CACHE_HOME="${BASE}/cache/xdg"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${XDG_CACHE_HOME}"

if [ -z "${OUTPUT_DIR}" ]; then
  echo "ERROR: --output-dir is required" >&2
  usage
  exit 2
fi
if [ "${AUTO_WAIT}" = "0" ] && [ -z "${STUDENT_GPUS}" ]; then
  echo "ERROR: provide --auto-wait OR --student-gpus" >&2
  usage
  exit 2
fi
if [ "${AUTO_WAIT}" = "1" ] && [ -n "${STUDENT_GPUS}" ]; then
  echo "ERROR: --auto-wait cannot be combined with --student-gpus" >&2
  exit 2
fi
if [ -n "${GEMINI_API_KEY:-}" ]; then
  echo "ERROR: GEMINI_API_KEY must NOT be set" >&2
  exit 2
fi
if [ ! -f "${TEST_PARQUET}" ]; then
  echo "ERROR: test.parquet not found: ${TEST_PARQUET}" >&2
  exit 2
fi
if [ -e "${OUTPUT_DIR}" ] && [ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null || true)" ]; then
  echo "ERROR: output-dir ${OUTPUT_DIR} already exists and is non-empty; choose a new dir" >&2
  exit 2
fi

"${PYTHON_BIN}" -B - "${TEST_PARQUET}" <<'PY'
import pandas as pd, sys
df = pd.read_parquet(sys.argv[1])
if len(df) != 494:
    raise SystemExit(f"ERROR: expected 494 rows in test.parquet, got {len(df)}")
n_tp = sum(1 for r in df["extra_info"] if isinstance(r, dict) and r.get("transfer_problem") and r.get("transfer_answer"))
if n_tp != 494:
    raise SystemExit(f"ERROR: expected 494/494 transfer variants, got {n_tp}")
print("[control] verified test.parquet rows=494 transfer_valid=494")
PY

RESOLVED_STUDENT="$("${PYTHON_BIN}" -B - "${STUDENT_MODEL}" "${STUDENT_SYMLINK}" <<'PY'
import sys
from pathlib import Path
for raw in sys.argv[1:3]:
    p = Path(raw)
    if p.exists() and (p / "config.json").is_file():
        print(p.resolve(strict=True))
        sys.exit(0)
shm = Path("/dev/shm/zhujie_a_group")
if shm.is_dir():
    for cfg in sorted(shm.rglob("config.json")):
        if "llama" in str(cfg).lower():
            print(cfg.parent.resolve(strict=True))
            sys.exit(0)
raise SystemExit("ERROR: student Meta-Llama-3.1-8B-Instruct not found under /dev/shm/zhujie_a_group")
PY
)"

mkdir -p "${OUTPUT_DIR}"
echo "$$" > "${OUTPUT_DIR}/runner.pid"

HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while true; do
  if [ "${AUTO_WAIT}" = "1" ]; then
    echo "[control] Waiting for 3 idle Pro 5000 GPUs (>=70000 MiB free, 0 compute apps)..."
    STUDENT_GPUS="$("${PYTHON_BIN}" -B "${HERE_DIR}/wait_gpus_author494_pro5000.py" \
      --num-gpus 3 \
      --status-file "${OUTPUT_DIR}/wait_status.json" \
      --log-file "${OUTPUT_DIR}/wait.log" \
      --port "${STUDENT_PORT}" \
      --min-free-mib 70000 \
      --poll-seconds "${POLL_SECONDS}")"
  fi

  set +e
  "${PYTHON_BIN}" -B - "${STUDENT_GPUS}" "${STUDENT_PORT}" "${OUTPUT_DIR}/gpu_preflight.log" <<'PY'
import socket, subprocess, sys
from pathlib import Path

student_gpus = [int(x) for x in sys.argv[1].split(",") if x.strip()]
port = int(sys.argv[2])
log_path = Path(sys.argv[3])

if len(student_gpus) != 3 or len(set(student_gpus)) != 3:
    raise SystemExit("ERROR: must provide 3 distinct student GPUs")

s = socket.socket()
try:
    s.bind(("127.0.0.1", port))
except OSError as exc:
    raise SystemExit(f"ERROR: port 127.0.0.1:{port} is already in use: {exc}")
finally:
    s.close()

gpu_csv = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.total", "--format=csv,noheader,nounits"],
    text=True,
)
gpus = {}
uuid_to_idx = {}
for line in gpu_csv.strip().splitlines():
    idx, uuid, free, total = [x.strip() for x in line.split(",")]
    gpus[int(idx)] = {"uuid": uuid, "free": int(free), "total": int(total)}
    uuid_to_idx[uuid] = int(idx)

apps_csv = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
    text=True,
).strip()
busy = {idx: [] for idx in gpus}
if apps_csv:
    for line in apps_csv.splitlines():
        uuid, pid, mem = [x.strip() for x in line.split(",")]
        if uuid in uuid_to_idx:
            busy[uuid_to_idx[uuid]].append((pid, mem))

with log_path.open("w", encoding="utf-8") as f:
    for idx in sorted(gpus):
        f.write(f"GPU {idx}: free={gpus[idx]['free']}MiB/{gpus[idx]['total']}MiB apps={busy[idx]}\n")

for idx in student_gpus:
    if idx not in gpus:
        raise SystemExit(f"ERROR: GPU {idx} does not exist")
    if gpus[idx]["free"] < 70000:
        raise SystemExit(f"ERROR: GPU {idx} free={gpus[idx]['free']}MiB < 70000MiB")
    if busy[idx]:
        raise SystemExit(f"ERROR: GPU {idx} has active compute processes: {busy[idx]}")
print(f"[control] preflight GPUs OK: student={student_gpus} port={port}")
PY
  PRE_RC=$?
  set -e
  if [ "${PRE_RC}" = "0" ]; then
    break
  fi
  if [ "${AUTO_WAIT}" = "1" ]; then
    echo "[control] GPU check raced right before launch; returning to wait loop..."
    sleep "${POLL_SECONDS}"
    continue
  fi
  exit "${PRE_RC}"
done

SERVING_API_KEY="$("${PYTHON_BIN}" -c 'import secrets; print(secrets.token_urlsafe(32))')"
export SERVING_API_KEY
STUDENT_SERVED_NAME="Llama-3.1-8B-Instruct-${USER}-author494"

STUDENT_PID=""
MONITOR_PID=""

stop_student() {
  set +e
  if [ -n "${STUDENT_PID}" ]; then
    "${PYTHON_BIN}" -B - "${STUDENT_PID}" <<'PY'
import os, signal, sys, time
from pathlib import Path

root_pid = int(sys.argv[1])

def descendants(parent: int) -> list[int]:
    children_map = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            st = (p / "stat").read_text()
            ppid = int(st[st.rfind(")") + 2:].split()[1])
            children_map.setdefault(ppid, []).append(int(p.name))
        except OSError:
            continue
    out, stack = [], [parent]
    while stack:
        cur = stack.pop()
        out.append(cur)
        stack.extend(children_map.get(cur, []))
    return out

pids = descendants(root_pid)
for pid in reversed(pids):
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
time.sleep(5)
for pid in reversed(pids):
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
PY
    wait "${STUDENT_PID}" 2>/dev/null || true
    STUDENT_PID=""
  fi
}

stop_monitor() {
  set +e
  if [ -n "${MONITOR_PID}" ] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=""
  fi
}

cleanup() {
  stop_monitor
  stop_student
}
trap cleanup EXIT INT TERM

while true; do
  CUDA_VISIBLE_DEVICES="${STUDENT_GPUS}" VLLM_API_KEY="${SERVING_API_KEY}" \
    "${VLLM_BIN}" serve "${RESOLVED_STUDENT}" \
    --host 127.0.0.1 \
    --port "${STUDENT_PORT}" \
    --served-model-name "${STUDENT_SERVED_NAME}" \
    --api-key "${SERVING_API_KEY}" \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --data-parallel-size 3 \
    --tensor-parallel-size 1 \
    --max-num-seqs 256 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    > "${OUTPUT_DIR}/student_server.log" 2>&1 &
  STUDENT_PID=$!
  echo "${STUDENT_PID}" > "${OUTPUT_DIR}/student.pid"

  if ! "${PYTHON_BIN}" -B - "${STUDENT_PID}" "${STUDENT_PORT}" <<'PY'; then
import os, sys, time, urllib.request

pid = int(sys.argv[1])
port = int(sys.argv[2])
deadline = time.monotonic() + 600
url = f"http://127.0.0.1:{port}/health"
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        raise SystemExit("ERROR: student server exited before becoming healthy")
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            if resp.status == 200:
                print(f"[control] student server healthy on 127.0.0.1:{port}")
                sys.exit(0)
    except OSError:
        pass
    time.sleep(5)
raise SystemExit("ERROR: student server health check timed out after 600s")
PY
    stop_student
    if [ "${AUTO_WAIT}" = "1" ]; then
      echo "[control] Student server failed startup (possible GPU race); returning to wait loop..."
      sleep "${POLL_SECONDS}"
      STUDENT_GPUS="$("${PYTHON_BIN}" -B "${HERE_DIR}/wait_gpus_author494_pro5000.py" \
        --num-gpus 3 \
        --status-file "${OUTPUT_DIR}/wait_status.json" \
        --log-file "${OUTPUT_DIR}/wait.log" \
        --port "${STUDENT_PORT}" \
        --min-free-mib 70000 \
        --poll-seconds "${POLL_SECONDS}")"
      continue
    fi
    exit 1
  fi
  break
done

"${PYTHON_BIN}" -B - "${STUDENT_GPUS}" "${OUTPUT_DIR}/gpu_watch.log" <<'PY' &
import subprocess, sys, time
from pathlib import Path

watched_indices = {int(x) for x in sys.argv[1].split(",")}
log_path = Path(sys.argv[2])

gpu_csv = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True
)
uuid_to_idx = {}
for line in gpu_csv.strip().splitlines():
    idx, uuid = [x.strip() for x in line.split(",")]
    uuid_to_idx[uuid] = int(idx)

baseline_pids: dict[int, set[str]] = {}
while True:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    cur_pids: dict[int, set[str]] = {idx: set() for idx in watched_indices}
    entries = []
    new_alerts = []
    if apps:
        for line in apps.splitlines():
            uuid, pid_s, mem = [x.strip() for x in line.split(",")]
            idx = uuid_to_idx.get(uuid)
            if idx in watched_indices:
                cur_pids[idx].add(pid_s)
                entries.append(f"gpu{idx}:pid={pid_s}({mem}MiB)")
                if idx in baseline_pids and pid_s not in baseline_pids[idx]:
                    new_alerts.append(f"gpu{idx}:pid={pid_s}")
    for idx, pids in cur_pids.items():
        if idx not in baseline_pids and pids:
            baseline_pids[idx] = set(pids)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] apps=[{', '.join(entries)}]\n")
        if new_alerts:
            f.write(f"[{ts}] NEW_HOST_PID_ALERT={new_alerts}\n")
    time.sleep(60)
PY
MONITOR_PID=$!

echo "[control] Starting run_control_author494_pro5000.py (num_problems=${NUM_PROBLEMS}, student_gpus=${STUDENT_GPUS}, k=${K_SAMPLES})..."
START_TS=$(date +%s)
set +e
"${PYTHON_BIN}" -B "${HERE_DIR}/run_control_author494_pro5000.py" \
  --repo "${REPO}" \
  --test-parquet "${TEST_PARQUET}" \
  --output-dir "${OUTPUT_DIR}" \
  --tag "${TAG}" \
  --num-problems "${NUM_PROBLEMS}" \
  --student-model "${STUDENT_SERVED_NAME}" \
  --student-api-base "http://127.0.0.1:${STUDENT_PORT}/v1" \
  --api-key-env SERVING_API_KEY \
  --student-temperature 0.6 \
  --student-max-tokens 1024 \
  --k "${K_SAMPLES}" \
  --student-concurrency "${STUDENT_CONCURRENCY}" \
  --row-concurrency "${ROW_CONCURRENCY}" \
  > "${OUTPUT_DIR}/control_run.log" 2>&1
CTRL_RC=$?
set -e
END_TS=$(date +%s)
echo "[control] finished rc=${CTRL_RC} elapsed=$((END_TS - START_TS))s"

stop_monitor
stop_student

if [ -f "${OUTPUT_DIR}/control_run.log" ]; then
  "${PYTHON_BIN}" -B - "${HERE_DIR}" "${OUTPUT_DIR}/control_run.log" "${CTRL_RC}" <<'PY'
import re, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from summarize_author494_pro5000 import extract_safe_failure_summary

text = Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace")
rc = int(sys.argv[3])
safe_re = re.compile(r"^(?:\[control\] |Control results written to: )")
for line in text.splitlines():
    if safe_re.match(line):
        print(line)
if rc != 0:
    fail_sum = extract_safe_failure_summary(text)
    if fail_sum:
        print("=== Sanitized Failure Summary ===")
        print(fail_sum)
PY
fi

exit "${CTRL_RC}"
