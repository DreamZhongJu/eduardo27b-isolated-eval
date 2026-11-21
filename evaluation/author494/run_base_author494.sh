#!/usr/bin/env bash
# Launch 3-GPU student vLLM server + 1-GPU Tutor eval (unmodified author code)
# using Agnes via --gemini interface. Aggregate status/logs only.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_base_author494.sh --tutor-gpu <id> --student-gpus <id1,id2,id3> \
    --output-dir <new_dir> [--num-problems 20|-1] [--student-port 18961] \
    [--tag base_qwen35_4b_author494] [--agnes-api-base https://api.agnes-ai.cn/v1] \
    [--agnes-model agnes-3.0-flash] [--agnes-key-env AGNES_API_KEY] \
    [--test-parquet <path>] [--prepare-only]

  run_base_author494.sh --status <output_dir>
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
  if [ -f "${OUT_DIR}/eval_run.log" ]; then
    HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    python3 -B - "${HERE_DIR}" "${OUT_DIR}/eval_run.log" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from summarize_author494 import extract_safe_eval_log_lines

text = Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace")
matched = extract_safe_eval_log_lines(text, include_rounds=True)
print(f"status_lines_matched={len(matched)}")
for line in matched[-10:]:
    print(" ", line)
PY
  else
    echo "eval_run.log: not found"
  fi
  if [ -f "${OUT_DIR}/gpu_watch.log" ]; then
    ALERTS=$(grep -c "FOREIGN_PID_ALERT" "${OUT_DIR}/gpu_watch.log" || true)
    echo "gpu_watch_foreign_alerts=${ALERTS}"
  fi
  exit 0
fi

ROOT="${A_GROUP_ROOT:-/workspace/A_group_repro_20261117}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/envs/env-candidate-vllm0191/bin/python}"
VLLM_BIN="${VLLM_BIN:-$(dirname "${PYTHON_BIN}")/vllm}"
REPO="${EDUARDO_REPO:-${ROOT}/source/EduardoRL}"
TUTOR_MODEL="${TUTOR_MODEL:-${ROOT}/models/Qwen3.5-4B}"
STUDENT_MODEL="${STUDENT_MODEL:-/workspace/models/Meta-Llama-3.1-8B-Instruct}"
OFFICIAL_DATA="${OFFICIAL_DATA:-${ROOT}/data/official}"
TEST_PARQUET="${ROOT}/data/author494_eval/test.parquet"

TUTOR_GPU=""
STUDENT_GPUS=""
OUTPUT_DIR=""
NUM_PROBLEMS="-1"
STUDENT_PORT="18961"
TAG="base_qwen35_4b_author494"
AGNES_API_BASE="https://api.agnes-ai.cn/v1"
AGNES_MODEL="agnes-3.0-flash"
AGNES_KEY_ENV="AGNES_API_KEY"
PREPARE_ONLY="0"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tutor-gpu) TUTOR_GPU="$2"; shift 2 ;;
    --student-gpus) STUDENT_GPUS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --num-problems) NUM_PROBLEMS="$2"; shift 2 ;;
    --student-port) STUDENT_PORT="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --agnes-api-base) AGNES_API_BASE="$2"; shift 2 ;;
    --agnes-model) AGNES_MODEL="$2"; shift 2 ;;
    --agnes-key-env) AGNES_KEY_ENV="$2"; shift 2 ;;
    --test-parquet) TEST_PARQUET="$2"; shift 2 ;;
    --prepare-only) PREPARE_ONLY="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

# Never use proxies; offline for HF/vLLM.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export WANDB_MODE=disabled
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export USER="${USER:-root}"
export PYTHONUNBUFFERED=1

if [ ! -f "${TEST_PARQUET}" ]; then
  echo "[author494] Materializing test.parquet via unmodified eduardo.process_dataset..."
  DATA_OUT_DIR="$(dirname "${TEST_PARQUET}")"
  TMP_JSONL_DIR="$(mktemp -d "${ROOT}/tmp_author494_jsonl_XXXXXX")"
  trap 'rm -rf "${TMP_JSONL_DIR}"' EXIT
  "${PYTHON_BIN}" -B - "${OFFICIAL_DATA}/data/train-00000-of-00001.parquet" \
    "${OFFICIAL_DATA}/data/test-00000-of-00001.parquet" "${TMP_JSONL_DIR}" <<'PY'
import json, sys
from pathlib import Path
import pyarrow.parquet as pq

train_pq, test_pq, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
for name, src in (("train.jsonl", train_pq), ("test.jsonl", test_pq)):
    rows = pq.read_table(src).to_pylist()
    with (out_dir / name).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[author494] wrote {len(rows)} rows to {name}")
PY
  PYTHONPATH="${REPO}" "${PYTHON_BIN}" -B -m eduardo.process_dataset \
    --data-dir "${TMP_JSONL_DIR}" \
    --num-holdout 500 \
    --seed 42 \
    --out-dir "${DATA_OUT_DIR}"
  rm -rf "${TMP_JSONL_DIR}"
  trap - EXIT
fi

"${PYTHON_BIN}" -B - "${TEST_PARQUET}" <<'PY'
import sys, pandas as pd
df = pd.read_parquet(sys.argv[1])
if len(df) != 494:
    raise SystemExit(f"ERROR: expected 494 rows in test.parquet, got {len(df)}")
n_tp = sum(1 for r in df["extra_info"] if isinstance(r, dict) and r.get("transfer_problem") and r.get("transfer_answer"))
if n_tp != 494:
    raise SystemExit(f"ERROR: expected 494/494 transfer variants, got {n_tp}")
print(f"[author494] verified test.parquet rows=494 transfer_valid=494")
PY

if [ "${PREPARE_ONLY}" = "1" ]; then
  exit 0
fi

if [ -z "${TUTOR_GPU}" ] || [ -z "${STUDENT_GPUS}" ] || [ -z "${OUTPUT_DIR}" ]; then
  echo "ERROR: --tutor-gpu, --student-gpus, and --output-dir are required" >&2
  usage
  exit 2
fi
if [ -n "${GEMINI_API_KEY:-}" ]; then
  echo "ERROR: GEMINI_API_KEY must NOT be set" >&2
  exit 2
fi
if [ -z "${!AGNES_KEY_ENV:-}" ]; then
  echo "ERROR: Agnes key env var ${AGNES_KEY_ENV} is not set" >&2
  exit 2
fi
if [ -e "${OUTPUT_DIR}" ] && [ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null || true)" ]; then
  echo "ERROR: output-dir ${OUTPUT_DIR} already exists and is non-empty; choose a new dir" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"
echo "$$" > "${OUTPUT_DIR}/runner.pid"

# Validate 4 distinct free GPUs (>= 78000 MiB free, 0 compute apps) and free port.
"${PYTHON_BIN}" -B - "${TUTOR_GPU}" "${STUDENT_GPUS}" "${STUDENT_PORT}" "${OUTPUT_DIR}/gpu_preflight.log" <<'PY'
import socket, subprocess, sys
from pathlib import Path

tutor_gpu = int(sys.argv[1])
student_gpus = [int(x) for x in sys.argv[2].split(",") if x.strip()]
port = int(sys.argv[3])
log_path = Path(sys.argv[4])

if len(student_gpus) != 3 or len(set([tutor_gpu] + student_gpus)) != 4:
    raise SystemExit("ERROR: must provide 1 tutor GPU and 3 distinct student GPUs (4 distinct total)")

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

selected = [tutor_gpu] + student_gpus
for idx in selected:
    if idx not in gpus:
        raise SystemExit(f"ERROR: GPU {idx} does not exist")
    if gpus[idx]["free"] < 78000:
        raise SystemExit(f"ERROR: GPU {idx} free={gpus[idx]['free']}MiB < 78000MiB")
    if busy[idx]:
        raise SystemExit(f"ERROR: GPU {idx} has active compute processes: {busy[idx]}")
print(f"[author494] preflight GPUs OK: tutor={tutor_gpu} student={student_gpus} port={port}")
PY

# Ephemeral loopback serving token (never logged or printed).
SERVING_API_KEY="$("${PYTHON_BIN}" -c 'import secrets; print(secrets.token_urlsafe(32))')"
export SERVING_API_KEY
STUDENT_SERVED_NAME="Llama-3.1-8B-Instruct-${USER}-author494"

STUDENT_PID=""
MONITOR_PID=""

cleanup() {
  set +e
  if [ -n "${MONITOR_PID}" ] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  if [ -n "${STUDENT_PID}" ]; then
    # Terminate only descendants of our launched student server, never others.
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
  fi
}
trap cleanup EXIT INT TERM

# Launch 3-GPU data-parallel student server matching launch_student_server.sh.
CUDA_VISIBLE_DEVICES="${STUDENT_GPUS}" VLLM_API_KEY="${SERVING_API_KEY}" \
  "${VLLM_BIN}" serve "${STUDENT_MODEL}" \
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

# Wait for student server health check (up to 600s).
"${PYTHON_BIN}" -B - "${STUDENT_PID}" "${STUDENT_PORT}" <<'PY'
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
                print(f"[author494] student server healthy on 127.0.0.1:{port}")
                sys.exit(0)
    except OSError:
        pass
    time.sleep(5)
raise SystemExit("ERROR: student server health check timed out after 600s")
PY

# Start 60-second GPU process monitor for the 4 selected GPUs.
"${PYTHON_BIN}" -B - "$$" "${TUTOR_GPU},${STUDENT_GPUS}" "${OUTPUT_DIR}/gpu_watch.log" <<'PY' &
import subprocess, sys, time
from pathlib import Path

root_pid = int(sys.argv[1])
watched_indices = {int(x) for x in sys.argv[2].split(",")}
log_path = Path(sys.argv[3])

def descendants(parent: int) -> set[int]:
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
    out, stack = set(), [parent]
    while stack:
        cur = stack.pop()
        out.add(cur)
        stack.extend(children_map.get(cur, []))
    return out

gpu_csv = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True
)
uuid_to_idx = {}
for line in gpu_csv.strip().splitlines():
    idx, uuid = [x.strip() for x in line.split(",")]
    uuid_to_idx[uuid] = int(idx)

while True:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    own = descendants(root_pid)
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    entries = []
    foreign = []
    if apps:
        for line in apps.splitlines():
            uuid, pid_s, mem = [x.strip() for x in line.split(",")]
            idx = uuid_to_idx.get(uuid)
            if idx in watched_indices:
                pid = int(pid_s)
                entries.append(f"gpu{idx}:pid={pid}({mem}MiB)")
                if pid not in own:
                    foreign.append(f"gpu{idx}:pid={pid}")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] apps=[{', '.join(entries)}]\n")
        if foreign:
            f.write(f"[{ts}] FOREIGN_PID_ALERT={foreign}\n")
    time.sleep(60)
PY
MONITOR_PID=$!

# Run unmodified author eval script on 1 Tutor GPU.
echo "[author494] Starting eval.run_env_test_eval (num_problems=${NUM_PROBLEMS}, tutor_gpu=${TUTOR_GPU}, student_gpus=${STUDENT_GPUS})..."
START_TS=$(date +%s)
set +e
CUDA_VISIBLE_DEVICES="${TUTOR_GPU}" PYTHONPATH="${REPO}" \
  "${PYTHON_BIN}" -B -m eval.run_env_test_eval \
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
  --judge-concurrency 3 \
  --eval-judge-concurrency 2 \
  --output-dir "${OUTPUT_DIR}" \
  --tag "${TAG}" \
  > "${OUTPUT_DIR}/eval_run.log" 2>&1
EVAL_RC=$?
set -e
END_TS=$(date +%s)
echo "[author494] eval finished rc=${EVAL_RC} elapsed=$((END_TS - START_TS))s"

HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Print ONLY aggregate summary lines from eval_run.log (never raw conversations/problems/WARNING/ERROR).
if [ -f "${OUTPUT_DIR}/eval_run.log" ]; then
  "${PYTHON_BIN}" -B - "${HERE_DIR}" "${OUTPUT_DIR}/eval_run.log" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from summarize_author494 import format_eval_log_excerpt

text = Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace")
excerpt = format_eval_log_excerpt(text)
if excerpt:
    print(excerpt)
PY
fi

RESULT_JSON=$(ls -t "${OUTPUT_DIR}"/transfer_eval_*.json 2>/dev/null | head -n 1 || true)
if [ -n "${RESULT_JSON}" ]; then
  "${PYTHON_BIN}" -B "${HERE_DIR}/summarize_author494.py" "${RESULT_JSON}"
fi

exit "${EVAL_RC}"
