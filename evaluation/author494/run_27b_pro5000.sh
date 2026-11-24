#!/usr/bin/env bash
# Pro 5000 launcher for student vLLM server (bfloat16) + Eduardo-27B Tutor eval (TP=2, bfloat16)
# using kit-v3 idempotent record/replay hardening and AgnesWindowClient (max 16 concurrent).
# Prefers 5 GPUs (Tutor TP=2 + Student DP=3); falls back to 4 GPUs (Tutor TP=2 + Student DP=2)
# when fewer than 5 GPUs are idle at launch time.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_27b_pro5000.sh (--tutor-gpus <id1,id2> --student-gpus <id1,id2[,id3]> | --auto-wait) \
    --output-dir <new_dir> [--tutor-tp 2] [--student-dp auto|2|3] \
    [--num-problems 20|-1] [--student-port 18971] \
    [--poll-seconds 60] [--tag eduardo_27b_author494_pro5000] \
    [--agnes-api-base https://api.agnes-ai.cn/v1] \
    [--agnes-model agnes-3.0-flash] [--agnes-key-env AGNES_API_KEY] \
    [--judge-concurrency 12] [--eval-judge-concurrency 4] \
    [--test-parquet <path>] [--resume]

  run_27b_pro5000.sh --status <output_dir>
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
print(f"wait_state={d.get('state')} polls={d.get('polls')} eligible={d.get('eligible_gpus', d.get('selected_gpus'))} updated_at={d.get('updated_at')}")
PY
  fi
  if [ -f "${OUT_DIR}/agnes_window_stats.json" ]; then
    python3 -B - "${OUT_DIR}/agnes_window_stats.json" <<'PY'
import json, sys
from pathlib import Path
w = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"agnes_window logical_calls={w.get('logical_calls')} http_attempts={w.get('http_attempts')} "
      f"max_window={w.get('max_limit_reached')} drops={w.get('window_drops')} "
      f"congestion={w.get('congestion_events')} http_429={w.get('http_429_events', 0)} "
      f"ambiguous={w.get('ambiguous_failures')}")
PY
  fi
  if [ -f "${OUT_DIR}/replay_stats.json" ]; then
    python3 -B - "${OUT_DIR}/replay_stats.json" <<'PY'
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"replay loaded={r.get('loaded_records')} dropped_tail={r.get('dropped_corrupt_tail')} "
      f"hits={r.get('total_hits')} live={r.get('total_live')} recorded={r.get('total_recorded')}")
PY
  fi
  if [ -f "${OUT_DIR}/eval_run.log" ]; then
    HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    python3 -B - "${HERE_DIR}" "${OUT_DIR}/eval_run.log" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from summarize_author494 import format_eval_log_excerpt

text = Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace")
matched_excerpt = format_eval_log_excerpt(text)
if matched_excerpt:
    print(matched_excerpt)
PY
  else
    echo "eval_run.log: not found"
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
TUTOR_MODEL="${TUTOR_MODEL:-/dev/shm/zhujie_a_group/Eduardo-27b-3bbf17e}"
STUDENT_MODEL="${STUDENT_MODEL:-/dev/shm/zhujie_a_group/Meta-Llama-3.1-8B-Instruct}"
STUDENT_SYMLINK="${STUDENT_SYMLINK:-${BASE}/models/Meta-Llama-3.1-8B-Instruct}"
TEST_PARQUET="${TEST_PARQUET:-${BASE}/data/author494_eval/test.parquet}"

TUTOR_GPUS=""
STUDENT_GPUS=""
TUTOR_TP="2"
STUDENT_DP="auto"
AUTO_WAIT="0"
POLL_SECONDS="60"
OUTPUT_DIR=""
NUM_PROBLEMS="-1"
STUDENT_PORT="18971"
TAG="eduardo_27b_author494_pro5000"
AGNES_API_BASE="https://api.agnes-ai.cn/v1"
AGNES_MODEL="agnes-3.0-flash"
AGNES_KEY_ENV="AGNES_API_KEY"
JUDGE_CONCURRENCY="12"
EVAL_JUDGE_CONCURRENCY="4"
RESUME="0"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tutor-gpus|--tutor-gpu) TUTOR_GPUS="$2"; shift 2 ;;
    --student-gpus) STUDENT_GPUS="$2"; shift 2 ;;
    --tutor-tp) TUTOR_TP="$2"; shift 2 ;;
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
    --test-parquet) TEST_PARQUET="$2"; shift 2 ;;
    --resume) RESUME="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

# Pro 5000 runtime environment: conda libstdc++ (CXXABI_1.3.15), caches under $BASE, no proxies.
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

if [ "${TUTOR_TP}" != "2" ]; then
  echo "ERROR: Pro 5000 requires --tutor-tp 2 (72 GB GPU cannot fit Eduardo-27B at TP=1); got ${TUTOR_TP}" >&2
  exit 2
fi
if [ "${STUDENT_DP}" != "auto" ] && [ "${STUDENT_DP}" != "2" ] && [ "${STUDENT_DP}" != "3" ]; then
  echo "ERROR: --student-dp must be auto, 2, or 3 (got ${STUDENT_DP})" >&2
  exit 2
fi
if [ $((JUDGE_CONCURRENCY + EVAL_JUDGE_CONCURRENCY)) -gt 16 ]; then
  echo "ERROR: judge_concurrency (${JUDGE_CONCURRENCY}) + eval_judge_concurrency (${EVAL_JUDGE_CONCURRENCY}) exceeds 16" >&2
  exit 2
fi
if [ -z "${OUTPUT_DIR}" ]; then
  echo "ERROR: --output-dir is required" >&2
  usage
  exit 2
fi
if [ "${AUTO_WAIT}" = "0" ] && { [ -z "${TUTOR_GPUS}" ] || [ -z "${STUDENT_GPUS}" ]; }; then
  echo "ERROR: provide --auto-wait OR both --tutor-gpus and --student-gpus" >&2
  usage
  exit 2
fi
if [ "${AUTO_WAIT}" = "1" ] && { [ -n "${TUTOR_GPUS}" ] || [ -n "${STUDENT_GPUS}" ]; }; then
  echo "ERROR: --auto-wait cannot be combined with --tutor-gpus/--student-gpus" >&2
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
if [ ! -f "${TEST_PARQUET}" ]; then
  echo "ERROR: test.parquet not found at ${TEST_PARQUET}" >&2
  exit 2
fi

"${PYTHON_BIN}" -B - "${TEST_PARQUET}" <<'PY'
import hashlib, sys, pandas as pd
from pathlib import Path
EXP_SHA = "338dc73b005b2c8036cc23959061007d7339c24d1988800b62543e1c8224385c"
p = Path(sys.argv[1])
df = pd.read_parquet(p)
if len(df) != 494:
    raise SystemExit(f"ERROR: expected 494 rows in test.parquet, got {len(df)}")
n_tp = sum(1 for r in df["extra_info"] if isinstance(r, dict) and r.get("transfer_problem") and r.get("transfer_answer"))
if n_tp != 494:
    raise SystemExit(f"ERROR: expected 494/494 transfer variants, got {n_tp}")
h = hashlib.sha256()
with p.open("rb") as f:
    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
        h.update(chunk)
got_sha = h.hexdigest()
if got_sha != EXP_SHA:
    raise SystemExit(f"ERROR: test.parquet sha256 mismatch: got={got_sha} exp={EXP_SHA}")
print(f"[author494_27b_pro5000] verified test.parquet rows=494 transfer_valid=494 sha256={got_sha}")
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

if [ "${RESUME}" = "1" ]; then
  if [ ! -f "${OUTPUT_DIR}/replay_cache/replay_manifest.json" ]; then
    echo "ERROR: --resume requires existing ${OUTPUT_DIR}/replay_cache/replay_manifest.json" >&2
    exit 2
  fi
  if ls "${OUTPUT_DIR}"/transfer_eval_*.json >/dev/null 2>&1; then
    echo "ERROR: ${OUTPUT_DIR} already has completed transfer_eval_*.json; refusing rerun" >&2
    exit 2
  fi
else
  if [ -e "${OUTPUT_DIR}" ] && [ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null || true)" ]; then
    echo "ERROR: output-dir ${OUTPUT_DIR} already exists and is non-empty; pass --resume or choose a new dir" >&2
    exit 2
  fi
fi

mkdir -p "${OUTPUT_DIR}"
echo "$$" > "${OUTPUT_DIR}/runner.pid"
HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WAIT_TARGET="5"
WAIT_FALLBACK="4"
if [ "${STUDENT_DP}" = "3" ]; then
  WAIT_TARGET="5"
  WAIT_FALLBACK="0"
elif [ "${STUDENT_DP}" = "2" ]; then
  WAIT_TARGET="4"
  WAIT_FALLBACK="0"
fi

ACTUAL_STUDENT_DP=""

assign_split_gpus() {
  local raw="$1"
  TUTOR_GPUS="$("${PYTHON_BIN}" -B - "${raw}" <<'PY'
import sys
gpus = [x.strip() for x in sys.argv[1].split(",") if x.strip()]
print(",".join(gpus[:2]))
PY
)"
  STUDENT_GPUS="$("${PYTHON_BIN}" -B - "${raw}" <<'PY'
import sys
gpus = [x.strip() for x in sys.argv[1].split(",") if x.strip()]
print(",".join(gpus[2:]))
PY
)"
  ACTUAL_STUDENT_DP="$("${PYTHON_BIN}" -B - "${raw}" <<'PY'
import sys
gpus = [x.strip() for x in sys.argv[1].split(",") if x.strip()]
print(len(gpus) - 2)
PY
)"
}

wait_for_cluster_gpus() {
  echo "[author494_27b_pro5000] Waiting for Pro 5000 GPUs (target=${WAIT_TARGET}, fallback=${WAIT_FALLBACK}, >=70000 MiB free, GPU6 quirk)..."
  READ_GPUS="$("${PYTHON_BIN}" -B "${HERE_DIR}/wait_gpus_27b_pro5000.py" \
    --status-file "${OUTPUT_DIR}/wait_status.json" \
    --log-file "${OUTPUT_DIR}/wait.log" \
    --port "${STUDENT_PORT}" \
    --num-gpus "${WAIT_TARGET}" \
    --fallback-min-gpus "${WAIT_FALLBACK}" \
    --min-free-mib 70000 \
    --poll-seconds "${POLL_SECONDS}")"
  assign_split_gpus "${READ_GPUS}"
}

while true; do
  if [ "${AUTO_WAIT}" = "1" ]; then
    wait_for_cluster_gpus
  else
    ACTUAL_STUDENT_DP="$("${PYTHON_BIN}" -B - "${STUDENT_GPUS}" <<'PY'
import sys
print(len([x.strip() for x in sys.argv[1].split(",") if x.strip()]))
PY
)"
    if [ "${STUDENT_DP}" != "auto" ] && [ "${ACTUAL_STUDENT_DP}" != "${STUDENT_DP}" ]; then
      echo "ERROR: --student-dp ${STUDENT_DP} disagrees with --student-gpus ${STUDENT_GPUS}" >&2
      exit 2
    fi
  fi

  set +e
  "${PYTHON_BIN}" -B - "${TUTOR_GPUS}" "${STUDENT_GPUS}" "${STUDENT_PORT}" "${OUTPUT_DIR}/gpu_preflight.log" <<'PY'
import socket, subprocess, sys
from pathlib import Path

tutor_gpus = [int(x) for x in sys.argv[1].split(",") if x.strip()]
student_gpus = [int(x) for x in sys.argv[2].split(",") if x.strip()]
port = int(sys.argv[3])
log_path = Path(sys.argv[4])

if len(tutor_gpus) != 2 or len(student_gpus) not in (2, 3) or len(set(tutor_gpus + student_gpus)) != 2 + len(student_gpus):
    raise SystemExit(f"ERROR: expected 2 tutor GPUs and 2 or 3 distinct student GPUs; got tutor={tutor_gpus} student={student_gpus}")

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

for idx in tutor_gpus + student_gpus:
    if idx not in gpus:
        raise SystemExit(f"ERROR: GPU {idx} does not exist")
    if gpus[idx]["free"] < 70000:
        raise SystemExit(f"ERROR: GPU {idx} free={gpus[idx]['free']}MiB < 70000MiB")
    if busy[idx]:
        raise SystemExit(f"ERROR: GPU {idx} has active compute processes: {busy[idx]}")
print(f"[author494_27b_pro5000] preflight GPUs OK: tutor={tutor_gpus} (TP=2) student={student_gpus} (DP={len(student_gpus)}) port={port}")
PY
  PRE_RC=$?
  set -e
  if [ "${PRE_RC}" = "0" ]; then
    break
  fi
  if [ "${AUTO_WAIT}" = "1" ]; then
    echo "[author494_27b_pro5000] GPU check raced right before launch; returning to wait loop..."
    sleep "${POLL_SECONDS}"
    continue
  fi
  exit "${PRE_RC}"
done

SERVING_API_KEY="$("${PYTHON_BIN}" -c 'import secrets; print(secrets.token_urlsafe(32))')"
export SERVING_API_KEY
STUDENT_SERVED_NAME="Llama-3.1-8B-Instruct-${USER}-author494-27b"

STUDENT_PID=""
HOLDER_PIDS=""
MONITOR_PID=""

release_holders() {
  set +e
  if [ -n "${HOLDER_PIDS}" ]; then
    for hpid in ${HOLDER_PIDS}; do
      kill -TERM "${hpid}" 2>/dev/null || true
    done
    for hpid in ${HOLDER_PIDS}; do
      wait "${hpid}" 2>/dev/null || true
    done
    HOLDER_PIDS=""
  fi
}

stop_student_and_holder() {
  set +e
  release_holders
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

cleanup() {
  set +e
  if [ -n "${MONITOR_PID}" ] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  stop_student_and_holder
}
trap cleanup EXIT INT TERM

while true; do
  HOLDER_PIDS=""
  HOLD_OK="1"
  IFS=',' read -ra TUTOR_ARR <<< "${TUTOR_GPUS}"
  for tgpu in "${TUTOR_ARR[@]}"; do
    hlog="${OUTPUT_DIR}/tutor_hold_${tgpu}.log"
    rm -f "${hlog}"
    CUDA_VISIBLE_DEVICES="${tgpu}" "${PYTHON_BIN}" -B "${HERE_DIR}/wait_gpus_27b_pro5000.py" \
      --hold --min-free-mib 70000 > "${hlog}" 2>&1 &
    hpid=$!
    HOLDER_PIDS="${HOLDER_PIDS} ${hpid}"
    if ! "${PYTHON_BIN}" -B - "${hpid}" "${hlog}" <<'PY'; then
import os, sys, time
from pathlib import Path
pid, log_p = int(sys.argv[1]), Path(sys.argv[2])
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    if log_p.is_file() and "RESERVED" in log_p.read_text(errors="replace"):
        sys.exit(0)
    try:
        os.kill(pid, 0)
    except OSError:
        sys.exit(1)
    time.sleep(0.5)
sys.exit(1)
PY
      HOLD_OK="0"
      break
    fi
  done

  if [ "${HOLD_OK}" != "1" ]; then
    stop_student_and_holder
    if [ "${AUTO_WAIT}" = "1" ]; then
      echo "[author494_27b_pro5000] Tutor GPU(s) ${TUTOR_GPUS} could not be held; returning to wait loop..."
      sleep "${POLL_SECONDS}"
      wait_for_cluster_gpus
      continue
    fi
    echo "ERROR: failed to reserve Tutor GPU(s) ${TUTOR_GPUS}" >&2
    exit 1
  fi

  CUDA_VISIBLE_DEVICES="${STUDENT_GPUS}" VLLM_API_KEY="${SERVING_API_KEY}" \
    "${VLLM_BIN}" serve "${RESOLVED_STUDENT}" \
    --host 127.0.0.1 \
    --port "${STUDENT_PORT}" \
    --served-model-name "${STUDENT_SERVED_NAME}" \
    --api-key "${SERVING_API_KEY}" \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --data-parallel-size "${ACTUAL_STUDENT_DP}" \
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
                print(f"[author494_27b_pro5000] student server healthy on 127.0.0.1:{port}")
                sys.exit(0)
    except OSError:
        pass
    time.sleep(5)
raise SystemExit("ERROR: student server health check timed out after 600s")
PY
    stop_student_and_holder
    if [ "${AUTO_WAIT}" = "1" ]; then
      echo "[author494_27b_pro5000] Student server failed startup (possible GPU race); returning to wait loop..."
      sleep "${POLL_SECONDS}"
      wait_for_cluster_gpus
      continue
    fi
    exit 1
  fi

  release_holders

  # Verify both Tutor GPUs are clean after handoff; swap raced Tutor GPU(s) if replacements exist.
  STILL_GOOD_TUTOR="$("${PYTHON_BIN}" -B - "${TUTOR_GPUS}" <<'PY'
import subprocess, sys, time
targets = [int(x) for x in sys.argv[1].split(",") if x.strip()]
good = []
for _ in range(15):
    gpu_csv = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    uuid_map, free_map = {}, {}
    for line in gpu_csv.strip().splitlines():
        idx_s, uuid, free_s = [x.strip() for x in line.split(",")]
        uuid_map[uuid] = int(idx_s)
        free_map[int(idx_s)] = int(free_s)
    apps_csv = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    busy_set = {uuid_map.get(line.split(",")[0].strip()) for line in apps_csv.splitlines() if line.strip()}
    good = [t for t in targets if free_map.get(t, 0) >= 70000 and t not in busy_set]
    if len(good) == len(targets):
        break
    time.sleep(1)
print(",".join(map(str, good)))
PY
)"
  GOOD_COUNT="$("${PYTHON_BIN}" -B - "${STILL_GOOD_TUTOR}" <<'PY'
import sys
print(len([x for x in sys.argv[1].split(",") if x.strip()]))
PY
)"
  if [ "${GOOD_COUNT}" -lt 2 ]; then
    NEED_COUNT=$((2 - GOOD_COUNT))
    EXCL_LIST="${STUDENT_GPUS}"
    if [ -n "${STILL_GOOD_TUTOR}" ]; then
      EXCL_LIST="${EXCL_LIST},${STILL_GOOD_TUTOR}"
    fi
    REPS="$("${PYTHON_BIN}" -B "${HERE_DIR}/wait_gpus_27b_pro5000.py" \
      --find-replacement --count "${NEED_COUNT}" --exclude "${EXCL_LIST}" --min-free-mib 70000 || true)"
    if [ -n "${REPS}" ]; then
      if [ -n "${STILL_GOOD_TUTOR}" ]; then
        TUTOR_GPUS="${STILL_GOOD_TUTOR},${REPS}"
      else
        TUTOR_GPUS="${REPS}"
      fi
      echo "[author494_27b_pro5000] Tutor GPU(s) raced during handoff; switched to ${TUTOR_GPUS}"
      break
    fi
    if [ "${AUTO_WAIT}" = "1" ]; then
      echo "[author494_27b_pro5000] Tutor GPU(s) ${TUTOR_GPUS} became busy during handoff; returning to wait loop..."
      stop_student_and_holder
      sleep "${POLL_SECONDS}"
      wait_for_cluster_gpus
      continue
    fi
    echo "ERROR: Tutor GPU(s) ${TUTOR_GPUS} no longer free (>=70000 MiB)" >&2
    exit 1
  fi
  break
done

"${PYTHON_BIN}" -B - "${TUTOR_GPUS},${STUDENT_GPUS}" "${OUTPUT_DIR}/gpu_watch.log" <<'PY' &
import subprocess, sys, time
from pathlib import Path

watched_indices = {int(x) for x in sys.argv[1].split(",") if x.strip()}
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

RESUME_FLAG=()
if [ "${RESUME}" = "1" ]; then
  RESUME_FLAG=(--resume)
  echo "=== RESUME $(date '+%Y-%m-%d %H:%M:%S') ===" >> "${OUTPUT_DIR}/eval_run.log"
else
  : > "${OUTPUT_DIR}/eval_run.log"
fi

echo "[author494_27b_pro5000] Starting run_eval_27b_author494.py (num_problems=${NUM_PROBLEMS}, resume=${RESUME}, tutor_gpus=${TUTOR_GPUS} TP=2, student_gpus=${STUDENT_GPUS} DP=${ACTUAL_STUDENT_DP}, judge_conc=${JUDGE_CONCURRENCY}+${EVAL_JUDGE_CONCURRENCY})..."
START_TS=$(date +%s)
set +e
CUDA_VISIBLE_DEVICES="${TUTOR_GPUS}" PYTHONPATH="${REPO}" \
  "${PYTHON_BIN}" -B "${HERE_DIR}/run_eval_27b_author494.py" \
  --agnes-stats-out "${OUTPUT_DIR}/agnes_window_stats.json" \
  --replay-stats-out "${OUTPUT_DIR}/replay_stats.json" \
  --student-model-dir "${RESOLVED_STUDENT}" \
  "${RESUME_FLAG[@]}" \
  --model "${TUTOR_MODEL}" \
  --tensor-parallel-size 2 \
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
  >> "${OUTPUT_DIR}/eval_run.log" 2>&1
EVAL_RC=$?
set -e
END_TS=$(date +%s)
echo "[author494_27b_pro5000] eval finished rc=${EVAL_RC} elapsed=$((END_TS - START_TS))s"

if [ -f "${OUTPUT_DIR}/eval_run.log" ]; then
  "${PYTHON_BIN}" -B - "${HERE_DIR}" "${OUTPUT_DIR}/eval_run.log" "${EVAL_RC}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from summarize_author494 import format_eval_log_excerpt, extract_safe_failure_summary

text = Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace")
rc = int(sys.argv[3])
excerpt = format_eval_log_excerpt(text)
if excerpt:
    print(excerpt)
if rc != 0:
    fail_sum = extract_safe_failure_summary(text)
    if fail_sum:
        print("=== Sanitized Failure Summary ===")
        print(fail_sum)
PY
fi

RESULT_JSON=$(ls -t "${OUTPUT_DIR}"/transfer_eval_*.json 2>/dev/null | head -n 1 || true)
if [ -n "${RESULT_JSON}" ]; then
  STATS_FLAG=()
  if [ -f "${OUTPUT_DIR}/agnes_window_stats.json" ]; then
    STATS_FLAG=(--agnes-stats "${OUTPUT_DIR}/agnes_window_stats.json")
  fi
  "${PYTHON_BIN}" -B "${HERE_DIR}/summarize_author494.py" \
    --paper-baseline eduardo-27b \
    --ours-label "Ours (Pro 5000 + Agnes)" \
    "${STATS_FLAG[@]}" \
    "${RESULT_JSON}"
fi

exit "${EVAL_RC}"
