#!/usr/bin/env bash
# Launch 3-GPU student vLLM server + 1-GPU Tutor eval (unmodified author code)
# on RTX PRO 5000 using Agnes (<=3 total concurrency). Aggregate status/logs only.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_author494_pro5000.sh --tutor-model <path> --output-dir <new_dir> \
    (--tutor-gpu <id> --student-gpus <id1,id2,id3> | --auto-wait) \
    [--num-problems 20|-1] [--student-port 18961] [--poll-seconds 60] \
    [--tag <label>] [--agnes-api-base https://api.agnes-ai.cn/v1] \
    [--agnes-model agnes-3.0-flash] [--agnes-key-env AGNES_API_KEY] \
    [--judge-concurrency 2] [--eval-judge-concurrency 1] \
    [--test-parquet <path>] [--prepare-only]

  run_author494_pro5000.sh --status <output_dir>
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
  if [ -f "${OUT_DIR}/eval_run.log" ]; then
    HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    python3 -B - "${HERE_DIR}" "${OUT_DIR}/eval_run.log" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from summarize_author494_pro5000 import extract_safe_eval_log_lines

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

TUTOR_MODEL=""
TUTOR_GPU=""
STUDENT_GPUS=""
AUTO_WAIT="0"
POLL_SECONDS="60"
OUTPUT_DIR=""
NUM_PROBLEMS="-1"
STUDENT_PORT="18961"
TAG=""
AGNES_API_BASE="https://api.agnes-ai.cn/v1"
AGNES_MODEL="agnes-3.0-flash"
AGNES_KEY_ENV="AGNES_API_KEY"
JUDGE_CONCURRENCY="2"
EVAL_JUDGE_CONCURRENCY="1"
PREPARE_ONLY="0"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tutor-model) TUTOR_MODEL="$2"; shift 2 ;;
    --tutor-gpu) TUTOR_GPU="$2"; shift 2 ;;
    --student-gpus) STUDENT_GPUS="$2"; shift 2 ;;
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
    --prepare-only) PREPARE_ONLY="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

# Pro 5000 runtime environment: conda libstdc++ (CXXABI_1.3.15), caches under $BASE, no proxies.
unset LD_LIBRARY_PATH HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true
export LD_LIBRARY_PATH="${ENV_DIR}/lib"
export PATH="${ENV_DIR}/bin:${PATH}"
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

# Materialize test.parquet via unmodified eduardo.process_dataset if needed.
if [ ! -f "${TEST_PARQUET}" ]; then
  echo "[author494] Materializing test.parquet via unmodified eduardo.process_dataset..."
  DATA_OUT_DIR="$(dirname "${TEST_PARQUET}")"
  RAW_DIR="${BASE}/data/official_5d7529f0"
  TMP_JSONL_DIR="$(mktemp -d "${TMPDIR}/author494_jsonl_XXXXXX")"
  trap 'rm -rf "${TMP_JSONL_DIR}"' EXIT
  "${PYTHON_BIN}" -B - "${BASE}/data" "${RAW_DIR}" "${TMP_JSONL_DIR}" <<'PY'
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
import pyarrow.parquet as pq

data_root = Path(sys.argv[1])
raw_dir = Path(sys.argv[2])
jsonl_dir = Path(sys.argv[3])

FILES = {
    "train": (
        "data/train-00000-of-00001.parquet",
        "5171c9891b502f9792b14ac2b15684ce85b8ceb2593689ed3e88b9f349b25d61",
    ),
    "test": (
        "data/test-00000-of-00001.parquet",
        "109b45cf350cb70414a294ff999d4f545b06e9df91e02ec1a0a3b30783c46ec0",
    ),
}
REV = "5d7529f06359f31b692c92634f5cc11c8b7953af"

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

for split, (rel, exp_sha) in FILES.items():
    fname = Path(rel).name
    found = None
    if data_root.is_dir():
        for cand in sorted(data_root.rglob(fname)):
            if cand.is_file() and sha256_file(cand) == exp_sha:
                found = cand
                break
    if found is None:
        dest = raw_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://hf-mirror.com/datasets/dmacjam/eduardoRL-transfer-dataset/resolve/{REV}/{rel}"
        print(f"[author494] Downloading {fname} from hf-mirror (no token)...")
        tmp = dest.with_suffix(".tmp")
        req = urllib.request.Request(url, headers={"User-Agent": "eduardo-author494-eval/1.0"})
        with urllib.request.urlopen(req, timeout=180) as resp, tmp.open("wb") as out:
            while True:
                buf = resp.read(1024 * 1024)
                if not buf:
                    break
                out.write(buf)
        got_sha = sha256_file(tmp)
        if got_sha != exp_sha:
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"ERROR: SHA256 mismatch for {fname}: got={got_sha} exp={exp_sha}")
        os.replace(tmp, dest)
        found = dest
    rows = pq.read_table(found).to_pylist()
    with (jsonl_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[author494] verified {fname} -> wrote {len(rows)} rows to {split}.jsonl")
PY
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH="${REPO}" "${PYTHON_BIN}" -B -m eduardo.process_dataset \
    --data-dir "${TMP_JSONL_DIR}" \
    --num-holdout 500 \
    --seed 42 \
    --out-dir "${DATA_OUT_DIR}"
  rm -rf "${TMP_JSONL_DIR}"
  trap - EXIT
fi

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"${PYTHON_BIN}" -B - "${TEST_PARQUET}" <<'PY'
import sys, pandas as pd
df = pd.read_parquet(sys.argv[1])
if len(df) != 494:
    raise SystemExit(f"ERROR: expected 494 rows in test.parquet, got {len(df)}")
n_tp = sum(1 for r in df["extra_info"] if isinstance(r, dict) and r.get("transfer_problem") and r.get("transfer_answer"))
if n_tp != 494:
    raise SystemExit(f"ERROR: expected 494/494 transfer variants, got {n_tp}")
print("[author494] verified test.parquet rows=494 transfer_valid=494")
PY

if [ "${PREPARE_ONLY}" = "1" ]; then
  exit 0
fi

# Enforce Pro 5000 Agnes concurrency limit <= 3.
"${PYTHON_BIN}" -B - "${JUDGE_CONCURRENCY}" "${EVAL_JUDGE_CONCURRENCY}" <<'PY'
import sys
jc, ejc = int(sys.argv[1]), int(sys.argv[2])
if jc <= 0 or ejc <= 0 or (jc + ejc) > 3:
    raise SystemExit(f"ERROR: Pro 5000 Agnes concurrency limit is <= 3, got {jc} + {ejc} = {jc + ejc}")
PY

if [ -z "${TUTOR_MODEL}" ]; then
  echo "ERROR: --tutor-model is required" >&2
  usage
  exit 2
fi
if [ ! -d "${TUTOR_MODEL}" ]; then
  echo "ERROR: --tutor-model directory not found: ${TUTOR_MODEL}" >&2
  exit 2
fi
if [ -z "${TAG}" ]; then
  echo "ERROR: --tag is required" >&2
  usage
  exit 2
fi
if [ -z "${OUTPUT_DIR}" ]; then
  echo "ERROR: --output-dir is required" >&2
  usage
  exit 2
fi
if [ "${AUTO_WAIT}" = "0" ] && { [ -z "${TUTOR_GPU}" ] || [ -z "${STUDENT_GPUS}" ]; }; then
  echo "ERROR: provide --auto-wait OR both --tutor-gpu and --student-gpus" >&2
  usage
  exit 2
fi
if [ "${AUTO_WAIT}" = "1" ] && { [ -n "${TUTOR_GPU}" ] || [ -n "${STUDENT_GPUS}" ]; }; then
  echo "ERROR: --auto-wait cannot be combined with --tutor-gpu/--student-gpus" >&2
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

# Resolve student model under /dev/shm/zhujie_a_group (or symlink target).
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
    echo "[author494] Waiting for 4 idle Pro 5000 GPUs (>=70000 MiB free, 0 compute apps)..."
    READ_GPUS="$("${PYTHON_BIN}" -B "${HERE_DIR}/wait_gpus_author494_pro5000.py" \
      --status-file "${OUTPUT_DIR}/wait_status.json" \
      --log-file "${OUTPUT_DIR}/wait.log" \
      --port "${STUDENT_PORT}" \
      --num-gpus 4 \
      --min-free-mib 70000 \
      --poll-seconds "${POLL_SECONDS}")"
    TUTOR_GPU="${READ_GPUS%% *}"
    STUDENT_GPUS="${READ_GPUS#* }"
  fi

  set +e
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

for idx in [tutor_gpu] + student_gpus:
    if idx not in gpus:
        raise SystemExit(f"ERROR: GPU {idx} does not exist")
    if gpus[idx]["free"] < 70000:
        raise SystemExit(f"ERROR: GPU {idx} free={gpus[idx]['free']}MiB < 70000MiB")
    if busy[idx]:
        raise SystemExit(f"ERROR: GPU {idx} has active compute processes: {busy[idx]}")
print(f"[author494] preflight GPUs OK: tutor={tutor_gpu} student={student_gpus} port={port}")
PY
  PRE_RC=$?
  set -e
  if [ "${PRE_RC}" = "0" ]; then
    break
  fi
  if [ "${AUTO_WAIT}" = "1" ]; then
    echo "[author494] GPU check raced right before launch; returning to wait loop..."
    sleep "${POLL_SECONDS}"
    continue
  fi
  exit "${PRE_RC}"
done

# Ephemeral loopback serving token (never logged or printed).
SERVING_API_KEY="$("${PYTHON_BIN}" -c 'import secrets; print(secrets.token_urlsafe(32))')"
export SERVING_API_KEY
STUDENT_SERVED_NAME="Llama-3.1-8B-Instruct-${USER}-author494"

STUDENT_PID=""
HOLDER_PID=""
MONITOR_PID=""

stop_student_and_holder() {
  set +e
  if [ -n "${HOLDER_PID}" ]; then
    kill -TERM "${HOLDER_PID}" 2>/dev/null || true
    wait "${HOLDER_PID}" 2>/dev/null || true
    HOLDER_PID=""
  fi
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
  CUDA_VISIBLE_DEVICES="${TUTOR_GPU}" "${PYTHON_BIN}" -B "${HERE_DIR}/wait_gpus_author494_pro5000.py" \
    --hold --min-free-mib 70000 > "${OUTPUT_DIR}/tutor_hold.log" 2>&1 &
  HOLDER_PID=$!

  if ! "${PYTHON_BIN}" -B - "${HOLDER_PID}" "${OUTPUT_DIR}/tutor_hold.log" <<'PY'; then
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
    stop_student_and_holder
    if [ "${AUTO_WAIT}" = "1" ]; then
      echo "[author494] Tutor GPU ${TUTOR_GPU} could not be held; returning to wait loop..."
      sleep "${POLL_SECONDS}"
      READ_GPUS="$("${PYTHON_BIN}" -B "${HERE_DIR}/wait_gpus_author494_pro5000.py" \
        --status-file "${OUTPUT_DIR}/wait_status.json" \
        --log-file "${OUTPUT_DIR}/wait.log" \
        --port "${STUDENT_PORT}" \
        --num-gpus 4 \
        --min-free-mib 70000 \
        --poll-seconds "${POLL_SECONDS}")"
      TUTOR_GPU="${READ_GPUS%% *}"
      STUDENT_GPUS="${READ_GPUS#* }"
      continue
    fi
    echo "ERROR: failed to reserve Tutor GPU ${TUTOR_GPU}" >&2
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
                print(f"[author494] student server healthy on 127.0.0.1:{port}")
                sys.exit(0)
    except OSError:
        pass
    time.sleep(5)
raise SystemExit("ERROR: student server health check timed out after 600s")
PY
    stop_student_and_holder
    if [ "${AUTO_WAIT}" = "1" ]; then
      echo "[author494] Student server failed startup (possible GPU race); returning to wait loop..."
      sleep "${POLL_SECONDS}"
      READ_GPUS="$("${PYTHON_BIN}" -B "${HERE_DIR}/wait_gpus_author494_pro5000.py" \
        --status-file "${OUTPUT_DIR}/wait_status.json" \
        --log-file "${OUTPUT_DIR}/wait.log" \
        --port "${STUDENT_PORT}" \
        --num-gpus 4 \
        --min-free-mib 70000 \
        --poll-seconds "${POLL_SECONDS}")"
      TUTOR_GPU="${READ_GPUS%% *}"
      STUDENT_GPUS="${READ_GPUS#* }"
      continue
    fi
    exit 1
  fi

  kill -TERM "${HOLDER_PID}" 2>/dev/null || true
  wait "${HOLDER_PID}" 2>/dev/null || true
  HOLDER_PID=""

  if ! "${PYTHON_BIN}" -B - "${TUTOR_GPU}" <<'PY'; then
import subprocess, sys, time
target = int(sys.argv[1])
for _ in range(15):
    gpu_csv = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    uuid_map = {}
    free_mib = 0
    for line in gpu_csv.strip().splitlines():
        idx_s, uuid, free_s = [x.strip() for x in line.split(",")]
        uuid_map[uuid] = int(idx_s)
        if int(idx_s) == target:
            free_mib = int(free_s)
    apps_csv = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    has_app = any(uuid_map.get(line.split(",")[0].strip()) == target for line in apps_csv.splitlines() if line.strip())
    if free_mib >= 70000 and not has_app:
        sys.exit(0)
    time.sleep(1)
sys.exit(1)
PY
    REP_GPU="$("${PYTHON_BIN}" -B "${HERE_DIR}/wait_gpus_author494_pro5000.py" --find-replacement --exclude "${STUDENT_GPUS}" --min-free-mib 70000 || true)"
    if [ -n "${REP_GPU}" ]; then
      echo "[author494] Tutor GPU ${TUTOR_GPU} became busy during handoff; switching to idle GPU ${REP_GPU}"
      TUTOR_GPU="${REP_GPU}"
    elif [ "${AUTO_WAIT}" = "1" ]; then
      echo "[author494] Tutor GPU ${TUTOR_GPU} became busy and no replacement GPU available; returning to wait loop..."
      stop_student_and_holder
      sleep "${POLL_SECONDS}"
      READ_GPUS="$("${PYTHON_BIN}" -B "${HERE_DIR}/wait_gpus_author494_pro5000.py" \
        --status-file "${OUTPUT_DIR}/wait_status.json" \
        --log-file "${OUTPUT_DIR}/wait.log" \
        --port "${STUDENT_PORT}" \
        --num-gpus 4 \
        --min-free-mib 70000 \
        --poll-seconds "${POLL_SECONDS}")"
      TUTOR_GPU="${READ_GPUS%% *}"
      STUDENT_GPUS="${READ_GPUS#* }"
      continue
    else
      echo "ERROR: Tutor GPU ${TUTOR_GPU} is no longer free (>=70000 MiB)" >&2
      exit 1
    fi
  fi
  break
done

# 60-second GPU process monitor for the 4 selected GPUs.
"${PYTHON_BIN}" -B - "${TUTOR_GPU},${STUDENT_GPUS}" "${OUTPUT_DIR}/gpu_watch.log" <<'PY' &
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

echo "[author494] Starting eval.run_env_test_eval (tag=${TAG}, num_problems=${NUM_PROBLEMS}, tutor_gpu=${TUTOR_GPU}, student_gpus=${STUDENT_GPUS}, agnes_conc=${JUDGE_CONCURRENCY}+${EVAL_JUDGE_CONCURRENCY})..."
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
  --judge-concurrency "${JUDGE_CONCURRENCY}" \
  --eval-judge-concurrency "${EVAL_JUDGE_CONCURRENCY}" \
  --output-dir "${OUTPUT_DIR}" \
  --tag "${TAG}" \
  > "${OUTPUT_DIR}/eval_run.log" 2>&1
EVAL_RC=$?
set -e
END_TS=$(date +%s)
echo "[author494] eval finished tag=${TAG} rc=${EVAL_RC} elapsed=$((END_TS - START_TS))s"

if [ -f "${OUTPUT_DIR}/eval_run.log" ]; then
  "${PYTHON_BIN}" -B - "${HERE_DIR}" "${OUTPUT_DIR}/eval_run.log" "${EVAL_RC}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from summarize_author494_pro5000 import format_eval_log_excerpt, extract_safe_failure_summary

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
  "${PYTHON_BIN}" -B "${HERE_DIR}/summarize_author494_pro5000.py" "${RESULT_JSON}"
fi

exit "${EVAL_RC}"
