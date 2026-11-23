#!/usr/bin/env bash
# Launch 1-GPU Tutor (Qwen3.5-4B) + 1-GPU Student (Llama-3.1-8B-Instruct, vLLM 0.18.0 float16)
# and run post-only evaluation reusing 27B pre/control records. Aggregate status/logs only.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_base4b_selfbuilt.sh (--tutor-gpu <id> --student-gpu <id> | --auto-wait) \
    --output-dir <dir> [--mode pilot|full] [--pilot-per-tier 3] \
    [--concurrency 8] [--agnes-workers 2] [--gate-first-n <auto>] [--max-est-days 3.0] [--resume] \
    [--teacher-port 18701] [--student-port 18702] [--poll-seconds 60] \
    [--skip-grading] [--skip-author-rule]

  run_base4b_selfbuilt.sh --grade-only <output_dir> [--mode pilot|full] [--pilot-per-tier 3] [--agnes-workers 2]
  run_base4b_selfbuilt.sh --status <output_dir>
EOF
}

HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE_DIR}/../.." && pwd)"

A_GROUP_ROOT="${A_GROUP_ROOT:-/workspace/A_group_repro_20261117}"
TUTOR_PY="${TUTOR_PY:-${A_GROUP_ROOT}/envs/env-candidate-vllm0191/bin/python}"
TUTOR_VLLM="${TUTOR_VLLM:-$(dirname "${TUTOR_PY}")/vllm}"
STUDENT_PY="${STUDENT_PY:-/workspace/eduardo-vllm018/bin/python}"
STUDENT_VLLM="${STUDENT_VLLM:-$(dirname "${STUDENT_PY}")/vllm}"
TUTOR_MODEL="${TUTOR_MODEL:-${A_GROUP_ROOT}/models/Qwen3.5-4B}"
STUDENT_MODEL="${STUDENT_MODEL:-/workspace/models/Meta-Llama-3.1-8B-Instruct}"
DATA_FILE="${DATA_FILE:-/workspace/data/math-transfer-test-499-49d3420/data/test.jsonl}"
REF_ROOT="${REF_ROOT:-/workspace/results/own-test-499-v1}"
OVERLAY_ROOT="${OVERLAY_ROOT:-/workspace/results/agnes-topbool-recovery-v1}"
DECISIONS_ROOT="${DECISIONS_ROOT:-/workspace/results/agnes-rejudge-v1}"
AGNES_KEY_FILE="${AGNES_KEY_FILE:-/workspace/.secrets/agnes_api_key}"

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
  python3 -B - "${OUT_DIR}" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
gate_p = out / "HEAD16_GATE.json"
if gate_p.is_file():
    g = json.loads(gate_p.read_text(encoding="utf-8"))
    print(
        f"head_gate passed={g.get('passed')} complete={g.get('complete')}/{g.get('gate_n')} "
        f"held={g.get('held')} stu_fail={g.get('student_calls_failed')} "
        f"sec_per_source={g.get('sec_per_source')} est_full_days={g.get('est_full_days')}"
    )
for stage in ("EASY", "MEDIUM", "HARD"):
    base = out / stage / "sources"
    if not base.is_dir():
        print(f"stage={stage} gen_complete=0 held=0 partial=0 | agnes_graded=0 | author_rule=0")
        continue
    dirs = [p for p in base.iterdir() if p.is_dir()]
    comp = sum(1 for p in dirs if (p / "RAW_COMPLETE.json").is_file())
    held = sum(1 for p in dirs if (p / "SOURCE_HELD.json").is_file())
    part = len(dirs) - comp - held
    g_dir = out / "AGNES" / stage / "sources"
    graded = len(list(g_dir.glob("*/RESULTS.json"))) if g_dir.is_dir() else 0
    ar_dir = out / "AUTHOR_RULE" / stage / "sources"
    ar_graded = len(list(ar_dir.glob("*/RESULTS.json"))) if ar_dir.is_dir() else 0
    print(f"stage={stage} gen_complete={comp} held={held} partial={part} | agnes_graded={graded} | author_rule={ar_graded}")
PY
  if [ -f "${OUT_DIR}/eval_run.log" ]; then
    python3 -B - "${HERE_DIR}" "${OUT_DIR}/eval_run.log" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from summarize_base4b import extract_safe_log_lines
lines = extract_safe_log_lines(Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace"))
print(f"status_lines_matched={len(lines)}")
for ln in lines[-20:]:
    print(" ", ln)
PY
  fi
  if [ -f "${OUT_DIR}/summary_agnes_report.txt" ]; then
    grep -E "^\[base4b_summary\] " "${OUT_DIR}/summary_agnes_report.txt" || true
  fi
  if [ -f "${OUT_DIR}/gpu_watch.log" ]; then
    ALERTS=$(grep -c "NEW_HOST_PID_ALERT" "${OUT_DIR}/gpu_watch.log" || true)
    echo "gpu_watch_new_host_pid_alerts=${ALERTS}"
  fi
  exit 0
fi

TUTOR_GPU=""
STUDENT_GPU=""
AUTO_WAIT="0"
POLL_SECONDS="60"
OUTPUT_DIR=""
MODE="pilot"
PILOT_PER_TIER="3"
CONCURRENCY="8"
AGNES_WORKERS="${AGNES_WORKERS:-2}"
GATE_FIRST_N="-1"
MAX_EST_DAYS="3.0"
RESUME="0"
TEACHER_PORT="18701"
STUDENT_PORT="18702"
SKIP_GRADING="0"
SKIP_AUTHOR_RULE="0"
GRADE_ONLY_DIR=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tutor-gpu) TUTOR_GPU="$2"; shift 2 ;;
    --student-gpu) STUDENT_GPU="$2"; shift 2 ;;
    --auto-wait) AUTO_WAIT="1"; shift ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --pilot-per-tier) PILOT_PER_TIER="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --agnes-workers) AGNES_WORKERS="$2"; shift 2 ;;
    --gate-first-n) GATE_FIRST_N="$2"; shift 2 ;;
    --max-est-days) MAX_EST_DAYS="$2"; shift 2 ;;
    --resume) RESUME="1"; shift ;;
    --teacher-port) TEACHER_PORT="$2"; shift 2 ;;
    --student-port) STUDENT_PORT="$2"; shift 2 ;;
    --skip-grading) SKIP_GRADING="1"; shift ;;
    --skip-author-rule) SKIP_AUTHOR_RULE="1"; shift ;;
    --grade-only) GRADE_ONLY_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy GEMINI_API_KEY
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 WANDB_MODE=disabled
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

if [ "${MODE}" = "full" ]; then
  PILOT_PER_TIER="0"
  if [ "${GATE_FIRST_N}" = "-1" ]; then GATE_FIRST_N="16"; fi
elif [ "${MODE}" = "pilot" ]; then
  if [ "${GATE_FIRST_N}" = "-1" ]; then GATE_FIRST_N="0"; fi
else
  echo "ERROR: --mode must be pilot or full" >&2
  exit 2
fi

if [ -n "${GRADE_ONLY_DIR}" ]; then
  test -s "${AGNES_KEY_FILE}"
  "${TUTOR_PY}" -B "${HERE_DIR}/run_post_eval.py" \
    --repo "${REPO_ROOT}" --data "${DATA_FILE}" \
    --ref-root "${REF_ROOT}" --overlay "${OVERLAY_ROOT}" --decisions "${DECISIONS_ROOT}" \
    --out-root "${GRADE_ONLY_DIR}" --key-file "${AGNES_KEY_FILE}" \
    --pilot-per-tier "${PILOT_PER_TIER}" --workers "${AGNES_WORKERS}" --run-grading \
    $( [ "${SKIP_AUTHOR_RULE}" = "0" ] && echo "--run-author-rule" )
  "${TUTOR_PY}" -B "${HERE_DIR}/summarize_base4b.py" \
    --repo "${REPO_ROOT}" --data "${DATA_FILE}" \
    --ref-root "${REF_ROOT}" --overlay "${OVERLAY_ROOT}" --decisions "${DECISIONS_ROOT}" \
    --out-root "${GRADE_ONLY_DIR}" --json-out "${GRADE_ONLY_DIR}/summary_agnes.json" \
    | tee "${GRADE_ONLY_DIR}/summary_agnes_report.txt"
  exit 0
fi

if [ -z "${OUTPUT_DIR}" ]; then
  echo "ERROR: --output-dir is required" >&2
  exit 2
fi
if [ "${AUTO_WAIT}" = "0" ] && { [ -z "${TUTOR_GPU}" ] || [ -z "${STUDENT_GPU}" ]; }; then
  echo "ERROR: provide --auto-wait OR both --tutor-gpu and --student-gpus" >&2
  exit 2
fi
if [ "${SKIP_GRADING}" = "0" ] && [ ! -s "${AGNES_KEY_FILE}" ]; then
  echo "ERROR: Agnes key file missing or empty: ${AGNES_KEY_FILE}" >&2
  exit 2
fi
if [ "${RESUME}" = "0" ] && [ -e "${OUTPUT_DIR}" ] && [ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null || true)" ]; then
  echo "ERROR: output-dir ${OUTPUT_DIR} already exists and is non-empty; pass --resume or choose a new dir" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"
echo "$$" > "${OUTPUT_DIR}/runner.pid"

SERVING_API_KEY="loopback-only"
STUDENT_PID=""
TEACHER_PID=""
HOLDER_PID=""
MONITOR_PID=""

kill_descendants() {
  local root_pid=$1
  [ -n "${root_pid}" ] || return 0
  "${TUTOR_PY}" -B - "${root_pid}" <<'PY' || true
import os, signal, sys, time
from pathlib import Path
root_pid = int(sys.argv[1])
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
out, stack = [], [root_pid]
while stack:
    cur = stack.pop()
    out.append(cur)
    stack.extend(children_map.get(cur, []))
for pid in reversed(out):
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
time.sleep(5)
for pid in reversed(out):
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
PY
  wait "${root_pid}" 2>/dev/null || true
}

cleanup() {
  set +e
  if [ -n "${MONITOR_PID}" ] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  if [ -n "${HOLDER_PID}" ]; then
    kill -TERM "${HOLDER_PID}" 2>/dev/null || true
    wait "${HOLDER_PID}" 2>/dev/null || true
    HOLDER_PID=""
  fi
  kill_descendants "${TEACHER_PID}"
  TEACHER_PID=""
  kill_descendants "${STUDENT_PID}"
  STUDENT_PID=""
}
trap cleanup EXIT INT TERM

while true; do
  if [ "${AUTO_WAIT}" = "1" ]; then
    echo "[base4b] Waiting for 2 idle GPUs (>=78000 MiB free, util=0, 0 apps)..."
    READ_GPUS="$("${TUTOR_PY}" -B "${HERE_DIR}/wait_gpus_base4b.py" \
      --status-file "${OUTPUT_DIR}/wait_status.json" \
      --log-file "${OUTPUT_DIR}/wait.log" \
      --ports "${TEACHER_PORT},${STUDENT_PORT}" \
      --num-gpus 2 \
      --min-free-mib 78000 \
      --poll-seconds "${POLL_SECONDS}")"
    TUTOR_GPU="${READ_GPUS%% *}"
    STUDENT_GPU="${READ_GPUS#* }"
  fi

  # Hold Tutor GPU while Student vLLM server starts up so another job doesn't grab it.
  CUDA_VISIBLE_DEVICES="${TUTOR_GPU}" "${TUTOR_PY}" -B "${HERE_DIR}/wait_gpus_base4b.py" \
    --hold --min-free-mib 78000 > "${OUTPUT_DIR}/tutor_hold.log" 2>&1 &
  HOLDER_PID=$!

  if ! "${TUTOR_PY}" -B - "${HOLDER_PID}" "${OUTPUT_DIR}/tutor_hold.log" <<'PY'; then
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
    cleanup
    if [ "${AUTO_WAIT}" = "1" ]; then
      sleep "${POLL_SECONDS}"
      continue
    fi
    echo "ERROR: failed to reserve Tutor GPU ${TUTOR_GPU}" >&2
    exit 1
  fi

  # Start Student server using exact 27B vLLM 0.18.0 + float16 flags.
  CUDA_VISIBLE_DEVICES="${STUDENT_GPU}" VLLM_API_KEY="${SERVING_API_KEY}" \
    "${STUDENT_VLLM}" serve "${STUDENT_MODEL}" \
    --host 127.0.0.1 \
    --port "${STUDENT_PORT}" \
    --served-model-name "Llama-3.1-8B-Instruct" \
    --api-key "${SERVING_API_KEY}" \
    --dtype float16 \
    --max-model-len 20000 \
    --gpu-memory-utilization 0.90 \
    --seed 42 \
    > "${OUTPUT_DIR}/student_server.log" 2>&1 &
  STUDENT_PID=$!

  if ! "${TUTOR_PY}" -B - "${STUDENT_PID}" "${STUDENT_PORT}" <<'PY'; then
import os, sys, time, urllib.request
pid, port = int(sys.argv[1]), int(sys.argv[2])
deadline = time.monotonic() + 600
url = f"http://127.0.0.1:{port}/health"
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        sys.exit(1)
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            if resp.status == 200:
                print(f"[base4b] student server healthy on 127.0.0.1:{port}")
                sys.exit(0)
    except OSError:
        pass
    time.sleep(5)
sys.exit(1)
PY
    cleanup
    if [ "${AUTO_WAIT}" = "1" ]; then
      sleep "${POLL_SECONDS}"
      continue
    fi
    echo "ERROR: student server failed startup" >&2
    exit 1
  fi

  # Release Tutor GPU holder and start Tutor Qwen3.5-4B server with exact 27B flags.
  kill -TERM "${HOLDER_PID}" 2>/dev/null || true
  wait "${HOLDER_PID}" 2>/dev/null || true
  HOLDER_PID=""

  CUDA_VISIBLE_DEVICES="${TUTOR_GPU}" VLLM_API_KEY="${SERVING_API_KEY}" \
    "${TUTOR_VLLM}" serve "${TUTOR_MODEL}" \
    --host 127.0.0.1 \
    --port "${TEACHER_PORT}" \
    --served-model-name "Qwen3.5-4B" \
    --api-key "${SERVING_API_KEY}" \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len 20000 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 8 \
    --enforce-eager \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --seed 42 \
    > "${OUTPUT_DIR}/teacher_server.log" 2>&1 &
  TEACHER_PID=$!

  if ! "${TUTOR_PY}" -B - "${TEACHER_PID}" "${TEACHER_PORT}" <<'PY'; then
import os, sys, time, urllib.request
pid, port = int(sys.argv[1]), int(sys.argv[2])
deadline = time.monotonic() + 600
url = f"http://127.0.0.1:{port}/health"
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        sys.exit(1)
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            if resp.status == 200:
                print(f"[base4b] teacher server healthy on 127.0.0.1:{port}")
                sys.exit(0)
    except OSError:
        pass
    time.sleep(5)
sys.exit(1)
PY
    cleanup
    if [ "${AUTO_WAIT}" = "1" ]; then
      sleep "${POLL_SECONDS}"
      continue
    fi
    echo "ERROR: teacher server failed startup" >&2
    exit 1
  fi
  break
done

# 60s host-PID GPU watcher (alerts only if an unexpected new host PID appears).
"${TUTOR_PY}" -B - "${TUTOR_GPU},${STUDENT_GPU}" "${OUTPUT_DIR}/gpu_watch.log" <<'PY' &
import subprocess, sys, time
from pathlib import Path
watched = {int(x) for x in sys.argv[1].split(",")}
log_p = Path(sys.argv[2])
uuid_to_idx = {}
for line in subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True).strip().splitlines():
    idx, uuid = [x.strip() for x in line.split(",")]
    uuid_to_idx[uuid] = int(idx)
baseline = {}
while True:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], text=True).strip()
    cur = {i: set() for i in watched}
    entries, alerts = [], []
    if apps:
        for line in apps.splitlines():
            uuid, pid_s, mem = [x.strip() for x in line.split(",")]
            idx = uuid_to_idx.get(uuid)
            if idx in watched:
                cur[idx].add(pid_s)
                entries.append(f"gpu{idx}:pid={pid_s}({mem}MiB)")
                if idx in baseline and pid_s not in baseline[idx]:
                    alerts.append(f"gpu{idx}:pid={pid_s}")
    for idx, pids in cur.items():
        if idx not in baseline and pids:
            baseline[idx] = set(pids)
    with log_p.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] apps=[{', '.join(entries)}]\n")
        if alerts:
            f.write(f"[{ts}] NEW_HOST_PID_ALERT={alerts}\n")
    time.sleep(60)
PY
MONITOR_PID=$!

echo "[base4b] Starting generation (mode=${MODE}, pilot_per_tier=${PILOT_PER_TIER}, concurrency=${CONCURRENCY}, gate_first_n=${GATE_FIRST_N}, max_est_days=${MAX_EST_DAYS}, tutor_gpu=${TUTOR_GPU}, student_gpu=${STUDENT_GPU})..."
START_TS=$(date +%s)
set +e
"${TUTOR_PY}" -B "${HERE_DIR}/run_post_eval.py" \
  --repo "${REPO_ROOT}" --data "${DATA_FILE}" \
  --ref-root "${REF_ROOT}" --overlay "${OVERLAY_ROOT}" --decisions "${DECISIONS_ROOT}" \
  --out-root "${OUTPUT_DIR}" \
  --teacher-model "${TUTOR_MODEL}" --teacher-served-name "Qwen3.5-4B" \
  --teacher-url "http://127.0.0.1:${TEACHER_PORT}/v1" \
  --student-url "http://127.0.0.1:${STUDENT_PORT}/v1" \
  --pilot-per-tier "${PILOT_PER_TIER}" \
  --concurrency "${CONCURRENCY}" \
  --gate-first-n "${GATE_FIRST_N}" \
  --max-est-days "${MAX_EST_DAYS}" \
  --run-generation \
  $( [ "${SKIP_AUTHOR_RULE}" = "0" ] && echo "--run-author-rule" ) \
  >> "${OUTPUT_DIR}/eval_run.log" 2>&1
GEN_RC=$?
set -e
GEN_END_TS=$(date +%s)
GEN_ELAPSED=$((GEN_END_TS - START_TS))
echo "[base4b] generation_finished rc=${GEN_RC} gen_elapsed=${GEN_ELAPSED}s"

# Stop GPU servers before Agnes grading so GPUs are immediately freed!
kill_descendants "${TEACHER_PID}"; TEACHER_PID=""
kill_descendants "${STUDENT_PID}"; STUDENT_PID=""
if [ -n "${MONITOR_PID}" ] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
  MONITOR_PID=""
fi

GRADE_RC=0
GRADE_ELAPSED=0
if [ "${GEN_RC}" = "0" ] && [ "${SKIP_GRADING}" = "0" ]; then
  echo "[base4b] Starting Agnes grading (1 process, workers=${AGNES_WORKERS})..."
  GRADE_START_TS=$(date +%s)
  set +e
  "${TUTOR_PY}" -B "${HERE_DIR}/run_post_eval.py" \
    --repo "${REPO_ROOT}" --data "${DATA_FILE}" \
    --ref-root "${REF_ROOT}" --overlay "${OVERLAY_ROOT}" --decisions "${DECISIONS_ROOT}" \
    --out-root "${OUTPUT_DIR}" --key-file "${AGNES_KEY_FILE}" \
    --pilot-per-tier "${PILOT_PER_TIER}" --workers "${AGNES_WORKERS}" --run-grading \
    >> "${OUTPUT_DIR}/eval_run.log" 2>&1
  GRADE_RC=$?
  set -e
  GRADE_END_TS=$(date +%s)
  GRADE_ELAPSED=$((GRADE_END_TS - GRADE_START_TS))
  echo "[base4b] grading_finished rc=${GRADE_RC} grade_elapsed=${GRADE_ELAPSED}s"
fi

TOTAL_ELAPSED=$(( $(date +%s) - START_TS ))
echo "[base4b] total_elapsed=${TOTAL_ELAPSED}s"

if [ -f "${OUTPUT_DIR}/eval_run.log" ]; then
  "${TUTOR_PY}" -B - "${HERE_DIR}" "${OUTPUT_DIR}/eval_run.log" "${GEN_RC}" "${GRADE_RC}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from summarize_base4b import extract_safe_log_lines, extract_safe_failure_summary
text = Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace")
for ln in extract_safe_log_lines(text):
    print(ln)
if int(sys.argv[3]) != 0 or int(sys.argv[4]) != 0:
    print("=== Sanitized Failure Summary ===")
    print(extract_safe_failure_summary(text))
PY
fi

if [ "${GEN_RC}" = "0" ] && [ "${GRADE_RC}" = "0" ] && [ "${SKIP_GRADING}" = "0" ]; then
  "${TUTOR_PY}" -B "${HERE_DIR}/summarize_base4b.py" \
    --repo "${REPO_ROOT}" --data "${DATA_FILE}" \
    --ref-root "${REF_ROOT}" --overlay "${OVERLAY_ROOT}" --decisions "${DECISIONS_ROOT}" \
    --out-root "${OUTPUT_DIR}" --elapsed-seconds "${TOTAL_ELAPSED}" \
    --gen-elapsed-seconds "${GEN_ELAPSED}" --grade-elapsed-seconds "${GRADE_ELAPSED}" \
    --json-out "${OUTPUT_DIR}/summary_agnes.json" \
    | tee "${OUTPUT_DIR}/summary_agnes_report.txt"
  if [ "${SKIP_AUTHOR_RULE}" = "0" ]; then
    "${TUTOR_PY}" -B "${HERE_DIR}/summarize_base4b.py" \
      --repo "${REPO_ROOT}" --data "${DATA_FILE}" \
      --ref-root "${REF_ROOT}" --overlay "${OVERLAY_ROOT}" --decisions "${DECISIONS_ROOT}" \
      --out-root "${OUTPUT_DIR}" --grade-subdir AUTHOR_RULE \
      --json-out "${OUTPUT_DIR}/summary_author_rule.json" \
      | tee "${OUTPUT_DIR}/summary_author_rule_report.txt"
  fi
fi

if [ "${GEN_RC}" != "0" ]; then exit "${GEN_RC}"; fi
exit "${GRADE_RC}"
