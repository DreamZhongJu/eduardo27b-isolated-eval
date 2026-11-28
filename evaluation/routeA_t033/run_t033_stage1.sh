#!/usr/bin/env bash
# T-033 Stage 1 single entrypoint for Pro 5000:
#   --preflight-only : CPU preflight + unit tests + train/dev vs test overlap check + Agnes ping
#   Main GPU run     : Phase 1 (90x32 fp16 vs bf16) -> Phase 2 (Author494 smoke16)
#                      -> Phase 3 (Selfbuilt smoke16) -> Phase 4 (Dev smoke10)
#                      -> Phase 5 (Stage 1 summary + machine-readable fields)
set -euo pipefail
umask 077

usage() {
  cat <<'EOF'
Usage:
  run_t033_stage1.sh --run-dir <dir> --preflight-only
  run_t033_stage1.sh --run-dir <dir> [--max-gpus 4] [--poll-seconds 60] [--resume]
  run_t033_stage1.sh --status [<run_dir>]
EOF
}

HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE_DIR}/../.." && pwd)"
BASE="${PRO5000_BASE:-${HOME}/bzhang/a_group_pro5000}"
ENV_DIR="${PRO5000_ENV_DIR:-${BASE}/envs/pro5000-fast}"
PY_BIN="${ENV_DIR}/bin/python"
EDUARDO_REPO="${EDUARDO_REPO:-${BASE}/source/EduardoRL}"

if [ "${1:-}" = "--status" ]; then
  RDIR="${2:-$(ls -td "${BASE}/runs/t033_stage1_"* 2>/dev/null | head -n 1 || true)}"
  [ -n "${RDIR}" ] && [ -d "${RDIR}" ] || { echo "找不到 t033_stage1 运行目录"; exit 1; }
  echo "RUN_DIR=${RDIR}"
  if [ -f "${RDIR}/stage1.pid" ] && kill -0 "$(cat "${RDIR}/stage1.pid")" 2>/dev/null; then
    echo "PROCESS=RUNNING (pid=$(cat "${RDIR}/stage1.pid"))"
  else
    echo "PROCESS=STOPPED (last_pid=$(cat "${RDIR}/stage1.pid" 2>/dev/null || echo NONE))"
  fi
  for ph in phase1_precision/precision_compare_summary.json phase2_author494_smoke16/author494_summary.json phase3_selfbuilt_smoke16/selfbuilt_summary.json phase4_dev_smoke10/dev_summary.json stage1_summary.json; do
    if [ -f "${RDIR}/${ph}" ]; then
      echo "READY: ${ph}"
    else
      echo "PENDING: ${ph}"
    fi
  done
  tail -n 25 "${RDIR}/stage1_bg.log" 2>/dev/null || true
  exit 0
fi

RUN_DIR=""
PREFLIGHT_ONLY="0"
MAX_GPUS="4"
POLL_SECONDS="60"
RESUME="0"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --preflight-only) PREFLIGHT_ONLY="1"; shift ;;
    --max-gpus) MAX_GPUS="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --resume) RESUME="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

[ -n "${RUN_DIR}" ] || { echo "ERROR: --run-dir required" >&2; exit 2; }
mkdir -p "${RUN_DIR}"

unset LD_LIBRARY_PATH HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy GEMINI_API_KEY CUDA_VISIBLE_DEVICES || true
export LD_LIBRARY_PATH="${ENV_DIR}/lib"
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${EDUARDO_REPO}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMPDIR="${BASE}/tmp" HF_HOME="${BASE}/cache/hf" TRITON_CACHE_DIR="${BASE}/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${BASE}/cache/inductor" VLLM_CACHE_ROOT="${BASE}/cache/vllm" XDG_CACHE_HOME="${BASE}/cache/xdg"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${XDG_CACHE_HOME}"

if [ -z "${AGNES_API_KEY:-}" ] && [ -s "${HOME}/.secrets/agnes_api_key" ]; then
  export AGNES_API_KEY="$(tr -d '\r\n' < "${HOME}/.secrets/agnes_api_key")"
fi

if [ "${PREFLIGHT_ONLY}" = "1" ]; then
  "${PY_BIN}" -B "${HERE_DIR}/preflight_t033.py" --base "${BASE}" --repo "${REPO_ROOT}" --out-json "${RUN_DIR}/preflight.json"
  TRAIN_PQ="$("${PY_BIN}" -B -c "import json,sys; print(json.load(open(sys.argv[1]))['t026_train_parquet'])" "${RUN_DIR}/preflight.json")"
  VAL_PQ="$("${PY_BIN}" -B -c "import json,sys; print(json.load(open(sys.argv[1]))['t026_val_parquet'])" "${RUN_DIR}/preflight.json")"
  "${PY_BIN}" -B "${HERE_DIR}/check_overlap_t033.py" --base "${BASE}" --train-parquet "${TRAIN_PQ}" --val-parquet "${VAL_PQ}" --out-json "${RUN_DIR}/overlap.json"
  "${PY_BIN}" -B "${REPO_ROOT}/evaluation/author494_pro5000/agnes_ping_author494.py" | tee "${RUN_DIR}/agnes_ping.log"
  echo "T033_STAGE1_PREFLIGHT_DONE=0"
  exit 0
fi

echo "$$" > "${RUN_DIR}/stage1.pid"

if [ ! -f "${RUN_DIR}/preflight.json" ]; then
  "${PY_BIN}" -B "${HERE_DIR}/preflight_t033.py" --base "${BASE}" --repo "${REPO_ROOT}" --out-json "${RUN_DIR}/preflight.json" --skip-tests
fi
if [ ! -f "${RUN_DIR}/overlap.json" ]; then
  TRAIN_PQ="$("${PY_BIN}" -B -c "import json,sys; print(json.load(open(sys.argv[1]))['t026_train_parquet'])" "${RUN_DIR}/preflight.json")"
  VAL_PQ="$("${PY_BIN}" -B -c "import json,sys; print(json.load(open(sys.argv[1]))['t026_val_parquet'])" "${RUN_DIR}/preflight.json")"
  "${PY_BIN}" -B "${HERE_DIR}/check_overlap_t033.py" --base "${BASE}" --train-parquet "${TRAIN_PQ}" --val-parquet "${VAL_PQ}" --out-json "${RUN_DIR}/overlap.json"
fi

# Phase 1: 90x32 Student fp16 vs bf16 precision comparison (1 GPU, 0 Agnes)
P1_JSON="${RUN_DIR}/phase1_precision/precision_compare_summary.json"
if [ ! -f "${P1_JSON}" ]; then
  echo "=== [T-033 Stage 1] Phase 1: Student 90x32 fp16 vs bf16 ==="
  "${PY_BIN}" -B "${HERE_DIR}/precision_compare_selfbuilt.py" \
    --base "${BASE}" --repo "${REPO_ROOT}" \
    --out-dir "${RUN_DIR}/phase1_precision" --auto-wait --poll-seconds "${POLL_SECONDS}"
fi

# Phase 2: Author 494 smoke 16 (4 GPUs if MAX_GPUS>=4 else 3 GPUs, bf16, Agnes <= 16)
P2_JSON="${RUN_DIR}/phase2_author494_smoke16/author494_summary.json"
if [ ! -f "${P2_JSON}" ]; then
  echo "=== [T-033 Stage 1] Phase 2: Author 494 smoke 16 ==="
  SDP=3
  if [ "${MAX_GPUS}" -lt 4 ]; then SDP=2; fi
  RESUME_OPT=()
  [ "${RESUME}" = "1" ] && RESUME_OPT=(--resume)
  bash "${HERE_DIR}/run_t033_author494.sh" \
    --tutor-model "${BASE}/runs/t026_stage3_formal_20261126_140508/exports/step_100" \
    --output-dir "${RUN_DIR}/phase2_author494_smoke16" \
    --auto-wait --student-dp "${SDP}" --num-problems 16 --max-agnes-window 16 \
    --poll-seconds "${POLL_SECONDS}" "${RESUME_OPT[@]}"
fi

# Phase 3: Self-built 3-tier smoke 16 (2 GPUs, bf16, Agnes <= 16)
P3_JSON="${RUN_DIR}/phase3_selfbuilt_smoke16/selfbuilt_summary.json"
if [ ! -f "${P3_JSON}" ]; then
  echo "=== [T-033 Stage 1] Phase 3: Self-built 3-tier smoke 16 ==="
  bash "${HERE_DIR}/run_t033_selfbuilt.sh" \
    --tutor-model "${BASE}/runs/t026_stage3_formal_20261126_140508/exports/step_100" \
    --output-dir "${RUN_DIR}/phase3_selfbuilt_smoke16" \
    --auto-wait --smoke-total 16 --agnes-workers 16 --poll-seconds "${POLL_SECONDS}"
fi

# Phase 4: Dev set smoke 10 (4 GPUs, bf16, 0 Agnes)
P4_JSON="${RUN_DIR}/phase4_dev_smoke10/dev_summary.json"
if [ ! -f "${P4_JSON}" ]; then
  echo "=== [T-033 Stage 1] Phase 4: Dev set smoke 10 ==="
  bash "${HERE_DIR}/run_t033_dev_sim.sh" \
    --run-dir "${RUN_DIR}/phase4_dev_smoke10" \
    --max-sources 10 --max-gpus "${MAX_GPUS}" --poll-seconds "${POLL_SECONDS}"
fi

# Phase 5: Aggregate numeric-only Stage 1 summary and emit fixed summary block
"${PY_BIN}" -B - "${RUN_DIR}" "${HERE_DIR}" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from common import format_machine_summary, write_json_atomic

rdir = Path(sys.argv[1])
pre = json.loads((rdir / "preflight.json").read_text(encoding="utf-8"))
ov = json.loads((rdir / "overlap.json").read_text(encoding="utf-8"))
p1 = json.loads((rdir / "phase1_precision/precision_compare_summary.json").read_text(encoding="utf-8"))
p2 = json.loads((rdir / "phase2_author494_smoke16/author494_summary.json").read_text(encoding="utf-8"))
p3 = json.loads((rdir / "phase3_selfbuilt_smoke16/selfbuilt_summary.json").read_text(encoding="utf-8"))
p4 = json.loads((rdir / "phase4_dev_smoke10/dev_summary.json").read_text(encoding="utf-8"))

calls = int(p1["total_calls"]) + int(p2["agnes_judge"]["total_votes"]) + int((p3.get("agnes_report") or {}).get("total_calls", 0)) + int(p4["grader_llm_calls"])
fails = int(p1["failures"]) + int(p2["agnes_judge"]["retry_failed_votes"]) + int(p2["metrics"]["n_student_calls_failed"]) + int((p3.get("agnes_report") or {}).get("technical_failures", 0)) + int(p3["student_calls_failed"]) + int(p4["grader_tech_failed"]) + int(p4["student_failed"])

p4_est_full_h = round((float(p4["elapsed_s"]) / max(1, int(p4["n_sources"]))) * 100.0 / 3600.0, 2)
summary = {
    "status": "STAGE1_DONE",
    "rc": 0,
    "calls": calls,
    "failures": fails,
    "preflight": {"routea_step100_files": pre["routea_step100_files"], "math_verify": pre["packages"]["math-verify"], "latex2sympy2_extended": pre["packages"]["latex2sympy2-extended"]},
    "overlap": ov,
    "precision_compare": p1,
    "smoke_author494": p2,
    "smoke_selfbuilt": p3,
    "smoke_dev": p4,
    "stage2_estimate": {
        "author494_494": {"gpus": 4, "agnes_calls_approx": 3952, "est_hours_from_t020_t022": "1.5-6.2h (AIMD <=16)"},
        "selfbuilt_1489": {"gpus_per_pair": 2, "agnes_calls": 47648, "est_gen_hours_2gpu": "11-15h (or ~6-8h with 2x2=4 GPUs)"},
        "dev_100": {"gpus": 4, "agnes_calls": 0, "est_hours_extrapolated": p4_est_full_h},
    },
}
write_json_atomic(rdir / "stage1_summary.json", summary)
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
ov_all = p1["overall_all_32"]
og = p1["orig_16"]
tr = p1["trans_16"]
print(
    format_machine_summary(
        {
            "status": "STAGE1_DONE",
            "rc": 0,
            "calls": calls,
            "failures": fails,
            "overlap_total": ov["overlap_total"],
            "prec_diff_all": f"{ov_all['diff_bf16_minus_fp16']:+.6f}",
            "prec_ci_all": f"[{ov_all['ci95_low']:+.6f},{ov_all['ci95_high']:+.6f}]",
            "prec_r_all": ov_all["pearson_r"],
            "prec_diff_orig": f"{og['diff_bf16_minus_fp16']:+.6f}",
            "prec_diff_trans": f"{tr['diff_bf16_minus_fp16']:+.6f}",
            "smoke_author494_n": p2["num_problems"],
            "smoke_selfbuilt_n": p3["gen_complete"],
            "smoke_dev_n": p4["n_sources"],
        }
    ),
    flush=True,
)
PY
