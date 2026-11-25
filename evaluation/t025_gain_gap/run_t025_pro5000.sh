#!/usr/bin/env bash
# Read-only Pro 5000 entry script for T-025 (Causes A-G).
# Zero GPUs, zero Agnes calls, writes only to $BASE/runs/t025_gain_gap_<ts>/.
set -euo pipefail

HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE_DIR}/../.." && pwd)"
BASE="${PRO5000_BASE:-${HOME}/bzhang/a_group_pro5000}"
ENV_DIR="${PRO5000_ENV_DIR:-${BASE}/envs/pro5000-fast}"
PY_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python}"

BUNDLE_DIR="${BUNDLE_DIR:-${BASE}/t021_bundle}"
T019_CSV="${T019_CSV:-${BASE}/t019_counts/base4b_counts.csv}"
T018_CSV="${T018_CSV:-${BASE}/t018_bundle/t018_per_problem.csv}"
TEST_PARQUET="${TEST_PARQUET:-${BASE}/data/author494_eval/test.parquet}"

AGROUP_DIR="${AGROUP_DIR:-}"
if [ -z "${AGROUP_DIR}" ]; then
  AGROUP_DIR="$(ls -td "${BASE}/runs/t021_stage2_full_"* 2>/dev/null | head -n 1 || true)"
fi
T022_JSON="${T022_JSON:-}"
if [ -z "${T022_JSON}" ]; then
  T022_JSON="$(ls -t "${BASE}/runs/t022_27b_pro5000_"*/full494/transfer_eval_*.json 2>/dev/null | head -n 1 || true)"
fi

STUDENT_TOK_DIR=""
for cand in "/dev/shm/zhujie_a_group/Meta-Llama-3.1-8B-Instruct" "${BASE}/models/Meta-Llama-3.1-8B-Instruct"; do
  if [ -f "${cand}/tokenizer.json" ]; then
    STUDENT_TOK_DIR="${cand}"
    break
  fi
done

if [ -z "${AGROUP_DIR}" ] || [ ! -d "${AGROUP_DIR}/AGNES" ]; then
  echo "STOP: T-021 full directory not found: ${AGROUP_DIR:-EMPTY}" >&2
  exit 2
fi
if [ -z "${T022_JSON}" ] || [ ! -f "${T022_JSON}" ]; then
  echo "STOP: T-022 full494 result JSON not found: ${T022_JSON:-EMPTY}" >&2
  exit 2
fi
if [ -z "${STUDENT_TOK_DIR}" ]; then
  echo "STOP: Meta-Llama-3.1-8B-Instruct tokenizer.json not found" >&2
  exit 2
fi

unset CUDA_VISIBLE_DEVICES HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy GEMINI_API_KEY AGNES_API_KEY || true
export CUDA_VISIBLE_DEVICES=""
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="${BASE}/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

OUT_DIR="${OUT_DIR:-${BASE}/runs/t025_gain_gap_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${TMPDIR}" "${OUT_DIR}"

T018_ARGS=()
if [ -f "${T018_CSV}" ]; then
  T018_ARGS=(--t018-table "${T018_CSV}")
fi
A100_ARGS=()
if [ -n "${A100_JSON:-}" ] && [ -f "${A100_JSON}" ]; then
  A100_ARGS=(--a100-json "${A100_JSON}")
fi

echo "[t025_pro5000] out_dir=${OUT_DIR} agroup_dir=${AGROUP_DIR} t022_json=${T022_JSON}"
"${PY_BIN}" -B "${HERE_DIR}/run_t025_pro5000.py" \
  --bundle-dir "${BUNDLE_DIR}" \
  --agroup-dir "${AGROUP_DIR}" \
  --base4b-counts-csv "${T019_CSV}" \
  "${T018_ARGS[@]}" \
  "${A100_ARGS[@]}" \
  --t022-json "${T022_JSON}" \
  --test-parquet "${TEST_PARQUET}" \
  --student-tokenizer-dir "${STUDENT_TOK_DIR}" \
  --json-out "${OUT_DIR}/t025_pro5000_summary.json" \
  | tee "${OUT_DIR}/t025_pro5000_report.txt"
