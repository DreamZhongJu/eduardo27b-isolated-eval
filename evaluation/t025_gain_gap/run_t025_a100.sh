#!/usr/bin/env bash
# Read-only A100 entry script for T-025 (Causes C, D, E1, G).
# Zero GPUs, zero Agnes calls, never modifies existing evaluation directories.
set -euo pipefail

HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE_DIR}/../.." && pwd)"
A_GROUP_ROOT="${A_GROUP_ROOT:-/workspace/A_group_repro_20261117}"
PY_BIN="${PYTHON_BIN:-${A_GROUP_ROOT}/envs/env-candidate-vllm0191/bin/python}"
if [ ! -x "${PY_BIN}" ]; then
  PY_BIN="/workspace/eduardo-vllm018/bin/python"
fi

DATA_JSONL="${DATA_JSONL:-/workspace/data/math-transfer-test-499-49d3420/data/test.jsonl}"
REF_ROOT="${REF_ROOT:-/workspace/results/own-test-499-v1}"
OVERLAY_ROOT="${OVERLAY_ROOT:-/workspace/results/agnes-topbool-recovery-v1}"
DECISIONS_ROOT="${DECISIONS_ROOT:-/workspace/results/agnes-rejudge-v1}"
STUDENT_TOK_DIR="${STUDENT_TOK_DIR:-/workspace/models/Meta-Llama-3.1-8B-Instruct}"
OUT_DIR="${OUT_DIR:-/workspace/tmp/t025_a100_out}"

T019_DIR="${T019_DIR:-}"
if [ -z "${T019_DIR}" ]; then
  T019_DIR="$(ls -td /workspace/results/*base4b*/ /workspace/results/*t019*/ 2>/dev/null | grep -v pilot | head -n 1 || true)"
  T019_DIR="${T019_DIR%/}"
fi

unset CUDA_VISIBLE_DEVICES HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy GEMINI_API_KEY AGNES_API_KEY || true
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

mkdir -p "${OUT_DIR}"
echo "[t025_a100] python=${PY_BIN} ref_root=${REF_ROOT} t019_dir=${T019_DIR:-NONE}"

BASE4B_ARGS=()
if [ -n "${T019_DIR}" ] && [ -d "${T019_DIR}/AGNES" ]; then
  BASE4B_ARGS=(--base4b-dir "${T019_DIR}")
fi

"${PY_BIN}" -B "${HERE_DIR}/run_t025_a100.py" \
  --repo "${REPO_ROOT}" \
  --data "${DATA_JSONL}" \
  --ref-root "${REF_ROOT}" \
  --overlay "${OVERLAY_ROOT}" \
  --decisions "${DECISIONS_ROOT}" \
  --student-tokenizer-dir "${STUDENT_TOK_DIR}" \
  "${BASE4B_ARGS[@]}" \
  --json-out "${OUT_DIR}/t025_a100_counts.json" \
  | tee "${OUT_DIR}/t025_a100_report.txt"
