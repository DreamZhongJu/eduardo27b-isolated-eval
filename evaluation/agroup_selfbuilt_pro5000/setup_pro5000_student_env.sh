#!/usr/bin/env bash
# Build isolated 27B Student environment on Pro 5000 under ~/bzhang/a_group_pro5000/envs/pro5000-student-vllm018.
# Requires exact versions: vllm==0.18.0, torch==2.10.0, transformers==4.57.6, math-verify==0.9.0.
# Fails closed (never substitutes another vLLM version) if vLLM 0.18.0 cannot install/run on Blackwell.
set -euo pipefail

BASE="${PRO5000_BASE:-${HOME}/bzhang/a_group_pro5000}"
STUDENT_ENV_DIR="${PRO5000_STUDENT_ENV_DIR:-${BASE}/envs/pro5000-student-vllm018}"
TUTOR_ENV_DIR="${PRO5000_ENV_DIR:-${BASE}/envs/pro5000-fast}"
TMP_DIR="${BASE}/tmp"
PIP_CACHE="${BASE}/cache/pip"
mkdir -p "${TMP_DIR}" "${PIP_CACHE}" "$(dirname "${STUDENT_ENV_DIR}")"
export TMPDIR="${TMP_DIR}"
export PIP_CACHE_DIR="${PIP_CACHE}"

_verify_pkgs() {
  local py="${STUDENT_ENV_DIR}/bin/python"
  [ -x "${py}" ] || return 1
  LD_LIBRARY_PATH="${STUDENT_ENV_DIR}/lib:${LD_LIBRARY_PATH:-}" "${py}" -B - <<'PY'
import importlib.metadata as md
import torch, transformers, vllm
req = {
    "vllm": "0.18.0",
    "torch": "2.10.0",
    "transformers": "4.57.6",
    "math-verify": "0.9.0",
}
bad = []
for name, prefix in req.items():
    got = md.version(name)
    print(f"student_pkg/{name}={got}")
    if not got.startswith(prefix):
        bad.append(f"{name}:{got}!={prefix}")
import openai
print(f"student_pkg/openai={openai.__version__}")
print(f"student_pkg/cuda_available={torch.cuda.is_available()}")
if bad:
    raise SystemExit("MISMATCH:" + ",".join(bad))
PY
}

if [ -x "${STUDENT_ENV_DIR}/bin/python" ]; then
  echo "[setup_student] Found existing ${STUDENT_ENV_DIR}, verifying..."
  if _verify_pkgs; then
    echo "[setup_student] Existing student env verified OK."
    exit 0
  fi
  echo "[setup_student] Existing env incomplete/mismatched; installing exact packages into ${STUDENT_ENV_DIR}..."
else
  CONDA_BIN="$(command -v conda || echo "")"
  if [ -n "${CONDA_BIN}" ]; then
    "${CONDA_BIN}" create -y -p "${STUDENT_ENV_DIR}" python=3.12 pip
  elif [ -x "${TUTOR_ENV_DIR}/bin/python" ]; then
    "${TUTOR_ENV_DIR}/bin/python" -m venv "${STUDENT_ENV_DIR}"
  else
    python3 -m venv "${STUDENT_ENV_DIR}"
  fi
fi

PIP="${STUDENT_ENV_DIR}/bin/pip"
"${PIP}" install --upgrade pip
"${PIP}" install --no-cache-dir "torch==2.10.0" --extra-index-url https://download.pytorch.org/whl/cu128
"${PIP}" install --no-cache-dir "vllm==0.18.0" "transformers==4.57.6" "openai==2.24.0" "math-verify==0.9.0"

echo "[setup_student] Final installed versions:"
_verify_pkgs
