#!/usr/bin/env bash
# T-031 Stage 1 Pro 5000 CPU-only entry script (Card v1 / Inbox #1).
# Runs purely on CPU (CUDA_VISIBLE_DEVICES="" + nice -n 19), never touches T-026 dirs,
# verifies math-verify==0.8.0 / latex2sympy2_extended==1.10.2 & hybrid_grader.py (7d2a5f46),
# runs unit tests in pro5000-fast, and runs Route C preliminary inventory on T-023 measured JSONL.
set -euo pipefail
umask 077

HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${HERE_DIR}/.." && pwd)"
BASE="${PRO5000_BASE:-${HOME}/bzhang/a_group_pro5000}"
ENV_DIR="${PRO5000_ENV_DIR:-${BASE}/envs/pro5000-fast}"
EDUARDO_REPO="${EDUARDO_REPO:-${BASE}/source/EduardoRL}"
MEASURED_JSONL="${MEASURED_JSONL:-${BASE}/runs/t023_stage2_20261125_163319/routeA_stage2_measured.jsonl}"
RUN_DIR="${1:-${BASE}/runs/t031_stage1_$(date +%Y%m%d_%H%M%S)}"

unset GEMINI_API_KEY HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy CUDA_VISIBLE_DEVICES || true
export CUDA_VISIBLE_DEVICES=""
export TMPDIR="${BASE}/tmp"
mkdir -p "${TMPDIR}" "${RUN_DIR}"
chmod 700 "${RUN_DIR}"

PYTHON_BIN="${ENV_DIR}/bin/python"
[ -x "${PYTHON_BIN}" ] || { echo "ERROR: missing python at ${PYTHON_BIN}" >&2; exit 2; }
[ -d "${EDUARDO_REPO}/eduardo" ] || { echo "ERROR: missing EduardoRL repo at ${EDUARDO_REPO}" >&2; exit 2; }
[ -f "${MEASURED_JSONL}" ] || { echo "ERROR: missing T-023 measured JSONL at ${MEASURED_JSONL}" >&2; exit 2; }

export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${CODE_ROOT}:${EDUARDO_REPO}"

echo "=== [1/3] Package Pins & hybrid_grader.py SHA-256 Check (CPU-only) ==="
nice -n 19 "${PYTHON_BIN}" -B - "${CODE_ROOT}" <<'PY'
import hashlib, importlib.metadata, sys
from pathlib import Path

code_root = Path(sys.argv[1])
pins = {"math-verify": "0.8.0", "latex2sympy2_extended": "1.10.2"}
bad = []
for pkg in ("math-verify", "latex2sympy2_extended", "latex2sympy2"):
    try:
        ver = importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        ver = "NOT_INSTALLED"
    print(f"  {pkg}={ver}")
    if pkg in pins and ver != pins[pkg]:
        bad.append(f"{pkg}={ver}!={pins[pkg]}")
    elif pkg == "latex2sympy2" and ver not in ("NOT_INSTALLED", "1.5.4"):
        bad.append(f"{pkg}={ver}!=1.5.4")

hg = code_root / "routeA_data" / "hybrid_grader.py"
h = hashlib.sha256(hg.read_bytes()).hexdigest()
exp = "7d2a5f46619218e92ed11efc4e7e07f190027ac0f1cca9336905c090ff62709a"
print(f"  hybrid_grader.py sha256={h} ok={h == exp}")
if h != exp:
    bad.append(f"hybrid_grader_sha={h}")
if bad:
    raise SystemExit("PRECHECK_FAILED: " + "; ".join(bad))
PY

echo "=== [2/3] Unit Tests in pro5000-fast (CPU-only, real math-verify==0.8.0) ==="
(cd "${CODE_ROOT}" && CUDA_VISIBLE_DEVICES="" nice -n 19 "${PYTHON_BIN}" -B -m unittest discover -s tests -p "test_routeC*.py" -v)

echo "=== [3/3] T-031 Stage 1 Preliminary Route C Inventory (CPU-only) ==="
CUDA_VISIBLE_DEVICES="" nice -n 19 "${PYTHON_BIN}" -B -m routeC_data.routeC_inventory \
  --measured-jsonl "${MEASURED_JSONL}" \
  --out-dir "${RUN_DIR}"

echo "RUN_DIR=${RUN_DIR}"
