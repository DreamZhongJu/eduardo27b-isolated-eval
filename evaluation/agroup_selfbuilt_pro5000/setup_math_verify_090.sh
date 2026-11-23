#!/usr/bin/env bash
# Build tiny CPU-only math-verify==0.9.0 environment under $BASE (on /data) for offline Author-Rule.
# Never modifies pro5000-fast and keeps all caches/tmp under $BASE (never touches /).
set -euo pipefail

BASE="${PRO5000_BASE:-${HOME}/bzhang/a_group_pro5000}"
TUTOR_ENV_DIR="${PRO5000_ENV_DIR:-${BASE}/envs/pro5000-fast}"
MV_ENV_DIR="${MATH_VERIFY_090_ENV_DIR:-${BASE}/envs/math_verify_090}"
TMP_DIR="${BASE}/tmp"
PIP_CACHE="${BASE}/cache/pip"
MIRROR_INDEX="${PIP_MIRROR_INDEX:-https://mirrors.aliyun.com/pypi/simple}"

mkdir -p "${TMP_DIR}" "${PIP_CACHE}" "$(dirname "${MV_ENV_DIR}")"
export TMPDIR="${TMP_DIR}"
export PIP_CACHE_DIR="${PIP_CACHE}"

_verify_mv() {
  local py="${MV_ENV_DIR}/bin/python"
  [ -x "${py}" ] || return 1
  "${py}" -B - <<'PY'
import importlib.metadata as md
from math_verify import parse, verify

ver = md.version("math-verify")
if not ver.startswith("0.9.0"):
    raise SystemExit(f"ERROR: expected math-verify 0.9.0, got {ver}")
assert verify(parse("\\boxed{2}"), parse("\\boxed{1+1}"))
print(f"author_rule_env/math-verify={ver} verify_smoke=OK")
PY
}

if [ -x "${MV_ENV_DIR}/bin/python" ] && _verify_mv; then
  echo "[setup_mv090] Existing ${MV_ENV_DIR} verified OK."
  exit 0
fi

rm -rf "${MV_ENV_DIR}"
if [ -x "${TUTOR_ENV_DIR}/bin/python" ]; then
  "${TUTOR_ENV_DIR}/bin/python" -m venv "${MV_ENV_DIR}"
else
  python3 -m venv "${MV_ENV_DIR}"
fi

"${MV_ENV_DIR}/bin/pip" install -i "${MIRROR_INDEX}" --no-cache-dir --upgrade pip
"${MV_ENV_DIR}/bin/pip" install -i "${MIRROR_INDEX}" --no-cache-dir "math-verify==0.9.0"
_verify_mv
echo "PRO5000_MV090_ENV_OK env=${MV_ENV_DIR}"
