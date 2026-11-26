#!/usr/bin/env bash
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE="${PRO5000_BASE:-$HOME/bzhang/a_group_pro5000}"
ENV_DIR="${PRO5000_ENV_DIR:-$BASE/envs/pro5000-fast}"
PY_BIN="$ENV_DIR/bin/python"
EDUARDO_REPO="$BASE/source/EduardoRL"

unset LD_LIBRARY_PATH HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy CUDA_VISIBLE_DEVICES || true
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export LD_LIBRARY_PATH="$ENV_DIR/lib"
export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$CODE_ROOT:$CODE_ROOT/training/a_group_pro5000:$EDUARDO_REPO"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMPDIR="$BASE/tmp" HF_HOME="$BASE/cache/hf" TRITON_CACHE_DIR="$BASE/cache/triton" \
       TORCHINDUCTOR_CACHE_DIR="$BASE/cache/inductor" VLLM_CACHE_ROOT="$BASE/cache/vllm" XDG_CACHE_HOME="$BASE/cache/xdg"

if [ "${1:-}" = "--status" ]; then
  shift
  exec "$PY_BIN" -B "$SCRIPT_DIR/stage2_routea_pilot.py" --base "$BASE" --status "$@"
fi

RUN_DIR=""
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"; shift 2 ;;
    *)
      EXTRA_ARGS+=("$1"); shift ;;
  esac
done

mkdir -p "$TMPDIR" "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME" "$BASE/runs"

if [ -n "$RUN_DIR" ]; then
  mkdir -p "$RUN_DIR"
  echo "$$" > "$RUN_DIR/stage2.pid"
  exec "$PY_BIN" -B "$SCRIPT_DIR/stage2_routea_pilot.py" \
    --base "$BASE" \
    --run-dir "$RUN_DIR" \
    "${EXTRA_ARGS[@]}"
fi

exec "$PY_BIN" -B "$SCRIPT_DIR/stage2_routea_pilot.py" \
  --base "$BASE" \
  "${EXTRA_ARGS[@]}"
