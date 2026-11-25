#!/usr/bin/env bash
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE="${PRO5000_BASE:-$HOME/bzhang/a_group_pro5000}"
ENV_DIR="${PRO5000_ENV_DIR:-$BASE/envs/pro5000-fast}"
PY_BIN="$ENV_DIR/bin/python"
EDUARDO_REPO="$BASE/source/EduardoRL"

if [ "${1:-}" = "--status" ]; then
  RUN_DIR="${2:-$(ls -td "$BASE/runs/t024_stage2_"* 2>/dev/null | head -n 1 || true)}"
  [ -n "$RUN_DIR" ] && [ -d "$RUN_DIR" ] || { echo "找不到 t024_stage2 运行目录"; exit 1; }
  echo "RUN_DIR=$RUN_DIR"
  if [ -f "$RUN_DIR/stage2.pid" ]; then
    PID="$(cat "$RUN_DIR/stage2.pid")"
    if kill -0 "$PID" 2>/dev/null; then
      echo "PROCESS=RUNNING (pid=$PID)"
    else
      echo "PROCESS=STOPPED (last_pid=$PID)"
    fi
  fi
  BASE_DONE="$(ls "$RUN_DIR/checkpoints/base/"*.json 2>/dev/null | wc -l || echo 0)"
  AG_DONE="$(ls "$RUN_DIR/checkpoints/a_group/"*.json 2>/dev/null | wc -l || echo 0)"
  echo "PROGRESS: base=${BASE_DONE}/200 a_group=${AG_DONE}/200"
  if [ -f "$RUN_DIR/stage2_summary.json" ]; then
    echo "SUMMARY_READY=$RUN_DIR/stage2_summary.json"
  fi
  tail -n 20 "$RUN_DIR/stage2_bg.log" 2>/dev/null || true
  exit 0
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

if [ -z "$RUN_DIR" ]; then
  RUN_DIR="$BASE/runs/t024_stage2_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$RUN_DIR"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy CUDA_VISIBLE_DEVICES || true
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$CODE_ROOT:$EDUARDO_REPO"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMPDIR="$BASE/tmp" HF_HOME="$BASE/cache/hf" TRITON_CACHE_DIR="$BASE/cache/triton" \
       TORCHINDUCTOR_CACHE_DIR="$BASE/cache/inductor" VLLM_CACHE_ROOT="$BASE/cache/vllm" XDG_CACHE_HOME="$BASE/cache/xdg"
mkdir -p "$TMPDIR" "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME"

echo "$$" > "$RUN_DIR/stage2.pid"
exec "$PY_BIN" -B "$SCRIPT_DIR/stage2_dev_sim.py" \
  --base "$BASE" \
  --run-dir "$RUN_DIR" \
  "${EXTRA_ARGS[@]}"
