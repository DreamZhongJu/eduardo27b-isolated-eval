#!/usr/bin/env bash
set -euo pipefail

BASE="${PRO5000_BASE:-$HOME/bzhang/a_group_pro5000}"
PY="$BASE/envs/pro5000-fast/bin/python"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$BASE/runs" "$BASE/tmp"
export PATH="$BASE/envs/pro5000-fast/bin:$PATH"
export LD_LIBRARY_PATH="$BASE/envs/pro5000-fast/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TMPDIR="$BASE/tmp"
export PYTHONDONTWRITEBYTECODE=1

echo "[1/3] 单元测试..."
(cd "$DIR" && "$PY" -B -m unittest -v test_stage1_collect.py test_compare_pro5000.py)

echo "[2/3] 采集环境/盘点/冒烟/27B KV 并发..."
"$PY" -B "$DIR/stage1_collect.py" --base "$BASE"

echo "[3/3] 启动单卡快慢对照与耐久 (slow 40m + fast 20m)..."
"$PY" -B "$DIR/compare_pro5000.py" --base "$BASE" 2>&1 | tee "$BASE/runs/stage1_compare.log"
