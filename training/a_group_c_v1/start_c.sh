#!/usr/bin/env bash
set -eu
umask 077
pkg=$(cd "$(dirname "$0")" && pwd)
root=/workspace/A_group_repro_20261117
py="$root/envs/env-candidate-vllm0191/bin/python"
work="$root/pilots/c_paper_v1"
log="$root/pilots/c_paper_v1-launch.log"
if [[ -e "$work" || -e "$log" ]]; then
  echo "已有 C 运行记录。以下为只读状态。"
  "$py" -B "$pkg/c_status.py" --work "$work"
  exit 1
fi
mkdir -p "$root/pilots"
"$py" -B "$pkg/../a_group_b_native_v1/stop_old_waiter.py"
# noclobber makes submitting the same run from two terminals safe.
set -o noclobber
exec 3>"$log"
nohup "$py" -B "$pkg/c_stage.py" --run --name c_paper_v1 < /dev/null >&3 2>&1 &
echo "已提交 C 单步，PID=$!。"
echo "查看状态：$py -B $pkg/c_status.py --work $work"
echo "收尾汇总：$work/C阶段报告.md"
