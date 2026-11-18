#!/usr/bin/env bash
# Offline acceptance of the existing B run in the A_group container.
# Only the two pinned, public audit tools are downloaded.
set -euo pipefail
umask 077
root=/workspace/A_group_repro_20261117
export py="$root/envs/env-candidate-vllm0191/bin/python"
export work="$root/pilots/b_native_b09d2fb"
out="$root/b-accept-b09d2fb-v1"
[[ -x "$py" && -d "$work" ]] || { echo '未找到本次运行或其解释器'; exit 1; }
echo "查看进度：tail -n 24 '$out/验收进度.md'"
if [[ -e "$out" ]]; then
    echo '已有验收目录，先查看进度。'
    exit 0
fi

rev=b0660163a907e06d3b33046fc40db50a3cfbe33b
base="https://gh-proxy.com/https://github.com/DreamZhongJu/eduardo27b-isolated-eval/raw/$rev/training/a_group_b_native_v1"
tmp=$(mktemp -d)
trap 'for f in audit_posthoc.py verify_checkpoints.py; do [[ ! -f "$tmp/$f" ]] || unlink "$tmp/$f"; done; rmdir "$tmp"' EXIT
for f in audit_posthoc.py verify_checkpoints.py; do
    curl -fsSL --retry 2 --connect-timeout 15 --max-time 60 "$base/$f" -o "$tmp/$f"
done
(
    cd "$tmp"
    sha256sum -c <<'SUMS'
2ad8d3ae15ea64cf786dd39f24aafe87d51c94ae26ea2a8bc21efe33cfc2baf6  audit_posthoc.py
665eabcf759f1e094e767b7077307d08c1e6dbc03b5ff57faf45483407284da4  verify_checkpoints.py
SUMS
)
mkdir -m 700 "$out"
cd "$out"
install -m 600 "$tmp"/*.py .
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$out/cache/hf" XDG_CACHE_HOME="$out/cache" GIT_OPTIONAL_LOCKS=0
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
nohup bash -c '
    echo "正在核对 B 的对话、技术失败记录和权重同步指标。"
    "$py" -B audit_posthoc.py --work "$work" --check-terminal-failures
    code=$?
    case "$code" in
        0) ;;
        2) echo "已停止：发现技术终止记录，B 待排查。"; exit 2 ;;
        3) echo "日志覆盖不足，技术失败数待定；继续独立核对检查点。" ;;
        *) echo "审计工具未完成，退出码为 $code。"; exit "$code" ;;
    esac
    echo "开始依次合并两个检查点。"
    "$py" -B verify_checkpoints.py --work "$work" --output "$PWD/checkpoints" || {
        echo "CPU 验收未完成，请查看上方失败阶段。"; exit 1;
    }
    echo "CPU 检查点验收结束；B 等待汇报确认。"
' >> 验收进度.md 2> 工具诊断.log < /dev/null &
echo "CPU 验收已提交后台，PID=$!。"
