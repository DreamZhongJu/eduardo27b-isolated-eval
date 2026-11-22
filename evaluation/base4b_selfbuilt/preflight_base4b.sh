#!/usr/bin/env bash
# Read-only preflight for A100 Qwen3.5-4B base x self-built 3-tier evaluation.
# Does NOT allocate GPUs, does NOT modify 27B records, and does NOT call Agnes.
set -euo pipefail

HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE_DIR}/../.." && pwd)"

A_GROUP_ROOT="${A_GROUP_ROOT:-/workspace/A_group_repro_20261117}"
TUTOR_PY="${TUTOR_PY:-${A_GROUP_ROOT}/envs/env-candidate-vllm0191/bin/python}"
STUDENT_PY="${STUDENT_PY:-/workspace/eduardo-vllm018/bin/python}"
TUTOR_MODEL="${TUTOR_MODEL:-${A_GROUP_ROOT}/models/Qwen3.5-4B}"
STUDENT_MODEL="${STUDENT_MODEL:-/workspace/models/Meta-Llama-3.1-8B-Instruct}"
DATA_FILE="${DATA_FILE:-/workspace/data/math-transfer-test-499-49d3420/data/test.jsonl}"
REF_ROOT="${REF_ROOT:-/workspace/results/own-test-499-v1}"
OVERLAY_ROOT="${OVERLAY_ROOT:-/workspace/results/agnes-topbool-recovery-v1}"
DECISIONS_ROOT="${DECISIONS_ROOT:-/workspace/results/agnes-rejudge-v1}"
REF_REPO="${REF_REPO:-/workspace/eduardo27b-isolated-eval}"
AGNES_KEY_FILE="${AGNES_KEY_FILE:-/workspace/.secrets/agnes_api_key}"

echo "=== 1. Env & Secrets Check (values never printed) ==="
for pvar in HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; do
  if [ -n "${!pvar:-}" ]; then
    echo "PROXY_WARNING: ${pvar}=已设"
  fi
done
if [ -s "${AGNES_KEY_FILE}" ]; then
  echo "AGNES_KEY_FILE(${AGNES_KEY_FILE}): 存在且非空"
else
  echo "AGNES_KEY_FILE(${AGNES_KEY_FILE}): 不存在或为空"
fi
if [ -n "${GEMINI_API_KEY:-}" ]; then
  echo "GEMINI_API_KEY: 已设 (警告: 本任务不应使用 Gemini)"
else
  echo "GEMINI_API_KEY: 未设"
fi
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "CUDA_VISIBLE_DEVICES: 已设 (建议 unset 后再选物理卡)"
else
  echo "CUDA_VISIBLE_DEVICES: 未设"
fi

echo "=== 2. Disk & Ports ==="
df -BG /workspace /tmp | awk 'NR==1 || !seen[$1]++'
"${TUTOR_PY}" -B - <<'PY'
import socket
for port in (18701, 18702, 18861, 18961):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
        state = "FREE"
    except OSError:
        state = "BUSY"
    finally:
        s.close()
    print(f"port 127.0.0.1:{port} -> {state}")
PY

echo "=== 3. GPUs & Per-GPU Processes ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
"${TUTOR_PY}" -B - <<'PY'
import subprocess
from pathlib import Path

gpu_out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True
)
uuid_to_idx = {}
for line in gpu_out.strip().splitlines():
    idx, uuid = [x.strip() for x in line.split(",")]
    uuid_to_idx[uuid] = idx

app_out = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
    text=True,
).strip()
if not app_out:
    print("compute_apps: none")
else:
    for line in app_out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        uuid, pid, pname, mem = parts[0], parts[1], parts[2], parts[3]
        idx = uuid_to_idx.get(uuid, "?")
        uid = "?"
        cmd_head = pname
        try:
            st = Path(f"/proc/{pid}/status").read_text(errors="replace")
            for sline in st.splitlines():
                if sline.startswith("Uid:"):
                    uid = sline.split()[1]
                    break
            raw = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            args = [a.decode("utf-8", "replace") for a in raw if a]
            if args:
                cmd_head = " ".join(Path(x).name for x in args[:2] if not x.startswith("-"))
        except OSError:
            pass
        print(f"GPU {idx}: pid={pid} uid={uid} mem={mem}MiB proc={cmd_head}")
PY

echo "=== 4. 27B Reference Code Binding & Python Environments ==="
if [ -d "${REF_REPO}" ]; then
  echo "ref_repo_head=$(git -C "${REF_REPO}" rev-parse HEAD 2>/dev/null || echo 'UNAVAILABLE')"
else
  echo "ref_repo_head=MISSING(${REF_REPO})"
fi
if [ -f "${REF_ROOT}/CODE_BINDING.json" ]; then
  "${TUTOR_PY}" -B - "${REF_ROOT}/CODE_BINDING.json" "${REPO_ROOT}" <<'PY'
import hashlib, json, sys
from pathlib import Path
binding = json.loads(Path(sys.argv[1]).read_bytes())
repo = Path(sys.argv[2])
print(f"ref_code_binding_commit={binding.get('git_commit')}")
files = binding.get("files", {})
mismatches = 0
for rel, exp in sorted(files.items()):
    if rel.startswith(("app/", "official/")):
        p = repo / "evaluation" / rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != exp:
            print(f"APP_OFFICIAL_DIFF: {rel}")
            mismatches += 1
print(f"ref_app_official_checked_against_kit mismatches={mismatches}")
PY
else
  echo "CODE_BINDING.json: MISSING at ${REF_ROOT}/CODE_BINDING.json"
fi

echo "--- Student env (${STUDENT_PY}) ---"
if [ -x "${STUDENT_PY}" ]; then
  CUDA_VISIBLE_DEVICES="" "${STUDENT_PY}" -B - "${TUTOR_MODEL}" <<'PY'
import importlib.metadata as m, sys
for pkg in ("torch", "vllm", "transformers", "openai", "math-verify"):
    try:
        print(f"student_env/{pkg}={m.version(pkg)}")
    except m.PackageNotFoundError:
        print(f"student_env/{pkg}=MISSING")
try:
    from transformers import AutoConfig
    AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True)
    print("student_env_load_qwen35_config=OK")
except Exception as exc:
    print(f"student_env_load_qwen35_config=FAIL({type(exc).__name__})")
PY
else
  echo "STUDENT_PY MISSING: ${STUDENT_PY}"
fi

echo "--- Tutor env (${TUTOR_PY}) ---"
if [ -x "${TUTOR_PY}" ]; then
  CUDA_VISIBLE_DEVICES="" "${TUTOR_PY}" -B - <<'PY'
import importlib.metadata as m
for pkg in ("torch", "vllm", "transformers", "openai", "math-verify"):
    try:
        print(f"tutor_env/{pkg}={m.version(pkg)}")
    except m.PackageNotFoundError:
        print(f"tutor_env/{pkg}=MISSING")
PY
else
  echo "TUTOR_PY MISSING: ${TUTOR_PY}"
fi

echo "=== 5. Tutor Qwen3.5-4B, Student Llama-3.1-8B, Data & 27B Manifest Check ==="
CUDA_VISIBLE_DEVICES="" "${TUTOR_PY}" -B - "${TUTOR_MODEL}" "${STUDENT_MODEL}" "${DATA_FILE}" <<'PY'
import hashlib, sys
from pathlib import Path
tutor_dir, student_dir, data_file = [Path(x) for x in sys.argv[1:4]]

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def git_sha1_file(p: Path) -> str:
    data = p.read_bytes()
    h = hashlib.sha1()
    h.update(f"blob {len(data)}\0".encode())
    h.update(data)
    return h.hexdigest()

TUTOR_LOCK = {
    "chat_template.jinja": ("git_sha1", "a585dec894e63da457d9440ec6aa7caa16d20860"),
    "config.json": ("git_sha1", "557d961b205319c6a7da5f757f565b69b3967b7d"),
    "merges.txt": ("git_sha1", "a494e019ca1502219fd0128658b979e5f05ae8e8"),
    "model.safetensors-00001-of-00002.safetensors": ("sha256", "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61"),
    "model.safetensors-00002-of-00002.safetensors": ("sha256", "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188"),
    "model.safetensors.index.json": ("git_sha1", "fddda6039f7c1d17260c9e923b8a72fd025d9a86"),
    "preprocessor_config.json": ("git_sha1", "2ea84a437d448ff71b08df68fdd949d5cc4ebb64"),
    "tokenizer.json": ("sha256", "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"),
    "tokenizer_config.json": ("git_sha1", "eda48d3e75a8e59a8479ee4ec8b37f76e711d9c1"),
    "video_preprocessor_config.json": ("git_sha1", "3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe"),
    "vocab.json": ("git_sha1", "0aa0ce0658d60ac4a5d609f4eadb0e8e43514176"),
}
t_bad = 0
for fname, (algo, exp) in TUTOR_LOCK.items():
    p = tutor_dir / fname
    if not p.is_file():
        print(f"TUTOR_MISSING: {fname}")
        t_bad += 1
        continue
    got = sha256_file(p) if algo == "sha256" else git_sha1_file(p)
    if got != exp:
        print(f"TUTOR_MISMATCH: {fname}")
        t_bad += 1
print(f"tutor_qwen35_4b_checked={len(TUTOR_LOCK)} mismatches={t_bad}")

STUDENT_LOCK = {
    "config.json": "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e",
    "tokenizer.json": "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
    "tokenizer_config.json": "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424",
    "model-00001-of-00004.safetensors": "2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668",
    "model-00002-of-00004.safetensors": "09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15",
    "model-00003-of-00004.safetensors": "fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa",
    "model-00004-of-00004.safetensors": "92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b",
}
s_bad = 0
for fname, exp in STUDENT_LOCK.items():
    p = student_dir / fname
    if not p.is_file() or sha256_file(p) != exp:
        print(f"STUDENT_MISMATCH: {fname}")
        s_bad += 1
print(f"student_llama31_8b_checked={len(STUDENT_LOCK)} mismatches={s_bad}")

exp_data = "5e5aa211549d221d2287529e5a4bac03006ba37d0dc7041c19b386f0eb4a5c01"
got_data = sha256_file(data_file) if data_file.is_file() else "MISSING"
print(f"data_test_jsonl={'OK' if got_data == exp_data else 'MISMATCH'} sha256={got_data}")
PY

CUDA_VISIBLE_DEVICES="" "${TUTOR_PY}" -B "${HERE_DIR}/run_post_eval.py" \
  --repo "${REPO_ROOT}" \
  --data "${DATA_FILE}" \
  --ref-root "${REF_ROOT}" \
  --overlay "${OVERLAY_ROOT}" \
  --decisions "${DECISIONS_ROOT}" \
  --check-manifest
