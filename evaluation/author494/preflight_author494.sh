#!/usr/bin/env bash
# Read-only preflight for A100 author-494 base evaluation.
# Does NOT allocate GPUs and does NOT call Agnes.
set -euo pipefail

ROOT="${A_GROUP_ROOT:-/workspace/A_group_repro_20261117}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/envs/env-candidate-vllm0191/bin/python}"
REPO="${EDUARDO_REPO:-${ROOT}/source/EduardoRL}"
TUTOR_MODEL="${TUTOR_MODEL:-${ROOT}/models/Qwen3.5-4B}"
STUDENT_MODEL="${STUDENT_MODEL:-/workspace/models/Meta-Llama-3.1-8B-Instruct}"
OFFICIAL_DATA="${OFFICIAL_DATA:-${ROOT}/data/official}"
AGNES_KEY_ENV="${AGNES_KEY_ENV:-AGNES_API_KEY}"
AGNES_KEY_FILE="${AGNES_KEY_FILE:-/workspace/.secrets/agnes_api_key}"

echo "=== 1. Env & Secrets Check (values never printed) ==="
for pvar in HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; do
  if [ -n "${!pvar:-}" ]; then
    echo "PROXY_WARNING: ${pvar}=已设"
  fi
done
if [ -n "${!AGNES_KEY_ENV:-}" ]; then
  echo "AGNES_KEY_ENV(${AGNES_KEY_ENV}): 已设"
else
  echo "AGNES_KEY_ENV(${AGNES_KEY_ENV}): 未设"
fi
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
  echo "CUDA_VISIBLE_DEVICES: 已设 (警告: 建议 unset 后再选物理卡)"
else
  echo "CUDA_VISIBLE_DEVICES: 未设"
fi

echo "=== 2. Disk & Ports ==="
df -BG "${ROOT}" /workspace /tmp | awk 'NR==1 || !seen[$1]++'
"${PYTHON_BIN}" -B - <<'PY'
import socket
for port in (18861, 18862, 18961, 18962, 8080):
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
"${PYTHON_BIN}" -B - <<'PY'
import subprocess
from pathlib import Path

gpu_out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
    text=True,
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
                # Only show executable basename + script basename (never values/prompts)
                cmd_head = " ".join(Path(x).name for x in args[:2] if not x.startswith("-"))
        except OSError:
            pass
        print(f"GPU {idx}: pid={pid} uid={uid} mem={mem}MiB proc={cmd_head}")
PY

echo "=== 4. Upstream Code (e8c89b7) & Package Versions ==="
git -C "${REPO}" rev-parse HEAD
git -C "${REPO}" status --porcelain --untracked-files=no
CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" -B - "${REPO}" "${TUTOR_MODEL}" "${STUDENT_MODEL}" "${OFFICIAL_DATA}" "${ROOT}" <<'PY'
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

repo, tutor_dir, student_dir, official_dir, root_dir = [Path(x) for x in sys.argv[1:6]]

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

EXPECTED_UPSTREAM = {
    "eval/__init__.py": "adb9b78aee3d45ed47de9d756dd391def5bbe73fab1689656f9a36477fb09594",
    "eval/run_env_test_eval.py": "f00a9730ef4f0e61f4eae7b5d8d1048beb5740068ad94a4071ca97b5f032479d",
    "eval/run_middleturn_eval.py": "8b095899f81e0872a36b77c4424e5107927645002727277ccce7db8cdd16ccbb",
    "eduardo/__init__.py": "c0061335e07747433ded663f72cb64768b6f4be240808e3b049ad256ae85708a",
    "eduardo/utils.py": "b80d8d512561d2de3bb0bce6e82a10e00916c0a1dc3f6d235b918e8c992e0eb4",
    "eduardo/process_dataset.py": "7624e717860f01a638d47abc81e6cec40369966213bba082bef4542feb196b1f",
    "eduardo/reward_function.py": "4b34fa6c90b9347d4fcc757b961ff7d8e0818ec21c8af3762ccfe133f343c69f",
    "eduardo/metrics/__init__.py": "ae61dcb15c0521d9aaef0a5dc19b688ff50b08555fd86fe76c7700cfd8211abc",
    "eduardo/metrics/correctness.py": "2d82121de9b316a941a8cc25fbfacc835efc38669777aaaaf3793f1a79dd55d9",
    "eduardo/metrics/judge_eval.py": "9f39cb38747c2a2140fcc62688cb77bf6af88636fcaacf81e49652c3c62fcaa5",
    "eduardo/metrics/judge_quality.py": "0f9cb191d367133489cda10c8b16013975cde1c54929dfa74bf06f6961d93212",
    "eduardo/metrics/length_penalty.py": "485f125b1336b90df8aa4c52511cf7073ae18968991299da8d17b30e477090f3",
    "eduardo/configs/prompts/judge_eval_correctness.txt": "f7be64dd6e782ecb50ffbd5ba7bb90e5ebed3158996368022d1cb6f965cefc0c",
    "eduardo/configs/prompts/judge_eval_pedagogy.txt": "00c800977218e6d6be4d6980c3c40360b59ef611bae22e44430f60b80a244f5d",
    "eduardo/configs/prompts/judge_leakage.txt": "a0338dd1ebe19bb30aa96b5cf401ea5e200a63a483bebb6b161dfc82e66630c2",
    "eduardo/configs/prompts/judge_quality.txt": "383740921ed0db6a53c3bd82314057447c55124790b49e9b2b071900cca3ffdf",
    "eduardo/configs/prompts/student_final_prompt.txt": "841a948b4af39fce42ab6bf3d35ba99d7fc7199f130b2f170f7c1035adcf4088",
    "eduardo/configs/prompts/student_final_transfer_prompt.txt": "98b959b06ebcfa5225028d08cce3a415a98a608975434d84daf4236f2ae489f6",
    "eduardo/configs/prompts/student_prompt.txt": "1d1f2d2688fb42b89559667f39a03c9110b043f3cf3f551182b6cb18a1671167",
    "eduardo/configs/prompts/teacher_prompt.txt": "d1fdc593889ebbe1ff5c2c89201919b4ca0b71f389a50361043f9c7f464ea1c4",
    "eduardo/scripts/launch_student_server.sh": "3e6e133306ac1838b7258a5303e33a6b1a0d756e6f98ce83efa705f57cef49b9",
}
code_bad = 0
for rel, exp in EXPECTED_UPSTREAM.items():
    p = repo / rel
    if not p.is_file():
        print(f"CODE_MISSING: {rel}")
        code_bad += 1
        continue
    got = sha256_file(p)
    if got != exp:
        print(f"CODE_DIFF: {rel} got={got} exp={exp}")
        code_bad += 1
print(f"upstream_files_checked={len(EXPECTED_UPSTREAM)} mismatches={code_bad}")

print("=== 5. Package Versions ===")
import torch, transformers, vllm, datasets, pandas, openai
print(f"python={sys.version.split()[0]} torch={torch.__version__} transformers={transformers.__version__} "
      f"vllm={vllm.__version__} datasets={datasets.__version__} pandas={pandas.__version__} openai={openai.__version__}")
for pkg in ("math-verify", "pyarrow"):
    try:
        print(f"{pkg}={importlib.metadata.version(pkg)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{pkg}=NOT_INSTALLED")

print("=== 6. Tutor Qwen3.5-4B SHA256 Check ===")
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
tutor_bad = 0
for fname, (algo, exp) in TUTOR_LOCK.items():
    p = tutor_dir / fname
    if not p.is_file():
        print(f"TUTOR_MISSING: {fname}")
        tutor_bad += 1
        continue
    s256 = sha256_file(p)
    got = s256 if algo == "sha256" else git_sha1_file(p)
    status = "OK" if got == exp else "MISMATCH"
    if status != "OK":
        tutor_bad += 1
    print(f"tutor/{fname}: {status} sha256={s256}")
print(f"tutor_checked={len(TUTOR_LOCK)} mismatches={tutor_bad}")

print("=== 7. Student Llama-3.1-8B-Instruct Check (HF 0e9e39f2) ===")
STUDENT_LOCK = {
    "config.json": "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e",
    "tokenizer.json": "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
    "tokenizer_config.json": "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424",
    "model-00001-of-00004.safetensors": "2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668",
    "model-00002-of-00004.safetensors": "09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15",
    "model-00003-of-00004.safetensors": "fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa",
    "model-00004-of-00004.safetensors": "92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b",
}
cfg = json.loads((student_dir / "config.json").read_text())
print(f"student_config: model_type={cfg.get('model_type')} hidden={cfg.get('hidden_size')} layers={cfg.get('num_hidden_layers')}")
idx = json.loads((student_dir / "model.safetensors.index.json").read_text())
shards = sorted(set(idx.get("weight_map", {}).values()))
student_bad = 0
for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"] + shards:
    p = student_dir / name
    if not p.is_file():
        print(f"student/{name}: MISSING")
        student_bad += 1
        continue
    got = sha256_file(p)
    exp = STUDENT_LOCK.get(name)
    if exp is None:
        print(f"student/{name}: PRESENT size={p.stat().st_size} sha256={got}")
    else:
        status = "OK" if got == exp else "MISMATCH"
        if status != "OK":
            student_bad += 1
        print(f"student/{name}: {status} size={p.stat().st_size} sha256={got}")
print(f"student_checked={len(STUDENT_LOCK)} mismatches={student_bad}")

print("=== 8. Dataset Check ===")
for rel, exp in (
    ("data/train-00000-of-00001.parquet", "5171c9891b502f9792b14ac2b15684ce85b8ceb2593689ed3e88b9f349b25d61"),
    ("data/test-00000-of-00001.parquet", "109b45cf350cb70414a294ff999d4f545b06e9df91e02ec1a0a3b30783c46ec0"),
):
    p = official_dir / rel
    if not p.is_file():
        print(f"official/{rel}: MISSING")
    else:
        got = sha256_file(p)
        print(f"official/{rel}: {'OK' if got == exp else 'MISMATCH'} sha256={got}")

for cand in (
    root_dir / "data/author_prepared/test.parquet",
    root_dir / "data/author494_eval/test.parquet",
):
    if cand.is_file():
        df = pandas.read_parquet(cand)
        n_tp = sum(1 for r in df["extra_info"] if isinstance(r, dict) and r.get("transfer_problem") and r.get("transfer_answer"))
        print(f"existing_test_parquet={cand} rows={len(df)} transfer_valid={n_tp} sha256={sha256_file(cand)}")
PY
