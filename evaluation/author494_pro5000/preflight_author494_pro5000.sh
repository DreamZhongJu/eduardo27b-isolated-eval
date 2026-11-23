#!/usr/bin/env bash
# Read-only preflight for Pro 5000 author-494 evaluation (A group step_100 & base).
# Does NOT allocate GPUs and does NOT call Agnes.
set -euo pipefail

BASE="${PRO5000_BASE:-${HOME}/bzhang/a_group_pro5000}"
ENV_DIR="${PRO5000_ENV_DIR:-${BASE}/envs/pro5000-fast}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python}"
REPO="${EDUARDO_REPO:-${BASE}/source/EduardoRL}"
A_GROUP_RUN_DIR="${A_GROUP_RUN_DIR:-${BASE}/runs/stage3_formal_20261121_144311}"
A_GROUP_MODEL="${A_GROUP_MODEL:-${A_GROUP_RUN_DIR}/exports/step_100}"
A_GROUP_SHA_FILE="${A_GROUP_SHA_FILE:-${A_GROUP_RUN_DIR}/export_step100_sha256.txt}"
A_GROUP_VERIFIED_JSON="${A_GROUP_VERIFIED_JSON:-${A_GROUP_RUN_DIR}/export_step100_verified.json}"
BASE_TUTOR_MODEL="${BASE_TUTOR_MODEL:-/data/models/Qwen3.5-4B}"
STUDENT_MODEL="${STUDENT_MODEL:-/dev/shm/zhujie_a_group/Meta-Llama-3.1-8B-Instruct}"
STUDENT_SYMLINK="${STUDENT_SYMLINK:-${BASE}/models/Meta-Llama-3.1-8B-Instruct}"
EVAL_PARQUET="${EVAL_PARQUET:-${BASE}/data/author494_eval/test.parquet}"
T018_MANIFEST="${T018_MANIFEST:-${BASE}/t018_bundle/t018_manifest.json}"
REQUIRE_T018_MANIFEST="${REQUIRE_T018_MANIFEST:-1}"
AGNES_KEY_ENV="${AGNES_KEY_ENV:-AGNES_API_KEY}"
AGNES_KEY_FILE="${AGNES_KEY_FILE:-${HOME}/.secrets/agnes_api_key}"
REQUIRE_EVAL_PARQUET="${REQUIRE_EVAL_PARQUET:-1}"
MIN_IDLE_GPUS="${MIN_IDLE_GPUS:-4}"

unset LD_LIBRARY_PATH HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true
export LD_LIBRARY_PATH="${ENV_DIR}/lib"
export PATH="${ENV_DIR}/bin:${PATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export TMPDIR="${BASE}/tmp"
export HF_HOME="${BASE}/cache/hf"
export TRITON_CACHE_DIR="${BASE}/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${BASE}/cache/inductor"
export VLLM_CACHE_ROOT="${BASE}/cache/vllm"
export XDG_CACHE_HOME="${BASE}/cache/xdg"

mkdir -p "${TMPDIR}" "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${XDG_CACHE_HOME}"

echo "=== 1. Env & Secrets Check (values never printed) ==="
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
df -BG "${BASE}" /data / | awk 'NR==1 || !seen[$1]++'
"${PYTHON_BIN}" -B - <<'PY'
import socket
for port in (18961, 18962, 18861, 18862, 8080):
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
"${PYTHON_BIN}" -B - "${MIN_IDLE_GPUS}" <<'PY'
import subprocess
import sys
import time
from pathlib import Path

min_idle = int(sys.argv[1])

def snap():
    gpu_out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    app_out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    uuid_to_idx = {}
    gpus = {}
    for line in gpu_out.strip().splitlines():
        idx_s, uuid, free_s, total_s, util_s = [x.strip() for x in line.split(",")]
        idx = int(idx_s)
        uuid_to_idx[uuid] = idx
        gpus[idx] = {
            "idx": idx,
            "uuid": uuid,
            "free": int(free_s),
            "total": int(total_s),
            "used": max(0, int(total_s) - int(free_s)),
            "util": int(util_s),
            "apps": [],
        }
    if app_out:
        for line in app_out.splitlines():
            parts = [x.strip() for x in line.split(",")]
            uuid, pid, pname, mem = parts[0], parts[1], parts[2], parts[3]
            idx = uuid_to_idx.get(uuid)
            if idx is not None:
                gpus[idx]["apps"].append((pid, pname, mem))
    return gpus

s1 = snap()
time.sleep(2)
s2 = snap()

apps_any = False
for idx in sorted(s2):
    for pid, pname, mem in s2[idx]["apps"]:
        apps_any = True
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
if not apps_any:
    print("compute_apps: none")

idle = []
for idx in sorted(s2):
    g1, g2 = s1[idx], s2[idx]
    if g1["uuid"] != g2["uuid"]:
        continue
    if g1["apps"] or g2["apps"] or g1["free"] < 70000 or g2["free"] < 70000:
        continue
    if idx == 6:
        if g1["used"] < 500 and g2["used"] < 500:
            idle.append(idx)
    elif g1["util"] < 10 and g2["util"] < 10:
        idle.append(idx)

print(f"idle_gpus_2snap={idle} count={len(idle)} required={min_idle}")
if len(idle) < min_idle:
    raise SystemExit(f"ERROR: need >= {min_idle} idle GPUs (>=70000 MiB free), got {len(idle)}")
PY

echo "=== 4. Upstream Code (e8c89b7), Packages, Models & Dataset ==="
git -C "${REPO}" rev-parse HEAD
git -C "${REPO}" status --porcelain --untracked-files=no

"${PYTHON_BIN}" -B - \
  "${REPO}" \
  "${A_GROUP_MODEL}" \
  "${A_GROUP_SHA_FILE}" \
  "${A_GROUP_VERIFIED_JSON}" \
  "${BASE_TUTOR_MODEL}" \
  "${STUDENT_MODEL}" \
  "${STUDENT_SYMLINK}" \
  "${BASE}" \
  "${EVAL_PARQUET}" \
  "${REQUIRE_EVAL_PARQUET}" \
  "${T018_MANIFEST}" \
  "${REQUIRE_T018_MANIFEST}" \
  "${AGNES_KEY_ENV}" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

(
    repo_s,
    agroup_dir_s,
    agroup_sha_s,
    agroup_ver_s,
    base_tutor_s,
    student_shm_s,
    student_sym_s,
    base_s,
    eval_parquet_s,
    req_eval_pq_s,
    t018_manifest_s,
    req_t018_s,
    agnes_key_env,
) = sys.argv[1:14]

repo = Path(repo_s)
agroup_dir = Path(agroup_dir_s)
agroup_sha_file = Path(agroup_sha_s)
agroup_ver_file = Path(agroup_ver_s)
base_tutor_dir = Path(base_tutor_s)
student_shm_dir = Path(student_shm_s)
student_sym_dir = Path(student_sym_s)
base_dir = Path(base_s)
eval_parquet = Path(eval_parquet_s)
require_eval_parquet = req_eval_pq_s == "1"
t018_manifest_path = Path(t018_manifest_s)
require_t018_manifest = req_t018_s == "1"

hard_errors = []

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

head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
if head != "e8c89b7293e65e632149571b16a1c2d4e33c77fe":
    hard_errors.append(f"UPSTREAM_HEAD_MISMATCH:{head}")
if status:
    hard_errors.append("UPSTREAM_DIRTY")

EXPECTED_UPSTREAM = {
    "eval/__init__.py": "adb9b78aee3d45ed47de9d756dd391def5bbe73fab1689656f9a36477fb09594",
    "eval/run_env_test_eval.py": "f00a9730ef4f0e61f4eae7b5d8d1048beb5740068ad94a4071ca97b5f032479d",
    "eval/run_middleturn_eval.py": "8b095899f81e0872a36b77c4424e5107927645002727277ccce7db8cdd16ccbb",
    "eduardo/__init__.py": "c0061335e07747433ded663f72cb64768b6f4be240808e3b049ad256ae85708a",
    "eduardo/create_transfer_dataset.py": "fd801436c412166230f7ff5d4442276336e1afeac8e5fede8009826298a15a5f",
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
if code_bad:
    hard_errors.append(f"UPSTREAM_MISMATCH:{code_bad}")

print("=== 5. Package Versions ===")
import torch, transformers, vllm, datasets, pandas, openai, numpy
print(
    f"python={sys.version.split()[0]} torch={torch.__version__} transformers={transformers.__version__} "
    f"vllm={vllm.__version__} numpy={numpy.__version__} datasets={datasets.__version__} "
    f"pandas={pandas.__version__} openai={openai.__version__}"
)
for pkg in ("math-verify", "pyarrow", "flash-attn", "flash-linear-attention", "fla-core", "causal-conv1d"):
    try:
        print(f"{pkg}={importlib.metadata.version(pkg)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{pkg}=NOT_INSTALLED")

print("=== 6. A-Group step_100 Export Check (T-017 收尾第 2 步锁定) ===")
STEP100_LOCK = {
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    "config.json": "6d320dedc3a50ea3eb4bf7ab503fed018f21784f001c61ff9de9a7ed411e69f0",
    "generation_config.json": "944b9abc4ec388933cc00c757a7b794683929e5a02e5123853892a750645689",
    "model.safetensors": "35435951b689199681a59cc9667cda29e75025f575cd8758bb8320e2f11d88e8",
    "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "7488550a894dcf741616126e322ea31d66cf9457b59ec5825f4c7efdced421f8",
}
if not agroup_dir.is_dir():
    print(f"A_GROUP_STEP100_DIR_MISSING: {agroup_dir}")
    hard_errors.append("A_GROUP_STEP100_DIR_MISSING")
else:
    if not agroup_ver_file.is_file() or not agroup_sha_file.is_file():
        print("A_GROUP_STEP100_MANIFEST_MISSING")
        hard_errors.append("A_GROUP_STEP100_MANIFEST_MISSING")
    else:
        ver = json.loads(agroup_ver_file.read_text(encoding="utf-8"))
        if ver.get("passed") is not True or ver.get("files_sha256") != STEP100_LOCK:
            print("A_GROUP_STEP100_RECEIPT_MISMATCH")
            hard_errors.append("A_GROUP_STEP100_RECEIPT_MISMATCH")
    actual_files = {}
    agroup_bad = 0
    for p in sorted(agroup_dir.rglob("*")):
        if p.is_symlink():
            print(f"a_group/{p.relative_to(agroup_dir)}: SYMLINK_FORBIDDEN")
            agroup_bad += 1
        elif p.is_file():
            rel = str(p.relative_to(agroup_dir))
            actual_files[rel] = sha256_file(p)
    for rel in sorted(set(STEP100_LOCK) | set(actual_files)):
        exp_h = STEP100_LOCK.get(rel)
        got_h = actual_files.get(rel)
        if exp_h is None:
            print(f"a_group/{rel}: EXTRA_FILE sha256={got_h}")
            agroup_bad += 1
        elif got_h is None:
            print(f"a_group/{rel}: MISSING")
            agroup_bad += 1
        elif got_h != exp_h:
            print(f"a_group/{rel}: MISMATCH got={got_h} exp={exp_h}")
            agroup_bad += 1
        else:
            print(f"a_group/{rel}: OK sha256={got_h}")
    print(f"a_group_step100_checked={len(STEP100_LOCK)} mismatches={agroup_bad}")
    if agroup_bad:
        hard_errors.append(f"A_GROUP_STEP100_MISMATCH:{agroup_bad}")

print("=== 7. Base Tutor Qwen3.5-4B (HF 851bf6e8) SHA256 Check ===")
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
    p = base_tutor_dir / fname
    if not p.is_file():
        print(f"base_tutor/{fname}: MISSING")
        tutor_bad += 1
        continue
    s256 = sha256_file(p)
    got = s256 if algo == "sha256" else git_sha1_file(p)
    status_s = "OK" if got == exp else "MISMATCH"
    if status_s != "OK":
        tutor_bad += 1
    print(f"base_tutor/{fname}: {status_s} sha256={s256}")
print(f"base_tutor_checked={len(TUTOR_LOCK)} mismatches={tutor_bad}")
if tutor_bad:
    hard_errors.append(f"BASE_TUTOR_MISMATCH:{tutor_bad}")

print("=== 8. Student Llama-3.1-8B-Instruct (/dev/shm/zhujie_a_group, HF 0e9e39f2) Check ===")
student_dir = None
for cand in (student_shm_dir, student_sym_dir):
    if cand.exists() and (cand / "config.json").is_file():
        student_dir = cand.resolve(strict=True)
        break
if student_dir is None:
    shm_root = Path("/dev/shm/zhujie_a_group")
    if shm_root.is_dir():
        for p in sorted(shm_root.rglob("config.json")):
            if "llama" in str(p).lower():
                student_dir = p.parent.resolve(strict=True)
                break

STUDENT_LOCK = {
    "config.json": "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e",
    "tokenizer.json": "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
    "tokenizer_config.json": "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424",
    "model-00001-of-00004.safetensors": "2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668",
    "model-00002-of-00004.safetensors": "09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15",
    "model-00003-of-00004.safetensors": "fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa",
    "model-00004-of-00004.safetensors": "92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b",
}
if student_dir is None:
    print("student_dir: MISSING")
    hard_errors.append("STUDENT_MISSING")
else:
    print(f"student_resolved_dir={student_dir}")
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
            status_s = "OK" if got == exp else "MISMATCH"
            if status_s != "OK":
                student_bad += 1
            print(f"student/{name}: {status_s} size={p.stat().st_size} sha256={got}")
    print(f"student_checked={len(STUDENT_LOCK)} mismatches={student_bad}")
    if student_bad:
        hard_errors.append(f"STUDENT_MISMATCH:{student_bad}")

print("=== 9. Dataset & A100 T-018 Table Check ===")
RAW_TEST_SHA = "109b45cf350cb70414a294ff999d4f545b06e9df91e02ec1a0a3b30783c46ec0"
RAW_TRAIN_SHA = "5171c9891b502f9792b14ac2b15684ce85b8ceb2593689ed3e88b9f349b25d61"
found_raw_test = None
found_raw_train = None
data_root = base_dir / "data"
if data_root.is_dir():
    for p in sorted(data_root.rglob("*.parquet")):
        if not p.is_file():
            continue
        if p.name.startswith("test-00000") and sha256_file(p) == RAW_TEST_SHA:
            found_raw_test = p
        elif p.name.startswith("train-00000") and sha256_file(p) == RAW_TRAIN_SHA:
            found_raw_train = p
print(f"raw_test_parquet={'OK ' + str(found_raw_test) if found_raw_test else 'NOT_FOUND'}")
print(f"raw_train_parquet={'OK ' + str(found_raw_train) if found_raw_train else 'NOT_FOUND'}")

eval_pq_sha = None
if eval_parquet.is_file():
    eval_pq_sha = sha256_file(eval_parquet)
    df = pandas.read_parquet(eval_parquet)
    n_tp = sum(1 for r in df["extra_info"] if isinstance(r, dict) and r.get("transfer_problem") and r.get("transfer_answer"))
    print(f"eval_test_parquet={eval_parquet} rows={len(df)} transfer_valid={n_tp} sha256={eval_pq_sha}")
    if len(df) != 494 or n_tp != 494:
        hard_errors.append(f"EVAL_PARQUET_INVALID:rows={len(df)},valid={n_tp}")
else:
    print(f"eval_test_parquet={eval_parquet} MISSING")
    if require_eval_parquet:
        hard_errors.append("EVAL_PARQUET_MISSING")

if t018_manifest_path.is_file():
    t_man = json.loads(t018_manifest_path.read_text(encoding="utf-8"))
    t_csv = t018_manifest_path.parent / str(t_man.get("table_filename", "t018_per_problem.csv"))
    t_csv_sha = sha256_file(t_csv) if t_csv.is_file() else "MISSING"
    print(f"t018_table={t_csv} sha256_match={t_csv_sha == t_man.get('table_sha256')}")
    if t_csv_sha != t_man.get("table_sha256"):
        hard_errors.append("T018_TABLE_SHA_MISMATCH")
    exp_pq_sha = t_man.get("test_parquet_sha256")
    pq_match = (eval_pq_sha is not None and eval_pq_sha == exp_pq_sha)
    print(f"test_parquet_matches_a100_t018={pq_match} (pro5000={eval_pq_sha} a100={exp_pq_sha})")
    if eval_pq_sha is not None and not pq_match:
        hard_errors.append("EVAL_PARQUET_SHA_DIFFERS_FROM_A100_T018")
else:
    print(f"t018_manifest={t018_manifest_path} MISSING")
    if require_t018_manifest:
        hard_errors.append("T018_MANIFEST_MISSING")

data_free_gb = shutil.disk_usage("/data").free / 1e9
print(f"data_free_gb={data_free_gb:.2f}")
if data_free_gb < 20.0:
    hard_errors.append(f"DATA_DISK_LOW:{data_free_gb:.2f}GB")

if not os.environ.get(agnes_key_env):
    hard_errors.append(f"AGNES_KEY_NOT_SET:{agnes_key_env}")
if os.environ.get("GEMINI_API_KEY"):
    hard_errors.append("GEMINI_API_KEY_SET")

if hard_errors:
    print("PREFLIGHT_FAILED:", ",".join(hard_errors))
    sys.exit(3)
print("PRO5000_PREFLIGHT_OK")
PY
