#!/usr/bin/env python3
"""Shared constants, privacy-safe file/GPU/process utilities for T-033."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

# Frozen SHA-256 checksums (never modify these modules).
AGNES_JUDGE_SHA256 = "2ffe746444939555b85e3557d2ddc3b61e3cd59d7997263909668876b924c871"
HYBRID_GRADER_SHA256 = "7d2a5f46619218e92ed11efc4e7e07f190027ac0f1cca9336905c090ff62709a"
PAIRED_GAIN_SHA256 = "b71e4fe9bf13f1e731ff3d2c8a591cab832d9090c30fe47eac76100b2b8245ad"
AUTHOR_COMMIT = "e8c89b7293e65e632149571b16a1c2d4e33c77fe"

# Verified dataset SHA-256 checksums.
AUTHOR494_PARQUET_SHA256 = "338dc73b005b2c8036cc23959061007d7339c24d1988800b62543e1c8224385c"
T023_JSONL_SHA256 = "9ff8449b781906290f9b671bc5be1982ca31e5401ebfba485474d475a60752ce"
T026_TRAIN_PARQUET_SHA256 = "1e85fd0f2d1282b409e8617b62591fd157b165b397aa3ad514ba3a95d1adee22"
T026_VAL_PARQUET_SHA256 = "9a6808eacdf3facfdeb39566dce26f13f6b46c8b9c332e1fd2853709f398355f"

# Machine-written T-026 Step-100 HF bf16 export SHA-256 lock (T-026_阶段3.md Section 6).
ROUTEA_STEP100_LOCK: dict[str, str] = {
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    "config.json": "6d320dedc3a50ea3eb4bf7ab503fed018f21784f001c61ff9de9a7ed411e69f0",
    "generation_config.json": "944b9abc4ec388933cc00c757a7b794683929e5a02e5123853892a5750645689",
    "model.safetensors": "9411088e35efc7106fa0f7ed2d9cc3990bfebb4180faa6f3bab2dce7501a96dd",
    "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "7488550a894dcf741616126e322ea31d66cf9457b59ec5825f4c7efdced421f8",
}

STUDENT_LLAMA8B_LOCK: dict[str, str] = {
    "config.json": "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e",
    "tokenizer.json": "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
    "tokenizer_config.json": "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424",
    "model-00001-of-00004.safetensors": "2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668",
    "model-00002-of-00004.safetensors": "09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15",
    "model-00003-of-00004.safetensors": "fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa",
    "model-00004-of-00004.safetensors": "92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b",
}

DEFAULT_BASE = Path.home() / "bzhang" / "a_group_pro5000"
DEFAULT_ROUTEA_RUN_REL = Path("runs/t026_stage3_formal_20261126_140508")
DEFAULT_ROUTEA_EXPORT_REL = DEFAULT_ROUTEA_RUN_REL / "exports" / "step_100"
DEFAULT_ROUTEA_SHA_REL = DEFAULT_ROUTEA_RUN_REL / "export_step100_sha256.txt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_key(source_id: str) -> str:
    return hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:24]


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def verify_tutor_bfloat16(model_dir: Path) -> str:
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        raise RuntimeError(f"Tutor config.json missing: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    dtype = str(cfg.get("torch_dtype") or cfg.get("dtype") or "")
    if dtype != "bfloat16":
        raise RuntimeError(f"Expected Tutor config dtype='bfloat16', got {dtype!r}")
    return dtype


def parse_sha256_manifest(sha_file: Path) -> dict[str, str]:
    if not sha_file.is_file():
        raise RuntimeError(f"SHA-256 manifest missing: {sha_file}")
    out: dict[str, str] = {}
    for raw in sha_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        digest = parts[0].lower()
        name = Path(parts[-1].lstrip("*")).name
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise RuntimeError(f"Invalid SHA-256 entry in {sha_file}: {name}")
        out[name] = digest
    if not out:
        raise RuntimeError(f"Empty SHA-256 manifest: {sha_file}")
    return out


def verify_dir_against_lock(
    target_dir: Path,
    lock: dict[str, str],
    *,
    label: str,
    manifest_file: Optional[Path] = None,
) -> dict[str, str]:
    if not target_dir.is_dir():
        raise RuntimeError(f"{label} directory missing: {target_dir}")
    if manifest_file is not None:
        manifest_map = parse_sha256_manifest(manifest_file)
        if manifest_map != lock:
            raise RuntimeError(
                f"{label} manifest mismatch with hardcoded lock: "
                f"extra={sorted(set(manifest_map) - set(lock))}, "
                f"missing={sorted(set(lock) - set(manifest_map))}"
            )
    verified: dict[str, str] = {}
    for fname, exp in lock.items():
        fpath = target_dir / fname
        if not fpath.is_file() or fpath.is_symlink():
            raise RuntimeError(f"{label} missing or symlinked file: {fpath}")
        actual = sha256_file(fpath)
        if actual != exp:
            raise RuntimeError(f"{label} SHA-256 mismatch for {fname}: actual={actual} expected={exp}")
        verified[fname] = actual
    return verified


def locate_eduardo_repo(base: Path) -> Path:
    candidates = []
    if os.environ.get("EDUARDO_REPO"):
        candidates.append(Path(os.environ["EDUARDO_REPO"]))
    candidates.extend([
        base / "source" / "EduardoRL",
        Path("/workspace/A_group_repro_20261117/source/EduardoRL"),
        REPO_ROOT.parent / "EduardoRL_A组源码_20261117",
    ])
    for cand in candidates:
        if (cand / "eval" / "run_env_test_eval.py").is_file():
            return cand
    raise RuntimeError("EduardoRL upstream repo not found; set EDUARDO_REPO")


def resolve_local_model(base: Path, name: str, keywords: tuple[str, ...]) -> Path:
    candidates = [
        Path("/dev/shm/zhujie_a_group") / name,
        base / "models" / name,
    ]
    for c in candidates:
        if c.exists() and (c / "config.json").is_file():
            return c.resolve(strict=True)
    for root in (Path("/dev/shm/zhujie_a_group"), base / "models"):
        if root.is_dir():
            for entry in sorted(root.iterdir()):
                if entry.is_dir() and (entry / "config.json").is_file():
                    if any(k in entry.name.lower() for k in keywords):
                        return entry.resolve(strict=True)
    raise RuntimeError(f"Model {name} not found under /dev/shm/zhujie_a_group or {base / 'models'}")


def parse_gpu_snapshot(gpu_csv: str, app_csv: str) -> dict[int, dict[str, Any]]:
    apps_by_uuid: dict[str, list[dict[str, int]]] = {}
    for line in app_csv.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        uuid, pid_s, used_s = parts[0], parts[1], parts[2]
        apps_by_uuid.setdefault(uuid, []).append(
            {"pid": int(pid_s), "used_mib": int(used_s)}
        )
    gpus: dict[int, dict[str, Any]] = {}
    for line in gpu_csv.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        idx = int(parts[0])
        free_mib = int(parts[3])
        total_mib = int(parts[4])
        used_mib = int(parts[2])
        gpus[idx] = {
            "index": idx,
            "uuid": parts[1],
            "used_mib": used_mib,
            "free_mib": free_mib,
            "total_mib": total_mib,
            "util": int(parts[5]),
            "apps": apps_by_uuid.get(parts[1], []),
        }
    return gpus


def snapshot_gpus() -> dict[int, dict[str, Any]]:
    gpu_csv = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=20,
    )
    app_csv = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=20,
    )
    return parse_gpu_snapshot(gpu_csv, app_csv)


def select_free_gpus(
    first: dict[int, dict[str, Any]],
    second: dict[int, dict[str, Any]],
    *,
    max_used_mib: int = 500,
    min_free_mib: int = 68000,
) -> list[int]:
    """Select GPUs with <max_used_mib, >=min_free_mib, and 0 apps across two snapshots.

    Follows 资源账本.md: GPU 6 reports util=100% when idle, so util is used only for
    tie-breaking preference (util < 10 first), never for exclusion.
    """
    preferred: list[int] = []
    quirk: list[int] = []
    for idx in sorted(set(first) & set(second)):
        a, b = first[idx], second[idx]
        if a["uuid"] != b["uuid"]:
            continue
        ok_a = not a["apps"] and a["used_mib"] < max_used_mib and a["free_mib"] >= min_free_mib
        ok_b = not b["apps"] and b["used_mib"] < max_used_mib and b["free_mib"] >= min_free_mib
        if ok_a and ok_b:
            if b["util"] < 10:
                preferred.append(idx)
            else:
                quirk.append(idx)
    return preferred + quirk


def wait_for_free_gpus(
    num_gpus: int,
    *,
    poll_seconds: int = 60,
    max_wait_minutes: int = 0,
    min_free_mib: int = 68000,
) -> list[int]:
    if num_gpus <= 0:
        raise ValueError("num_gpus must be positive")
    deadline = time.monotonic() + max_wait_minutes * 60.0 if max_wait_minutes > 0 else None
    polls = 0
    while True:
        s1 = snapshot_gpus()
        time.sleep(2)
        s2 = snapshot_gpus()
        free = select_free_gpus(s1, s2, min_free_mib=min_free_mib)
        if len(free) >= num_gpus:
            picked = free[:num_gpus]
            print(f"[t033_gpu] acquired {num_gpus} GPUs: {picked} (all_free={free})", file=sys.stderr, flush=True)
            return picked
        polls += 1
        if deadline is not None and time.monotonic() >= deadline:
            raise RuntimeError(f"Timed out waiting for {num_gpus} idle GPUs (free={free})")
        print(f"[t033_gpu] poll={polls} free={free} (<{num_gpus}), sleeping {poll_seconds}s...", file=sys.stderr, flush=True)
        time.sleep(poll_seconds)


def pick_free_ports(preferred: list[int]) -> list[int]:
    ports: list[int] = []
    socks: list[socket.socket] = []
    try:
        for port in preferred:
            s = socket.socket()
            try:
                s.bind(("127.0.0.1", port))
                ports.append(port)
                socks.append(s)
            except OSError:
                s.close()
                s2 = socket.socket()
                s2.bind(("127.0.0.1", 0))
                ports.append(int(s2.getsockname()[1]))
                socks.append(s2)
        return ports
    finally:
        for s in socks:
            s.close()


def wait_for_health(url: str, proc: subprocess.Popen, label: str, timeout_s: float = 600.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(f"{label} exited early with rc={rc}")
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return
        except OSError:
            pass
        time.sleep(2)
    raise RuntimeError(f"{label} health check timed out ({timeout_s}s)")


def terminate_procs(procs: list[tuple[str, subprocess.Popen]]) -> None:
    for _, proc in reversed(procs):
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 30.0
    for _, proc in reversed(procs):
        if proc.poll() is None:
            try:
                proc.wait(timeout=max(0.5, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=10)


def build_clean_env(base: Path, *, token: str, eduardo_repo: Path) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.upper().endswith(("_KEY", "_TOKEN", "_SECRET", "_PASSWORD"))
        and k.upper() not in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "CUDA_VISIBLE_DEVICES")
    }
    env_lib = Path(sys.prefix) / "lib"
    ld_parts = [str(env_lib)] if env_lib.is_dir() else []
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    env.update(
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        VLLM_NO_USAGE_STATS="1",
        WANDB_MODE="disabled",
        DO_NOT_TRACK="1",
        NO_PROXY="127.0.0.1,localhost,::1",
        no_proxy="127.0.0.1,localhost,::1",
        VLLM_API_KEY=token,
        SERVING_API_KEY=token,
        TMPDIR=str(base / "tmp"),
        HF_HOME=str(base / "cache" / "hf"),
        TRITON_CACHE_DIR=str(base / "cache" / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(base / "cache" / "inductor"),
        VLLM_CACHE_ROOT=str(base / "cache" / "vllm"),
        XDG_CACHE_HOME=str(base / "cache" / "xdg"),
        PYTHONPATH=f"{REPO_ROOT}:{eduardo_repo}",
    )
    if ld_parts:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_parts)
    return env


def ensure_base_cache_dirs(base: Path) -> None:
    for sub in ("tmp", "cache/hf", "cache/triton", "cache/inductor", "cache/vllm", "cache/xdg", "runs"):
        (base / sub).mkdir(parents=True, exist_ok=True)


def locate_file_by_sha256(
    roots: list[Path],
    filename: str,
    expected_sha256: str,
) -> Optional[Path]:
    for root in roots:
        if not root.is_dir():
            continue
        for cand in sorted(root.rglob(filename)):
            if cand.is_file() and not cand.is_symlink():
                if sha256_file(cand) == expected_sha256:
                    return cand
    return None


def disk_free_gb(path: Path) -> float:
    target = path if path.exists() else path.parent
    return round(shutil.disk_usage(target).free / 1e9, 2)


def kill_pid_tree(root_pid: int) -> None:
    """Terminate a process and its /proc descendants cleanly without touching own PGID."""
    children_map: dict[int, list[int]] = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            st = (p / "stat").read_text()
            ppid = int(st[st.rfind(")") + 2 :].split()[1])
            children_map.setdefault(ppid, []).append(int(p.name))
        except OSError:
            continue
    order: list[int] = []
    stack = [root_pid]
    while stack:
        cur = stack.pop()
        order.append(cur)
        stack.extend(children_map.get(cur, []))
    for pid in reversed(order):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    time.sleep(5)
    for pid in reversed(order):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def format_machine_summary(fields: dict[str, Any]) -> str:
    """Format fixed machine-readable 'key: value' summary block (工人须知第 13 条)."""
    lines = []
    for k, v in fields.items():
        key = str(k).strip().replace(" ", "_")
        lines.append(f"{key}: {v}")
    return "\n".join(lines)
