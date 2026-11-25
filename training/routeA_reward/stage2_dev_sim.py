"""Pro 5000 Stage 2 dev-set simulation orchestrator for T-024 (Route A reward).

Workflow:
  1. Preflight (CPU):
     - Verify Python packages: ``math-verify==0.8.0``, ``latex2sympy2_extended==1.10.2``
     - Verify SHA-256 of ``hybrid_grader.py`` (``7d2a5f46...``), T-023 measured JSONL
       (``9ff8449b...``), ``exports/step_100`` weights, base ``Qwen3.5-4B``, Student
       ``Llama-3.1-8B``, Grader ``Qwen3.6-27B``, and ``EduardoRL@e8c89b7``
     - Re-run all unit tests (``test_routeA_reward.py`` + ``test_stage2_dev_sim.py``)
       inside ``pro5000-fast`` (with real ``math-verify==0.8.0``)
  2. Acquire 4-5 free Pro 5000 GPUs (``used_mib < 500`` & 0 apps, double-checked):
     - 5 GPUs: 1 Tutor + 2 Student + 2 Grader (TP=2)
     - 4 GPUs: 1 Tutor + 1 Student + 2 Grader (TP=2)
  3. Generate 2 dialogues per dev source problem (100 dev sources, 178 variants)
     for ``base`` (Qwen3.5-4B) and ``a_group`` (step_100), using exact training
     rollout settings (bf16, temp=1.15, max_tokens=662, enable_thinking=True,
     max_user_turns=10, min_teacher_turns_before_eoc=3, student temp=0.6/500).
  4. Climb the ladder 16 times per dialogue using ``hybrid_grader.py`` (no gate
     judge), with per-dialogue checkpointing for ``--resume`` and >1% technical
     failure auto-stop.
  5. Run checks (a)-(g) via ``stage2_analysis.py`` and write numeric-only
     ``stage2_summary.json`` (ZERO problem/answer/response text printed or exported).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
WORKTREE_ROOT = HERE.parent.parent
if str(WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))

EXPECTED_HYBRID_GRADER_SHA256 = (
    "7d2a5f46619218e92ed11efc4e7e07f190027ac0f1cca9336905c090ff62709a"
)
EXPECTED_T023_JSONL_DEFAULT = Path(
    "/data/users/zhujie/bzhang/a_group_pro5000/runs/"
    "t023_stage2_20261125_163319/routeA_stage2_measured.jsonl"
)
EXPECTED_T023_JSONL_SHA256 = (
    "9ff8449b781906290f9b671bc5be1982ca31e5401ebfba485474d475a60752ce"
)
EXPECTED_DEV_SOURCES = 100
EXPECTED_DEV_VARIANTS = 178
EXPECTED_AUTHOR_COMMIT = "e8c89b7293e65e632149571b16a1c2d4e33c77fe"
DEFAULT_BASE = Path.home() / "bzhang" / "a_group_pro5000"
DEFAULT_BASE_TUTOR = Path("/data/models/Qwen3.5-4B")
DEFAULT_AGROUP_TUTOR = Path(
    "/data/users/zhujie/bzhang/a_group_pro5000/runs/"
    "stage3_formal_20261121_144311/exports/step_100"
)
DEFAULT_STEP100_SHA_FILE = Path(
    "/data/users/zhujie/bzhang/a_group_pro5000/runs/"
    "stage3_formal_20261121_144311/export_step100_sha256.txt"
)

# Exact training rollout settings from eduardo/configs/user.yaml & eduardo_interaction_config.yaml
TUTOR_TEMPERATURE = 1.15
TUTOR_MAX_TOKENS_PER_TURN = 662  # 150 visible + 512 thinking
TUTOR_MAX_MODEL_LEN = 10240
TUTOR_SEQ_STOP_BUFFER = 700
STUDENT_INTERACTION_TEMP = 0.6
STUDENT_INTERACTION_MAX_TOKENS = 500
STUDENT_FINAL_TEMP = 0.6
STUDENT_FINAL_MAX_TOKENS = 800
MAX_USER_TURNS = 10
MIN_TEACHER_TURNS_BEFORE_EOC = 3
NUM_DIALOGUES_PER_SOURCE = 2
NUM_CLIMBS_PER_DIALOGUE = 16
END_OF_CONVERSATION = "<end_of_conversation>"
STUDENT_OPENING_TEMPLATE = "This is the problem I am trying to understand: {problem}"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def parse_gpu_snapshot(gpu_csv: str, app_csv: str) -> dict[int, dict[str, Any]]:
    apps_by_uuid: dict[str, list[dict[str, int]]] = {}
    for line in app_csv.splitlines():
        if not line.strip():
            continue
        uuid, pid_s, used_s = [p.strip() for p in line.split(",")]
        apps_by_uuid.setdefault(uuid, []).append(
            {"pid": int(pid_s), "used_mib": int(used_s)}
        )
    gpus: dict[int, dict[str, Any]] = {}
    for line in gpu_csv.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        idx = int(parts[0])
        gpus[idx] = {
            "index": idx,
            "uuid": parts[1],
            "used_mib": int(parts[2]),
            "free_mib": int(parts[3]),
            "total_mib": int(parts[4]),
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
        timeout=15,
    )
    app_csv = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    )
    return parse_gpu_snapshot(gpu_csv, app_csv)


def select_free_gpus(
    first: dict[int, dict[str, Any]],
    second: dict[int, dict[str, Any]],
    *,
    max_used_mib: int = 500,
    min_free_mib: int = 65000,
) -> list[int]:
    """Return GPUs idle across two snapshots (GPU 6 ignores sensor util per ledger)."""
    free: list[int] = []
    for idx in sorted(set(first.keys()) & set(second.keys())):
        a, b = first[idx], second[idx]
        if (
            not a["apps"]
            and not b["apps"]
            and a["used_mib"] < max_used_mib
            and b["used_mib"] < max_used_mib
            and a["free_mib"] >= min_free_mib
            and b["free_mib"] >= min_free_mib
        ):
            free.append(idx)
    return free


def allocate_gpu_layout(free_gpus: list[int]) -> Optional[dict[str, Any]]:
    """Assign 5 GPUs (1 tutor + 2 student + 2 grader) or 4 GPUs (1+1+2)."""
    if len(free_gpus) >= 5:
        picked = free_gpus[:5]
        return {
            "num_gpus": 5,
            "tutor_gpu": picked[0],
            "student_gpus": picked[1:3],
            "grader_gpus": picked[3:5],
        }
    if len(free_gpus) == 4:
        picked = free_gpus[:4]
        return {
            "num_gpus": 4,
            "tutor_gpu": picked[0],
            "student_gpus": picked[1:2],
            "grader_gpus": picked[2:4],
        }
    return None


def wait_for_gpu_layout(
    *,
    poll_seconds: int = 60,
    max_wait_minutes: int = 0,
    prefer_5_grace_polls: int = 0,
) -> dict[str, Any]:
    """Poll nvidia-smi until 4 or 5 free GPUs are available."""
    deadline = (
        time.monotonic() + max_wait_minutes * 60.0 if max_wait_minutes > 0 else None
    )
    polls = 0
    while True:
        first = snapshot_gpus()
        time.sleep(2)
        second = snapshot_gpus()
        free = select_free_gpus(first, second)
        layout = allocate_gpu_layout(free)
        if layout is not None:
            if layout["num_gpus"] == 5 or polls >= prefer_5_grace_polls:
                print(
                    f"[gpu_wait] acquired {layout['num_gpus']} GPUs: "
                    f"tutor={layout['tutor_gpu']} student={layout['student_gpus']} "
                    f"grader={layout['grader_gpus']} (all_free={free})",
                    flush=True,
                )
                return layout
        polls += 1
        if deadline is not None and time.monotonic() >= deadline:
            raise RuntimeError(f"等待 4-5 张空卡超时，当前空卡={free}")
        print(
            f"[gpu_wait] poll={polls} free={free} (<4), sleeping {poll_seconds}s...",
            flush=True,
        )
        time.sleep(poll_seconds)


def verify_packages() -> dict[str, Optional[str]]:
    """Verify exact math-verify==0.8.0 and latex2sympy2_extended==1.10.2."""
    versions: dict[str, Optional[str]] = {}
    for pkg in ("math-verify", "latex2sympy2-extended", "latex2sympy2", "vllm", "torch", "openai"):
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = None
    if versions["math-verify"] != "0.8.0":
        raise RuntimeError(
            f"math-verify 版本不符: actual={versions['math-verify']} expected=0.8.0"
        )
    if versions["latex2sympy2-extended"] != "1.10.2":
        raise RuntimeError(
            f"latex2sympy2-extended 版本不符: actual={versions['latex2sympy2-extended']} expected=1.10.2"
        )
    return versions


def verify_model_dir(label: str, path: Path) -> Path:
    if path.is_symlink() and not path.exists():
        raise RuntimeError(f"{label} 模型软链接失效: {path}")
    if not path.is_dir():
        raise RuntimeError(f"缺少 {label} 模型目录: {path}")
    resolved = path.resolve(strict=True)
    if not (resolved / "config.json").is_file():
        raise RuntimeError(f"{label} 模型缺少 config.json: {path}")
    shards = [p for p in resolved.glob("*.safetensors") if p.is_file() and p.stat().st_size > 0]
    if not shards:
        raise RuntimeError(f"{label} 模型缺少有效 *.safetensors: {path}")
    idx_file = resolved / "model.safetensors.index.json"
    if idx_file.is_file():
        wmap = json.loads(idx_file.read_text(encoding="utf-8")).get("weight_map", {})
        for sname in sorted(set(wmap.values())):
            sp = resolved / sname
            if not sp.is_file() or sp.stat().st_size <= 0:
                raise RuntimeError(f"{label} 模型分片缺失: {sp}")
    return path


def verify_step100_sha_manifest(step100_dir: Path, sha_file: Path) -> dict[str, str]:
    """Verify exports/step_100 against machine-generated export_step100_sha256.txt."""
    if not sha_file.is_file():
        raise RuntimeError(f"缺少 step_100 校验清单: {sha_file}")
    verified: dict[str, str] = {}
    for raw in sha_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        expected_sha = parts[0].lower()
        rel = parts[-1].lstrip("*")
        fpath = step100_dir / Path(rel).name
        if not fpath.is_file():
            raise RuntimeError(f"step_100 缺少文件: {fpath}")
        actual = file_sha256(fpath)
        if actual != expected_sha:
            raise RuntimeError(
                f"step_100 文件 {fpath.name} SHA-256 不符: actual={actual} expected={expected_sha}"
            )
        verified[fpath.name] = actual
    if not verified:
        raise RuntimeError(f"step_100 校验清单为空: {sha_file}")
    return verified


def resolve_local_model(base: Path, preferred: Path, keywords: tuple[str, ...]) -> Path:
    candidates = [
        preferred,
        Path("/dev/shm/zhujie_a_group") / preferred.name,
        base / "models" / preferred.name,
    ]
    for c in candidates:
        if c.is_dir():
            return c
    for root in (Path("/dev/shm/zhujie_a_group"), base / "models"):
        if root.is_dir():
            for entry in sorted(root.iterdir()):
                if entry.is_dir() and any(k in entry.name.lower() for k in keywords):
                    return entry
    return preferred


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
            raise RuntimeError(f"{label} 服务提前退出 rc={rc}")
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return
        except OSError:
            pass
        time.sleep(2)
    raise RuntimeError(f"{label} 健康检查超时 ({timeout_s}s)")


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
        and k.upper() not in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
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
        PYTHONPATH=f"{WORKTREE_ROOT}:{eduardo_repo}",
    )
    if ld_parts:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_parts)
    return env


def run_preflight(
    *,
    base: Path,
    eduardo_repo: Path,
    t023_jsonl: Path,
    base_tutor: Path,
    agroup_tutor: Path,
    step100_sha_file: Path,
    student_model: Path,
    grader_model: Path,
    skip_tests: bool = False,
) -> dict[str, Any]:
    """Run all pre-GPU checks: packages, SHA-256s, models, EduardoRL commit, pytest."""
    versions = verify_packages()

    # 1. hybrid_grader.py SHA-256 (exact match to T-023)
    hg_path = HERE / "hybrid_grader.py"
    hg_sha = file_sha256(hg_path)
    if hg_sha != EXPECTED_HYBRID_GRADER_SHA256:
        raise RuntimeError(
            f"hybrid_grader.py SHA-256 不符: actual={hg_sha} expected={EXPECTED_HYBRID_GRADER_SHA256}"
        )

    # 2. T-023 measured JSONL SHA-256 & dev split schema/count validation (no text printed)
    if not t023_jsonl.is_file():
        raise RuntimeError(f"缺少 T-023 重测包: {t023_jsonl}")
    jsonl_sha = file_sha256(t023_jsonl)
    if jsonl_sha != EXPECTED_T023_JSONL_SHA256:
        raise RuntimeError(
            f"T-023 重测包 SHA-256 不符: actual={jsonl_sha} expected={EXPECTED_T023_JSONL_SHA256}"
        )
    parsed_dev = load_dev_problems(t023_jsonl)
    dev_sources = len(parsed_dev)
    dev_variants = sum(len(r["variants"]) for r in parsed_dev)
    if dev_sources != EXPECTED_DEV_SOURCES or dev_variants != EXPECTED_DEV_VARIANTS:
        raise RuntimeError(
            f"dev 题量不符: sources={dev_sources} (expect {EXPECTED_DEV_SOURCES}), "
            f"variants={dev_variants} (expect {EXPECTED_DEV_VARIANTS})"
        )

    # 3. Author upstream EduardoRL commit check
    if not (eduardo_repo / ".git").exists():
        raise RuntimeError(f"缺少作者仓库: {eduardo_repo}")
    rev = subprocess.check_output(
        ["git", "-C", str(eduardo_repo), "rev-parse", "HEAD"],
        text=True,
        timeout=30,
    ).strip()
    if rev != EXPECTED_AUTHOR_COMMIT:
        raise RuntimeError(f"EduardoRL 提交不符: actual={rev} expected={EXPECTED_AUTHOR_COMMIT}")

    # 4. Model directories & step_100 manifest check
    verify_model_dir("BaseTutor", base_tutor)
    verify_model_dir("AGroupStep100", agroup_tutor)
    step100_hashes = verify_step100_sha_manifest(agroup_tutor, step100_sha_file)
    verify_model_dir("Student", student_model)
    verify_model_dir("Grader27B", grader_model)

    # 5. Disk check
    free_gb = round(shutil.disk_usage("/data").free / 1e9, 2) if Path("/data").exists() else round(shutil.disk_usage(base).free / 1e9, 2)
    if free_gb < 10.0:
        raise RuntimeError(f"/data 剩余空间过低: {free_gb} GB < 10 GB")

    # 6. Re-run ALL unit tests in pro5000-fast (with real math-verify==0.8.0) via stdlib unittest
    unittest_summary = "skipped"
    if not skip_tests:
        test_env = dict(os.environ, PYTHONPATH=f"{WORKTREE_ROOT}:{eduardo_repo}", PYTHONDONTWRITEBYTECODE="1")
        cmd = [
            sys.executable, "-B", "-m", "unittest", "discover",
            "-s", str(HERE), "-p", "test_*.py", "-v",
        ]
        proc = subprocess.run(cmd, env=test_env, capture_output=True, text=True, timeout=300)
        combined = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
        tail_lines = [ln for ln in combined.splitlines() if ln.strip()][-6:]
        if proc.returncode != 0:
            raise RuntimeError(
                f"pro5000-fast 单测失败 rc={proc.returncode}:\n" + "\n".join(tail_lines)
            )
        if "skipped=" in "\n".join(tail_lines).lower():
            raise RuntimeError(
                f"pro5000-fast 单测不应有跳过项（math-verify==0.8.0 必须实跑）:\n" + "\n".join(tail_lines)
            )
        unittest_summary = " | ".join(tail_lines[-2:])

    return {
        "packages": versions,
        "hybrid_grader_sha256": hg_sha,
        "t023_jsonl_sha256": jsonl_sha,
        "dev_sources": dev_sources,
        "dev_variants": dev_variants,
        "author_commit": rev,
        "step100_verified_files": len(step100_hashes),
        "disk_free_gb": free_gb,
        "unittest_summary": unittest_summary,
    }


def load_dev_problems(t023_jsonl: Path) -> list[dict[str, Any]]:
    """Load only split='dev' problems with pre-measured p_new/b/ladder_rank."""
    rows: list[dict[str, Any]] = []
    with t023_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "dev":
                continue
            src_prob = (
                row.get("source_problem")
                if row.get("source_problem") is not None
                else row.get("problem")
            )
            src_ans = (
                row.get("source_answer")
                if row.get("source_answer") is not None
                else row.get("answer")
            )
            if not isinstance(src_prob, str) or not src_prob.strip():
                raise ValueError(f"dev source_id={row.get('source_id')} missing source problem")
            if not isinstance(src_ans, str) or not src_ans.strip():
                raise ValueError(f"dev source_id={row.get('source_id')} missing source answer")
            raw_vars = row.get("variants") or []
            if not isinstance(raw_vars, list) or not raw_vars:
                raise ValueError(f"dev source_id={row.get('source_id')} has no variants")
            for v in raw_vars:
                v_prob = v.get("problem")
                v_ans = v.get("answer")
                if not isinstance(v_prob, str) or not v_prob.strip():
                    raise ValueError(f"dev variant={v.get('variant_id')} missing problem")
                if not isinstance(v_ans, str) or not v_ans.strip():
                    raise ValueError(f"dev variant={v.get('variant_id')} missing answer")
                if not (math.isfinite(float(v["p_new"])) and math.isfinite(float(v["b"]))):
                    raise ValueError(f"dev variant={v.get('variant_id')} has non-finite p_new/b")
                if type(v.get("ladder_rank")) is not int:
                    raise ValueError(f"dev variant={v.get('variant_id')} missing int ladder_rank")
                if type(v.get("n_correct_hybrid")) is not int:
                    raise ValueError(f"dev variant={v.get('variant_id')} missing int n_correct_hybrid")
            variants = sorted(
                raw_vars,
                key=lambda v: (int(v["ladder_rank"]), str(v.get("variant_id", ""))),
            )
            rows.append(
                {
                    "source_id": str(row["source_id"]),
                    "problem": src_prob,
                    "answer": src_ans,
                    "variants": variants,
                }
            )
    rows.sort(key=lambda r: r["source_id"])
    return rows


async def rollout_one_dialogue(
    *,
    problem: str,
    tutor_client: Any,
    tutor_model: str,
    student_client: Any,
    student_model: str,
    teacher_system_prompt: str,
    student_system_prompt: str,
    strip_teacher_controls_fn: Any,
    tutor_semaphore: asyncio.Semaphore,
    student_semaphore: asyncio.Semaphore,
    max_user_turns: int = MAX_USER_TURNS,
    min_teacher_turns_before_eoc: int = MIN_TEACHER_TURNS_BEFORE_EOC,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Run one multi-turn dialogue matching training rollout config."""
    from eduardo.utils import call_openai_with_retries

    teacher_sys = teacher_system_prompt.format(problem=problem)
    student_sys = student_system_prompt.format(problem=problem)
    opener = STUDENT_OPENING_TEMPLATE.format(problem=problem)

    # Transcript in reward_view format: assistant=teacher, user=student (no system msg).
    transcript: list[dict[str, str]] = [{"role": "user", "content": opener}]
    eoc_used = False
    truncated_seq = False
    teacher_turns = 0
    student_turns = 0
    student_failed = False

    async def _call_tutor(msgs: list[dict[str, str]]) -> tuple[str, int]:
        resp = await tutor_client.chat.completions.create(
            model=tutor_model,
            messages=msgs,
            temperature=TUTOR_TEMPERATURE,
            max_tokens=TUTOR_MAX_TOKENS_PER_TURN,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        text = resp.choices[0].message.content or ""
        total_tok = int(getattr(getattr(resp, "usage", None), "total_tokens", 0) or 0)
        return text, total_tok

    async def _call_student(msgs: list[dict[str, str]]) -> str:
        resp = await student_client.chat.completions.create(
            model=student_model,
            messages=msgs,
            temperature=STUDENT_INTERACTION_TEMP,
            max_tokens=STUDENT_INTERACTION_MAX_TOKENS,
        )
        return (resp.choices[0].message.content or "").strip()

    for turn_idx in range(max_user_turns + 1):
        tutor_msgs = [{"role": "system", "content": teacher_sys}, *transcript]
        tutor_text, total_tok = await call_openai_with_retries(
            _call_tutor,
            tutor_msgs,
            label="tutor_rollout",
            num_retries=3,
            backoff_base=2.0,
            backoff_cap=30.0,
            jitter=True,
            use_fallback=False,
            semaphore=tutor_semaphore,
        )
        transcript.append({"role": "assistant", "content": tutor_text})
        teacher_turns += 1

        # Honour <end_of_conversation> after >= min_teacher_turns_before_eoc prior turns.
        n_prior = teacher_turns - 1
        if END_OF_CONVERSATION in tutor_text and n_prior >= min_teacher_turns_before_eoc:
            eoc_used = True
            break

        if turn_idx >= max_user_turns:
            break
        if total_tok >= TUTOR_MAX_MODEL_LEN - TUTOR_SEQ_STOP_BUFFER:
            truncated_seq = True
            break

        # Student perspective (roles flipped, teacher thinking/controls stripped).
        student_msgs: list[dict[str, str]] = [{"role": "system", "content": student_sys}]
        for m in transcript:
            if m["role"] == "assistant":
                student_msgs.append(
                    {
                        "role": "user",
                        "content": strip_teacher_controls_fn(
                            m["content"], assume_thinking=True
                        ),
                    }
                )
            else:
                student_msgs.append({"role": "assistant", "content": m["content"]})

        try:
            student_reply = await call_openai_with_retries(
                _call_student,
                student_msgs,
                label="student_interaction",
                num_retries=3,
                backoff_base=2.0,
                backoff_cap=30.0,
                jitter=True,
                use_fallback=False,
                semaphore=student_semaphore,
            )
        except Exception:
            student_failed = True
            break

        transcript.append({"role": "user", "content": student_reply})
        student_turns += 1

    return transcript, {
        "teacher_turns": teacher_turns,
        "student_turns": student_turns,
        "eoc_used": eoc_used,
        "truncated_seq": truncated_seq,
        "student_interaction_failed": student_failed,
    }


async def climb_16_recorded(
    *,
    transcript: list[dict[str, str]],
    problem: str,
    variants: list[dict[str, Any]],
    student_client: Any,
    student_model: str,
    grader: Any,
    student_semaphore: asyncio.Semaphore,
    num_climbs: int = NUM_CLIMBS_PER_DIALOGUE,
) -> dict[str, Any]:
    """Run 16 independent ladder climbs and record per-climb numeric/bool outcomes."""
    from eduardo.metrics import correctness as author_correctness
    from eduardo.utils import call_openai_with_retries

    climbs_steps: list[list[dict[str, Any]]] = [[] for _ in range(num_climbs)]
    active = list(range(num_climbs))

    student_calls = 0
    student_failed = 0
    grader_rule_correct = 0
    grader_llm_calls = 0
    grader_llm_unknown = 0
    grader_tech_failed = 0
    grader_llm_time_s = 0.0
    grader_llm_latencies_ms: list[float] = []

    for variant in variants:
        if not active:
            break
        v_prob = str(variant["problem"])
        v_ans = str(variant["answer"])
        v_b = float(variant["b"])
        v_p = float(variant["p_new"])
        v_rank = int(variant["ladder_rank"])
        v_id = str(variant.get("variant_id", ""))

        final_msgs = author_correctness._build_final_attempt_messages(
            transcript,
            problem,
            transfer_problem=v_prob,
            omit_teacher_turns=True,
        )

        async def _sample_one(msgs_payload=final_msgs) -> Optional[str]:
            return await call_openai_with_retries(
                author_correctness._sample_once,
                student_client,
                student_model,
                msgs_payload,
                STUDENT_FINAL_TEMP,
                STUDENT_FINAL_MAX_TOKENS,
                label="student_final",
                num_retries=3,
                backoff_base=2.0,
                backoff_cap=30.0,
                jitter=True,
                use_fallback=True,
                fallback=None,
                semaphore=student_semaphore,
            )

        student_calls += len(active)
        responses = await asyncio.gather(*[_sample_one() for _ in active])

        async def _grade_timed(resp_text: str):
            t0 = time.perf_counter()
            gres = await grader.grade(v_prob, v_ans, resp_text)
            dt = time.perf_counter() - t0
            return gres, dt

        gradable_pos: list[int] = []
        grade_tasks = []
        for pos, text in enumerate(responses):
            cid = active[pos]
            if text is None:
                student_failed += 1
                climbs_steps[cid].append(
                    {
                        "variant_id": v_id,
                        "ladder_rank": v_rank,
                        "b": v_b,
                        "p_new": v_p,
                        "correct": False,
                        "tech_failed": True,
                        "fail_kind": "student",
                    }
                )
            else:
                gradable_pos.append(pos)
                grade_tasks.append(_grade_timed(text))

        graded = await asyncio.gather(*grade_tasks) if grade_tasks else []

        next_active: list[int] = []
        for pos, (gres, dt_s) in zip(gradable_pos, graded):
            cid = active[pos]
            if gres.rule_correct:
                grader_rule_correct += 1
            if gres.llm_called:
                grader_llm_calls += 1
                grader_llm_time_s += dt_s
                grader_llm_latencies_ms.append(round(dt_s * 1000.0, 2))
            if gres.is_unknown:
                grader_llm_unknown += 1
            if gres.technical_failure:
                grader_tech_failed += 1
                climbs_steps[cid].append(
                    {
                        "variant_id": v_id,
                        "ladder_rank": v_rank,
                        "b": v_b,
                        "p_new": v_p,
                        "correct": False,
                        "tech_failed": True,
                        "fail_kind": "grader",
                    }
                )
                continue

            climbs_steps[cid].append(
                {
                    "variant_id": v_id,
                    "ladder_rank": v_rank,
                    "b": v_b,
                    "p_new": v_p,
                    "correct": bool(gres.correct),
                    "tech_failed": False,
                    "rule_correct": bool(gres.rule_correct),
                    "llm_called": bool(gres.llm_called),
                    "is_unknown": bool(gres.is_unknown),
                }
            )
            if gres.correct:
                next_active.append(cid)

        active = next_active

    return {
        "climbs": [{"climb_index": i, "steps": climbs_steps[i]} for i in range(num_climbs)],
        "student_calls": student_calls,
        "student_failed": student_failed,
        "grader_rule_correct": grader_rule_correct,
        "grader_llm_calls": grader_llm_calls,
        "grader_llm_unknown": grader_llm_unknown,
        "grader_tech_failed": grader_tech_failed,
        "grader_llm_time_s": round(grader_llm_time_s, 4),
        "grader_llm_latencies_ms": grader_llm_latencies_ms,
    }


class RoundRobinClientPool:
    """Thin async wrapper balancing chat.completions.create across 1-2 vLLM replicas."""

    def __init__(self, clients: list[Any]):
        self._clients = list(clients)
        self._idx = 0
        self.chat = type(
            "_Chat",
            (),
            {"completions": type("_Comp", (), {"create": self._create})()},
        )()

    async def _create(self, **kwargs):
        c = self._clients[self._idx % len(self._clients)]
        self._idx += 1
        return await c.chat.completions.create(**kwargs)


async def run_tutor_phase(
    *,
    tutor_label: str,
    dev_rows: list[dict[str, Any]],
    tutor_client: Any,
    tutor_model: str,
    student_client: Any,
    student_model: str,
    grader: Any,
    work_dir: Path,
    tutor_conc: int = 16,
    student_conc: int = 32,
) -> list[dict[str, Any]]:
    """Run all 100 dev problems x 2 dialogues x 16 ladder climbs for one Tutor."""
    from eduardo.utils import load_prompt, strip_teacher_controls
    from training.routeA_reward.hybrid_grader import check_technical_failure_rate

    teacher_sys_prompt = load_prompt("teacher_prompt")
    student_sys_prompt = load_prompt("student_prompt")

    ckpt_dir = work_dir / "checkpoints" / tutor_label
    ckpt_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    priv_dir = work_dir / "private_dialogues" / tutor_label
    priv_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    tutor_sem = asyncio.Semaphore(tutor_conc)
    student_sem = asyncio.Semaphore(student_conc)
    item_sem = asyncio.Semaphore(tutor_conc)

    results: list[dict[str, Any]] = []
    done_count = 0
    total_tasks = len(dev_rows) * NUM_DIALOGUES_PER_SOURCE
    cum_llm_calls = 0
    cum_tech_failed = 0
    t_start = time.monotonic()

    async def _process_one(row: dict[str, Any], d_idx: int) -> dict[str, Any]:
        nonlocal done_count, cum_llm_calls, cum_tech_failed
        sid = row["source_id"]
        safe_sid = sid.replace("/", "_")
        ckpt_path = ckpt_dir / f"{safe_sid}_d{d_idx}.json"
        if ckpt_path.is_file():
            rec = json.loads(ckpt_path.read_text(encoding="utf-8"))
            done_count += 1
            return rec

        async with item_sem:
            t0 = time.monotonic()
            transcript, dmeta = await rollout_one_dialogue(
                problem=row["problem"],
                tutor_client=tutor_client,
                tutor_model=tutor_model,
                student_client=student_client,
                student_model=student_model,
                teacher_system_prompt=teacher_sys_prompt,
                student_system_prompt=student_sys_prompt,
                strip_teacher_controls_fn=strip_teacher_controls,
                tutor_semaphore=tutor_sem,
                student_semaphore=student_sem,
            )
            # Save private transcript on Pro 5000 only (0700), never logged.
            priv_path = priv_dir / f"{safe_sid}_d{d_idx}.json"
            write_json_atomic(priv_path, {"source_id": sid, "dialogue_idx": d_idx, "messages": transcript})

            ladder_res = await climb_16_recorded(
                transcript=transcript,
                problem=row["problem"],
                variants=row["variants"],
                student_client=student_client,
                student_model=student_model,
                grader=grader,
                student_semaphore=student_sem,
                num_climbs=NUM_CLIMBS_PER_DIALOGUE,
            )
            rec = {
                "tutor": tutor_label,
                "source_id": sid,
                "dialogue_idx": d_idx,
                "dialogue_meta": dmeta,
                "wall_s": round(time.monotonic() - t0, 2),
                **ladder_res,
            }
            write_json_atomic(ckpt_path, rec)

            done_count += 1
            cum_llm_calls += int(ladder_res["grader_llm_calls"])
            cum_tech_failed += int(ladder_res["grader_tech_failed"])
            if done_count % 20 == 0 or done_count == total_tasks:
                elapsed = time.monotonic() - t_start
                print(
                    f"[{tutor_label}] progress={done_count}/{total_tasks} "
                    f"elapsed={elapsed:.1f}s llm_calls={cum_llm_calls} "
                    f"grader_tech_fail={cum_tech_failed}",
                    flush=True,
                )
            if cum_llm_calls >= 50 and check_technical_failure_rate(cum_tech_failed, cum_llm_calls):
                raise RuntimeError(
                    f"27B 判分技术失败率超过 1%: {cum_tech_failed}/{cum_llm_calls}"
                )
            return rec

    coros = [
        _process_one(row, d_idx)
        for row in dev_rows
        for d_idx in range(NUM_DIALOGUES_PER_SOURCE)
    ]
    results = await asyncio.gather(*coros)
    return list(results)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--t023-jsonl", type=Path, default=EXPECTED_T023_JSONL_DEFAULT)
    ap.add_argument("--base-tutor", type=Path, default=DEFAULT_BASE_TUTOR)
    ap.add_argument("--agroup-tutor", type=Path, default=DEFAULT_AGROUP_TUTOR)
    ap.add_argument("--step100-sha-file", type=Path, default=DEFAULT_STEP100_SHA_FILE)
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--max-wait-minutes", type=int, default=0)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--tutor-conc", type=int, default=16)
    ap.add_argument("--student-conc", type=int, default=32)
    args = ap.parse_args()

    os.umask(0o077)
    base = args.base.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    for sub in ("tmp", "cache/hf", "cache/triton", "cache/inductor", "cache/vllm", "cache/xdg"):
        (base / sub).mkdir(parents=True, exist_ok=True)

    eduardo_repo = base / "source" / "EduardoRL"
    if str(eduardo_repo) not in sys.path:
        sys.path.insert(0, str(eduardo_repo))

    student_model = resolve_local_model(
        base, Path("Meta-Llama-3.1-8B-Instruct"), ("llama-3.1-8b", "8b-instruct")
    )
    grader_model = resolve_local_model(
        base, Path("Qwen3.6-27B"), ("qwen3.6-27b", "27b")
    )

    print("[preflight] running package, SHA-256, model, and unit-test checks...", flush=True)
    preflight = run_preflight(
        base=base,
        eduardo_repo=eduardo_repo,
        t023_jsonl=args.t023_jsonl,
        base_tutor=args.base_tutor,
        agroup_tutor=args.agroup_tutor,
        step100_sha_file=args.step100_sha_file,
        student_model=student_model,
        grader_model=grader_model,
        skip_tests=args.skip_tests,
    )
    write_json_atomic(run_dir / "preflight.json", preflight)
    print(f"[preflight] OK: {json.dumps(preflight, ensure_ascii=False)}", flush=True)
    if args.preflight_only:
        print("PREFLIGHT_DONE=0", flush=True)
        return 0

    layout = wait_for_gpu_layout(
        poll_seconds=args.poll_seconds,
        max_wait_minutes=args.max_wait_minutes,
    )
    write_json_atomic(run_dir / "gpu_layout.json", layout)

    dev_rows = load_dev_problems(args.t023_jsonl)
    source_meta = {r["source_id"]: {"variants": r["variants"]} for r in dev_rows}

    import openai
    from training.routeA_reward.hybrid_grader import AsyncAdaptiveWindow, HybridGrader
    from training.routeA_reward.stage2_analysis import analyze_stage2

    token = secrets.token_urlsafe(32)
    env_base = build_clean_env(base, token=token, eduardo_repo=eduardo_repo)
    ports = pick_free_ports([19161, 19162, 19163, 19164])
    tutor_port, s0_port, s1_port, g_port = ports

    managed: list[tuple[str, subprocess.Popen]] = []

    def spawn(name: str, cmd: list[str], cuda_devs: str, log_name: str) -> subprocess.Popen:
        log = (run_dir / log_name).open("ab")
        try:
            proc = subprocess.Popen(
                cmd,
                env=dict(env_base, CUDA_VISIBLE_DEVICES=cuda_devs),
                cwd=run_dir,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log.close()
        managed.append((name, proc))
        return proc

    t_gpu_start = time.monotonic()
    try:
        # 1. Start Student vLLM replica(s) (bf16)
        student_ports = [s0_port, s1_port][: len(layout["student_gpus"])]
        for idx, (gpu_id, port) in enumerate(zip(layout["student_gpus"], student_ports)):
            cmd = [
                sys.executable, "-B", "-m", "vllm.entrypoints.openai.api_server",
                "--model", str(student_model),
                "--served-model-name", "student-local",
                "--host", "127.0.0.1", "--port", str(port),
                "--dtype", "bfloat16", "--gpu-memory-utilization", "0.9",
                "--max-model-len", "32768", "--max-num-seqs", "256",
                "--tensor-parallel-size", "1",
                "--enable-chunked-prefill", "--enable-prefix-caching",
            ]
            spawn(f"student_{idx}", cmd, str(gpu_id), f"vllm_student_{idx}.log")

        # 2. Start 27B Grader vLLM (TP=2, bf16)
        g_cmd = [
            sys.executable, "-B", "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(grader_model),
            "--served-model-name", "grader-27b",
            "--host", "127.0.0.1", "--port", str(g_port),
            "--dtype", "bfloat16", "--gpu-memory-utilization", "0.9",
            "--max-model-len", "32768", "--max-num-seqs", "256",
            "--tensor-parallel-size", "2",
            "--enable-chunked-prefill", "--enable-prefix-caching",
        ]
        spawn("grader_27b", g_cmd, ",".join(map(str, layout["grader_gpus"])), "vllm_grader_27b.log")

        for name, proc in managed:
            if name.startswith("student_"):
                p = student_ports[int(name[-1])]
                wait_for_health(f"http://127.0.0.1:{p}/health", proc, name, timeout_s=420)
            elif name == "grader_27b":
                wait_for_health(f"http://127.0.0.1:{g_port}/health", proc, name, timeout_s=600)
        print("[serving] Student & 27B Grader healthy!", flush=True)

        all_records: list[dict[str, Any]] = []
        tutor_specs = [
            ("base", args.base_tutor),
            ("a_group", args.agroup_tutor),
        ]

        async def _execute_tutor_phase(tutor_label: str) -> list[dict[str, Any]]:
            # Create fresh async clients and congestion window inside this event loop.
            t_client = openai.AsyncClient(
                api_key=token,
                base_url=f"http://127.0.0.1:{tutor_port}/v1",
                timeout=300.0,
                max_retries=0,
            )
            s_clients = [
                openai.AsyncClient(
                    api_key=token,
                    base_url=f"http://127.0.0.1:{p}/v1",
                    timeout=300.0,
                    max_retries=0,
                )
                for p in student_ports
            ]
            g_client = openai.AsyncClient(
                api_key=token,
                base_url=f"http://127.0.0.1:{g_port}/v1",
                timeout=300.0,
                max_retries=0,
            )
            try:
                grader = HybridGrader(
                    client=g_client,
                    model="grader-27b",
                    window=AsyncAdaptiveWindow(max_window=16),
                    use_vllm_no_thinking=True,
                )
                return await run_tutor_phase(
                    tutor_label=tutor_label,
                    dev_rows=dev_rows,
                    tutor_client=t_client,
                    tutor_model=f"tutor-{tutor_label}",
                    student_client=RoundRobinClientPool(s_clients),
                    student_model="student-local",
                    grader=grader,
                    work_dir=run_dir,
                    tutor_conc=args.tutor_conc,
                    student_conc=args.student_conc,
                )
            finally:
                await t_client.close()
                for sc in s_clients:
                    await sc.close()
                await g_client.close()

        for tutor_label, tutor_path in tutor_specs:
            t_cmd = [
                sys.executable, "-B", "-m", "vllm.entrypoints.openai.api_server",
                "--model", str(tutor_path),
                "--served-model-name", f"tutor-{tutor_label}",
                "--host", "127.0.0.1", "--port", str(tutor_port),
                "--dtype", "bfloat16", "--gpu-memory-utilization", "0.88",
                "--max-model-len", str(TUTOR_MAX_MODEL_LEN),
                "--max-num-seqs", "64",
                "--tensor-parallel-size", "1",
                "--enable-chunked-prefill", "--enable-prefix-caching",
            ]
            t_proc = spawn(
                f"tutor_{tutor_label}",
                t_cmd,
                str(layout["tutor_gpu"]),
                f"vllm_tutor_{tutor_label}.log",
            )
            try:
                wait_for_health(
                    f"http://127.0.0.1:{tutor_port}/health",
                    t_proc,
                    f"tutor_{tutor_label}",
                    timeout_s=420,
                )
                print(f"[serving] tutor={tutor_label} healthy on GPU {layout['tutor_gpu']}", flush=True)
                recs = asyncio.run(_execute_tutor_phase(tutor_label))
                all_records.extend(recs)
            finally:
                terminate_procs([(f"tutor_{tutor_label}", t_proc)])
                managed = [(n, p) for n, p in managed if p is not t_proc]
                time.sleep(3)

        # Tear down remaining vLLM servers before analysis
        terminate_procs(managed)
        managed.clear()

        gpu_elapsed_s = round(time.monotonic() - t_gpu_start, 2)
        analysis = analyze_stage2(all_records, source_meta)
        summary = {
            "status": "STAGE2_DONE",
            "rc": 0,
            "gpu_elapsed_s": gpu_elapsed_s,
            "gpu_layout": layout,
            "preflight": preflight,
            **analysis,
        }
        write_json_atomic(run_dir / "stage2_summary.json", summary)

        print("=== T-024 STAGE 2 SUMMARY (NUMERIC ONLY) ===", flush=True)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print("STAGE2_DONE=0", flush=True)
        return 0
    finally:
        terminate_procs(managed)


if __name__ == "__main__":
    sys.exit(main())
