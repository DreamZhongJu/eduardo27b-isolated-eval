"""Pro 5000 card-free Stage 1 preflight for T-026 Route A training.

Checks (CPU-only, CUDA_VISIBLE_DEVICES=""):
  1. math-verify==0.8.0 and latex2sympy2_extended==1.10.2 versions
  2. SHA-256 of hybrid_grader.py (7d2a5f46...), T-023 measured JSONL (9ff8449b...),
     EduardoRL@e8c89b7 clean commit, /dev/shm Student & 27B models, Tutor Qwen3.5-4B
  3. /data free space >= 70 GB
  4. All 8 GPUs (0..7) idle across two snapshots 2s apart
  5. Generate Route A train/val parquets into $BASE/data/routeA_t026
  6. Run Hydra --check-only via pro5000_trainer_entry.py
  7. Run all unit tests inside pro5000-fast (real math-verify==0.8.0, zero skips)
  8. Extract T-017 offline training control table (no test sets touched)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent.parent
PRO5000_DIR = CODE_ROOT / "training" / "a_group_pro5000"
for p in (CODE_ROOT, PRO5000_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pro5000_config as pcfg
from stage1_collect import find_local_model
from stage2_pilot import (
    base_subprocess_env,
    build_trainer_env,
    default_base,
    ensure_eduardo_repo,
    make_ray_tmp,
    verify_model_dir,
    write_access_log_config,
    write_interaction_yaml,
)
from stage2_precheck import snapshot_gpus
from stage3_formal import STAGE3_CKPT_HINT_BYTES, STAGE3_DISK_FLOOR_BYTES, verify_stage3_gpus_clean
from training.routeA_reward.hybrid_grader import verify_hybrid_grader_sha256
from training.routeA_train.extract_t017_control import DEFAULT_T017_RUN_DIR, extract_t017_control, format_t017_markdown_table
from training.routeA_train.prepare_parquet import (
    DEFAULT_T023_JSONL,
    EXPECTED_T023_JSONL_SHA256,
    prepare_routea_parquets,
    sha256_file,
)

EXPECTED_HYBRID_GRADER_SHA256 = (
    "7d2a5f46619218e92ed11efc4e7e07f190027ac0f1cca9336905c090ff62709a"
)
UNITTEST_MODULES = (
    "training/routeA_reward/test_routeA_reward.py",
    "training/routeA_reward/test_stage2_dev_sim.py",
    "training/a_group_pro5000/test_stage3_formal.py",
    "training/a_group_pro5000/test_stage2_pilot.py",
    "training/routeA_train/test_routeA_train.py",
)


def verify_grading_packages() -> Dict[str, str]:
    mv_ver = importlib_metadata.version("math-verify")
    l2s_ver = importlib_metadata.version("latex2sympy2-extended")
    if mv_ver != "0.8.0":
        raise RuntimeError(f"math-verify 版本不符: actual={mv_ver} expected=0.8.0")
    if l2s_ver != "1.10.2":
        raise RuntimeError(f"latex2sympy2-extended 版本不符: actual={l2s_ver} expected=1.10.2")
    try:
        l2_base = importlib_metadata.version("latex2sympy2")
    except importlib_metadata.PackageNotFoundError:
        l2_base = "not-installed"
    return {
        "math-verify": mv_ver,
        "latex2sympy2-extended": l2s_ver,
        "latex2sympy2": l2_base,
    }


def verify_model_shards_and_sha(label: str, model_dir: Path, sha_manifest: Path | None = None) -> Dict[str, Any]:
    verify_model_dir(label, model_dir)
    resolved = model_dir.resolve(strict=True)
    shards = sorted(p for p in resolved.glob("*.safetensors") if p.is_file())
    shard_hashes: Dict[str, str] = {}
    for sp in shards:
        shard_hashes[sp.name] = sha256_file(sp)

    manifest_checked = False
    if sha_manifest is not None and sha_manifest.is_file():
        for raw_line in sha_manifest.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            exp_hash, rel_name = parts[0].lower(), parts[1].strip().lstrip("./")
            target = resolved / rel_name
            if not target.is_file():
                raise RuntimeError(f"{label} 模型清单文件缺失: {target}")
            actual = shard_hashes.get(rel_name) or sha256_file(target)
            if actual != exp_hash:
                raise RuntimeError(f"{label} 模型文件 {rel_name} sha256 不一致: {actual} != {exp_hash}")
        manifest_checked = True

    return {
        "path": str(model_dir),
        "resolved": str(resolved),
        "in_dev_shm": str(resolved).startswith("/dev/shm"),
        "num_shards": len(shards),
        "shard_sha256": shard_hashes,
        "manifest_checked": manifest_checked,
    }


def run_all_unittests(py_exe: Path, repo: Path) -> Dict[str, Any]:
    env = dict(
        os.environ,
        PYTHONPATH=f"{CODE_ROOT}:{PRO5000_DIR}:{repo}",
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
    )
    results: List[Dict[str, Any]] = []
    for rel_test in UNITTEST_MODULES:
        tpath = CODE_ROOT / rel_test
        cmd = [str(py_exe), "-B", "-m", "unittest", str(tpath), "-v"]
        proc = subprocess.run(cmd, cwd=CODE_ROOT, env=env, capture_output=True, text=True, timeout=300)
        tail_lines = [ln for ln in (proc.stdout + "\n" + proc.stderr).splitlines() if ln.strip()][-6:]
        if proc.returncode != 0:
            raise RuntimeError(f"单元测试失败 {rel_test} rc={proc.returncode}:\n" + "\n".join(tail_lines))
        if "skipped=" in "\n".join(tail_lines):
            raise RuntimeError(f"pro5000-fast 中不允许跳过测试 ({rel_test}):\n" + "\n".join(tail_lines))
        results.append({"test_file": rel_test, "rc": proc.returncode, "tail": tail_lines})
    return {"passed_files": len(results), "files": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=default_base())
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--t023-jsonl", type=Path, default=DEFAULT_T023_JSONL)
    parser.add_argument("--t017-run-dir", type=Path, default=DEFAULT_T017_RUN_DIR)
    parser.add_argument("--tutor-model", type=Path, default=pcfg.TEACHER_MODEL_DEFAULT)
    parser.add_argument("--judge-concurrency", type=int, default=64)
    parser.add_argument("--grader-max-concurrency", type=int, default=32)
    args = parser.parse_args()
    os.umask(0o077)

    base = args.base.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    py_exe = Path(sys.executable).resolve()

    pkg_versions = verify_grading_packages()
    print(f"[preflight] packages={pkg_versions}", flush=True)

    hg_path = CODE_ROOT / "training" / "routeA_reward" / "hybrid_grader.py"
    hg_sha = verify_hybrid_grader_sha256(hg_path, EXPECTED_HYBRID_GRADER_SHA256)
    print(f"[preflight] hybrid_grader.py sha256={hg_sha}", flush=True)

    repo = ensure_eduardo_repo(base)
    print(f"[preflight] EduardoRL repo={repo} commit={pcfg.AUTHOR_COMMIT}", flush=True)

    receipt = base / "envs" / "pro5000-fast" / "kernel_equivalence.json"
    if not receipt.is_file() or not json.loads(receipt.read_text(encoding="utf-8")).get("accepted"):
        raise RuntimeError("缺少或未通过 Stage 1 快内核等价回执 kernel_equivalence.json")

    models_root = base / "models"
    student_dir = find_local_model(models_root, models_root / "Meta-Llama-3.1-8B-Instruct", ("llama-3.1-8b", "8b-instruct"))
    judge_dir = find_local_model(models_root, models_root / "Qwen3.6-27B", ("qwen3.6-27b", "27b"))

    tutor_info = verify_model_shards_and_sha("Tutor", args.tutor_model)
    student_info = verify_model_shards_and_sha("Student", student_dir, models_root / "sha256_8B.txt")
    judge_info = verify_model_shards_and_sha("Judge", judge_dir, models_root / "sha256_27B.txt")
    if not student_info["in_dev_shm"] or not judge_info["in_dev_shm"]:
        raise RuntimeError(f"Student/27B 未位于 /dev/shm: student={student_info['resolved']} judge={judge_info['resolved']}")
    print(
        f"[preflight] models verified: tutor_shards={tutor_info['num_shards']} "
        f"student_shm={student_info['in_dev_shm']} ({student_info['num_shards']} shards) "
        f"judge_shm={judge_info['in_dev_shm']} ({judge_info['num_shards']} shards)",
        flush=True,
    )

    free_bytes = shutil.disk_usage("/data").free
    print(f"[preflight] /data free={free_bytes / 1e9:.2f} GB ({free_bytes / 1024**3:.2f} GiB)", flush=True)
    if free_bytes < STAGE3_DISK_FLOOR_BYTES:
        raise RuntimeError(f"/data 剩余空间不足 70 GB: {free_bytes / 1e9:.2f} GB")

    first = snapshot_gpus()
    time.sleep(2)
    second = snapshot_gpus()
    verify_stage3_gpus_clean(first, second)
    gpu_summary = {
        str(i): {"used_mib": second[i]["used_mib"], "apps": len(second[i]["apps"])}
        for i in sorted(second)
    }
    print(f"[preflight] all 8 GPUs idle: {gpu_summary}", flush=True)

    data_dir = base / "data" / "routeA_t026"
    t023_sha = sha256_file(args.t023_jsonl)
    if t023_sha != EXPECTED_T023_JSONL_SHA256:
        raise RuntimeError(f"T-023 JSONL sha256 不符: {t023_sha}")
    sys_path_added = False
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
        sys_path_added = True
    try:
        dataset_summary = prepare_routea_parquets(
            input_jsonl=args.t023_jsonl,
            out_dir=data_dir,
            expected_sha256=EXPECTED_T023_JSONL_SHA256,
        )
    finally:
        if sys_path_added and str(repo) in sys.path:
            sys.path.remove(str(repo))
    print(
        f"[preflight] parquets ready: train={dataset_summary['train_after_filter']['n_sources']} "
        f"(removed_ceiling={dataset_summary['train_removed_ceiling']['n_sources']}) "
        f"val={dataset_summary['val_dev']['n_sources']}",
        flush=True,
    )

    # Hydra --check-only with Route A overrides
    hydra_work = Path(tempfile.mkdtemp(dir=base / "tmp", prefix=".t026_hydra_"))
    raytmp = make_ray_tmp(base)
    try:
        (hydra_work / "queue-metrics").mkdir(mode=0o700, exist_ok=True)
        (hydra_work / "routeA-metrics").mkdir(mode=0o700, exist_ok=True)
        interaction = write_interaction_yaml(repo, hydra_work, "Llama-3.1-8B-Instruct-preflight", "http://127.0.0.1:18861/v1", 32)
        env_base = base_subprocess_env(base)
        env_base["VLLM_API_KEY"] = "preflight-placeholder"
        env_base["SERVING_API_KEY"] = "preflight-placeholder"
        trainer_env = build_trainer_env(
            base=base,
            work=hydra_work,
            ckpt_dir=hydra_work / "checkpoints",
            raytmp=raytmp,
            tutor_gpus=[0, 1, 2, 3],
            all_gpus=[0, 1, 2, 3, 4, 5, 6, 7],
            student_gpus=[4, 6],
            layout={"student_concurrency": 32, "judge_concurrency": args.judge_concurrency},
            repo=repo,
            tutor_model=args.tutor_model,
            student_api_base="http://127.0.0.1:18861/v1",
            judge_api_base="http://127.0.0.1:18862/v1",
            start_step=0,
            total_steps=100,
            save_freq=5,
            test_freq=25,
            max_ckpt_keep=1,
            experiment="routeA-t026-preflight",
            ckpt_hint=STAGE3_CKPT_HINT_BYTES,
        )
        trainer_env.update(
            {
                "PRO5000_REWARD_MODE": "ladder_rasch",
                "PRO5000_DATA_DIR": str(data_dir),
                "PRO5000_THETA_SCALE": "0.25",
                "PRO5000_LADDER_CLIMBS": "8",
                "PRO5000_GRADER_CONC": str(args.grader_max_concurrency),
                "ROUTEA_STEP_METRICS_DIR": str(hydra_work / "routeA-metrics"),
            }
        )
        overrides = pcfg.build_overrides(
            base=base,
            work=hydra_work,
            ckpt_dir=hydra_work / "checkpoints",
            raytmp=raytmp,
            student_name="Llama-3.1-8B-Instruct-preflight",
            interaction=interaction,
            total_steps=100,
            save_freq=5,
            test_freq=25,
            tutor_gpus=4,
            optimizer_offload=True,
            student_concurrency=32,
            judge_concurrency=args.judge_concurrency,
            max_actor_ckpt_to_keep=1,
            reward_mode="ladder_rasch",
            data_dir=data_dir,
            theta_scale=0.25,
            ladder_num_climbs=8,
            grader_max_concurrency=args.grader_max_concurrency,
            code_root=CODE_ROOT,
        )
        pcfg.run_preflight(
            python_exe=py_exe,
            trainer_entry=PRO5000_DIR / "pro5000_trainer_entry.py",
            repo=repo,
            work=hydra_work,
            tutor_model=args.tutor_model,
            student_api_base="http://127.0.0.1:18861/v1",
            judge_api_base="http://127.0.0.1:18862/v1",
            overrides=overrides,
            check_env=dict(env_base, **trainer_env),
            code_root=CODE_ROOT,
        )
        print("[preflight] Hydra Route A --check-only passed: PRO5000_PRECHECK_OK", flush=True)
    finally:
        shutil.rmtree(raytmp, ignore_errors=True)
        shutil.rmtree(hydra_work, ignore_errors=True)

    tests_report = run_all_unittests(py_exe, repo)
    print(f"[preflight] All {tests_report['passed_files']} unit test suites passed", flush=True)

    t017_baseline = extract_t017_control(args.t017_run_dir)
    t017_json = run_dir / "t017_control_baseline.json"
    t017_md = run_dir / "t017_control_table.md"
    t017_json.write_text(json.dumps(t017_baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    t017_md.write_text(format_t017_markdown_table(t017_baseline), encoding="utf-8")

    summary = {
        "status": "STAGE1_PREFLIGHT_DONE",
        "rc": 0,
        "packages": pkg_versions,
        "hybrid_grader_sha256": hg_sha,
        "t023_jsonl_sha256": t023_sha,
        "eduardo_commit": pcfg.AUTHOR_COMMIT,
        "disk_free_gb": round(free_bytes / 1e9, 2),
        "gpus": gpu_summary,
        "models": {
            "tutor": tutor_info,
            "student": student_info,
            "judge": judge_info,
        },
        "dataset_summary": dataset_summary,
        "hydra_preflight": "PRO5000_PRECHECK_OK",
        "unittests": tests_report,
        "t017_control_summary": {
            "n_steps": t017_baseline["n_steps"],
            "available_reward_related_keys": t017_baseline["available_reward_related_keys"],
            "per_item_rewards": t017_baseline["per_item_rewards"],
            "summary": t017_baseline["summary"],
        },
    }
    (run_dir / "stage1_preflight_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("STAGE1_PREFLIGHT_DONE=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
