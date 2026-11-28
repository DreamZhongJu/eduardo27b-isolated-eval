"""Pro 5000 card-free Stage 1 preflight for T-034 Route C training.

Checks (CPU-only, CUDA_VISIBLE_DEVICES="", does NOT require idle GPUs):
  1. math-verify==0.8.0 and latex2sympy2_extended==1.10.2 versions
  2. SHA-256 of hybrid_grader.py (7d2a5f46...) in both training/routeA_reward and routeA_data,
     EduardoRL@e8c89b7 clean commit, kernel_equivalence.json, /dev/shm Student & 27B models,
     Tutor Qwen3.5-4B
  3. SHA-256 of T-023 measured JSONL (9ff8449b...), T-031 source pretest JSONL (a031276f...),
     and T-026 train.parquet (1e85fd0f...) / val.parquet (9a6808ea...)
  4. Generate Route C train/val parquets into $BASE/data/routeC_t034, checking T-031 counts
     (train 289/801/805, val 15/46/39), variant field completeness, and T-026 order/fallback parity
  5. Inspect T-026 formal run logs (read-only) to check if Step 1-3 source_ids were logged
  6. Run Hydra --check-only via pro5000_trainer_entry.py pointing at $BASE/data/routeC_t034
  7. Run all 6 unit test suites inside pro5000-fast (real math-verify==0.8.0, zero skips)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
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
    write_interaction_yaml,
)
from stage3_formal import STAGE3_CKPT_HINT_BYTES
from training.routeA_reward.hybrid_grader import verify_hybrid_grader_sha256
from training.routeA_train.preflight_stage1 import (
    EXPECTED_HYBRID_GRADER_SHA256,
    verify_grading_packages,
    verify_model_shards_and_sha,
)
from training.routeA_train.prepare_parquet import EXPECTED_T023_JSONL_SHA256, sha256_file
from training.routeC_train.prepare_parquet import (
    DEFAULT_T023_JSONL,
    DEFAULT_T026_TRAIN_PARQUET,
    DEFAULT_T026_VAL_PARQUET,
    DEFAULT_T031_SOURCE_PRETEST,
    EXPECTED_T026_TRAIN_PARQUET_SHA256,
    EXPECTED_T026_VAL_PARQUET_SHA256,
    EXPECTED_T031_SOURCE_PRETEST_SHA256,
    prepare_routec_parquets,
)

UNITTEST_MODULES = (
    "training/routeA_reward/test_routeA_reward.py",
    "training/routeA_reward/test_stage2_dev_sim.py",
    "training/a_group_pro5000/test_stage3_formal.py",
    "training/a_group_pro5000/test_stage2_pilot.py",
    "training/routeA_train/test_routeA_train.py",
    "training/routeC_train/test_routeC_train.py",
)


def _resolve_existing(primary: Path, fallback: Path) -> Path:
    if primary.is_file():
        return primary
    return fallback


def check_t026_step1_3_source_ids(base: Path) -> str:
    """Read-only check whether T-026 formal run logs contain per-step source_ids."""
    runs_dir = base / "runs"
    t026_dirs = sorted(runs_dir.glob("t026_stage3_formal_*")) if runs_dir.is_dir() else []
    if not t026_dirs:
        return "unverified_t026_run_dir_not_found"
    latest = t026_dirs[-1]
    for s in (1, 2, 3):
        sp = latest / "step-statistics" / f"{s}.json"
        if not sp.is_file():
            return "unverified_missing_step_statistics"
        try:
            text = sp.read_text(encoding="utf-8")
        except OSError:
            return "unverified_unreadable_step_statistics"
        if "source_id" in text:
            return "present_in_step_statistics"
    return "unverified_no_source_id_in_t026_logs"


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
    parser.add_argument("--t023-jsonl", type=Path, default=None)
    parser.add_argument("--t031-source-pretest", type=Path, default=None)
    parser.add_argument("--t026-train-parquet", type=Path, default=None)
    parser.add_argument("--t026-val-parquet", type=Path, default=None)
    parser.add_argument("--tutor-model", type=Path, default=pcfg.TEACHER_MODEL_DEFAULT)
    parser.add_argument("--judge-concurrency", type=int, default=16)
    parser.add_argument("--grader-max-concurrency", type=int, default=32)
    args = parser.parse_args()
    os.umask(0o077)

    base = args.base.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    py_exe = Path(sys.executable).resolve()

    t023_jsonl = args.t023_jsonl or _resolve_existing(
        base / "runs" / "t023_stage2_20261125_163319" / "routeA_stage2_measured.jsonl",
        DEFAULT_T023_JSONL,
    )
    t031_pretest = args.t031_source_pretest or _resolve_existing(
        base / "runs" / "t031_stage2_20261128_095549" / "source_pretest_measured.jsonl",
        DEFAULT_T031_SOURCE_PRETEST,
    )
    t026_train = args.t026_train_parquet or _resolve_existing(
        base / "data" / "routeA_t026" / "train.parquet",
        DEFAULT_T026_TRAIN_PARQUET,
    )
    t026_val = args.t026_val_parquet or _resolve_existing(
        base / "data" / "routeA_t026" / "val.parquet",
        DEFAULT_T026_VAL_PARQUET,
    )

    pkg_versions = verify_grading_packages()
    print(f"[preflight] packages={pkg_versions}", flush=True)

    hg_reward = verify_hybrid_grader_sha256(
        CODE_ROOT / "training" / "routeA_reward" / "hybrid_grader.py",
        EXPECTED_HYBRID_GRADER_SHA256,
    )
    hg_data = verify_hybrid_grader_sha256(
        CODE_ROOT / "routeA_data" / "hybrid_grader.py",
        EXPECTED_HYBRID_GRADER_SHA256,
    )
    print(f"[preflight] hybrid_grader.py sha256={hg_reward} (routeA_data={hg_data})", flush=True)

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
        raise RuntimeError(
            f"Student/27B 未位于 /dev/shm: student={student_info['resolved']} judge={judge_info['resolved']}"
        )
    print(
        f"[preflight] models verified: tutor_shards={tutor_info['num_shards']} "
        f"student_shm={student_info['in_dev_shm']} ({student_info['num_shards']} shards, manifest={student_info['manifest_checked']}) "
        f"judge_shm={judge_info['in_dev_shm']} ({judge_info['num_shards']} shards, manifest={judge_info['manifest_checked']})",
        flush=True,
    )

    free_bytes = shutil.disk_usage("/data").free
    disk_free_gb = round(free_bytes / 1e9, 2)
    print(f"[preflight] /data free={disk_free_gb:.2f} GB (informational only in Stage 1)", flush=True)

    data_dir = base / "data" / "routeC_t034"
    sys_path_added = False
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
        sys_path_added = True
    try:
        dataset_summary = prepare_routec_parquets(
            input_jsonl=t023_jsonl,
            source_pretest_jsonl=t031_pretest,
            out_dir=data_dir,
            t026_train_parquet=t026_train,
            t026_val_parquet=t026_val,
            expected_t023_sha256=EXPECTED_T023_JSONL_SHA256,
            expected_source_pretest_sha256=EXPECTED_T031_SOURCE_PRETEST_SHA256,
            expected_t026_train_sha256=EXPECTED_T026_TRAIN_PARQUET_SHA256,
            expected_t026_val_sha256=EXPECTED_T026_VAL_PARQUET_SHA256,
            enforce_t031_counts=True,
        )
    finally:
        if sys_path_added and str(repo) in sys.path:
            sys.path.remove(str(repo))

    tr_bc = dataset_summary["train"]["branch_counts"]
    va_bc = dataset_summary["val"]["branch_counts"]
    tr_hl = dataset_summary["train"]["hard_posttest_levels"]
    tr_el = dataset_summary["train"]["easy_posttest_levels"]
    va_hl = dataset_summary["val"]["hard_posttest_levels"]
    va_el = dataset_summary["val"]["easy_posttest_levels"]
    print(
        f"[preflight] routeC parquets ready: train=({tr_bc['easy']}/{tr_bc['hard']}/{tr_bc['fallback']}) "
        f"val=({va_bc['easy']}/{va_bc['hard']}/{va_bc['fallback']}) "
        f"hard_levels=({tr_hl['1']}/{tr_hl['2']}/{tr_hl['3+']}) "
        f"easy_levels=({tr_el['1']}/{tr_el['2']}/{tr_el['3+']})",
        flush=True,
    )

    t026_step_order_check = check_t026_step1_3_source_ids(base)

    # Hydra --check-only with Route C parquet paths and T-026 frozen recipe
    hydra_work = Path(tempfile.mkdtemp(dir=base / "tmp", prefix=".t034_hydra_"))
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
            experiment="routeC-t034-preflight",
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
        print("[preflight] Hydra Route C --check-only passed: PRO5000_PRECHECK_OK", flush=True)
    finally:
        shutil.rmtree(raytmp, ignore_errors=True)
        shutil.rmtree(hydra_work, ignore_errors=True)

    tests_report = run_all_unittests(py_exe, repo)
    print(f"[preflight] All {tests_report['passed_files']} unit test suites passed", flush=True)

    train_sha = dataset_summary["files"]["train.parquet"]["sha256"]
    val_sha = dataset_summary["files"]["val.parquet"]["sha256"]
    t026_order_ok = (
        dataset_summary["t026_parity"]["train"]["order_match"]
        and dataset_summary["t026_parity"]["val"]["order_match"]
    )
    t026_fb_ok = (
        dataset_summary["t026_parity"]["train"]["fallback_exact_match"]
        and dataset_summary["t026_parity"]["val"]["fallback_exact_match"]
    )

    summary = {
        "status": "STAGE1_PREFLIGHT_DONE",
        "rc": 0,
        "packages": pkg_versions,
        "hybrid_grader_sha256": hg_reward,
        "t023_jsonl_sha256": dataset_summary["input_jsonl_sha256"],
        "t031_source_pretest_sha256": dataset_summary["source_pretest_sha256"],
        "eduardo_commit": pcfg.AUTHOR_COMMIT,
        "disk_free_gb": disk_free_gb,
        "models": {
            "tutor": tutor_info,
            "student": student_info,
            "judge": judge_info,
        },
        "dataset_summary": dataset_summary,
        "t026_step1_3_source_ids": t026_step_order_check,
        "hydra_preflight": "PRO5000_PRECHECK_OK",
        "unittests": tests_report,
    }
    (run_dir / "stage1_preflight_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("")
    print("status: STAGE1_PREFLIGHT_DONE")
    print("rc: 0")
    print("calls: 0")
    print("failures: 0")
    print(f"train_easy: {tr_bc['easy']}")
    print(f"train_hard: {tr_bc['hard']}")
    print(f"train_fallback: {tr_bc['fallback']}")
    print(f"val_easy: {va_bc['easy']}")
    print(f"val_hard: {va_bc['hard']}")
    print(f"val_fallback: {va_bc['fallback']}")
    print(f"hard_ladder_train: {tr_hl['1']}/{tr_hl['2']}/{tr_hl['3+']}")
    print(f"easy_ladder_train: {tr_el['1']}/{tr_el['2']}/{tr_el['3+']}")
    print(f"hard_ladder_val: {va_hl['1']}/{va_hl['2']}/{va_hl['3+']}")
    print(f"easy_ladder_val: {va_el['1']}/{va_el['2']}/{va_el['3+']}")
    print(f"t026_order_match: {t026_order_ok}")
    print(f"t026_fallback_exact_match: {t026_fb_ok}")
    print(f"t026_step1_3_source_ids: {t026_step_order_check}")
    print("missing_variant_fields: 0")
    print("hydra_preflight: PRO5000_PRECHECK_OK")
    print(f"unittests_passed: {tests_report['passed_files']}/{len(UNITTEST_MODULES)}")
    print(f"disk_free_gb: {disk_free_gb}")
    print(f"train_parquet_sha256: {train_sha}")
    print(f"val_parquet_sha256: {val_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
