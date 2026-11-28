#!/usr/bin/env python3
"""Read-only CPU preflight for T-033 (Route A Step-100 on Pro 5000).

Verifies:
  1. Grading packages: math-verify==0.8.0, latex2sympy2_extended==1.10.2
  2. Route A Step-100 HF bf16 export (7 files) vs T-026 machine-written manifest
  3. Frozen module hashes (agnes_judge.py, hybrid_grader.py, paired_gain.py)
  4. Upstream EduardoRL@e8c89b7
  5. Test datasets & training parquets (author494 test.parquet, t021_bundle, t023 JSONL,
     T-026 train.parquet & val.parquet)
  6. Local Student (Llama-3.1-8B-Instruct) & Grader (Qwen3.6-27B) in /dev/shm
  7. Disk space, idle Pro 5000 GPUs (2-snap check), Agnes key presence, and unit tests.
"""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
AGROUP_SB_DIR = REPO_ROOT / "evaluation" / "agroup_selfbuilt_pro5000"
BASE4B_DIR = REPO_ROOT / "evaluation" / "base4b_selfbuilt"
for _p in (str(HERE), str(AGROUP_SB_DIR), str(BASE4B_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (  # noqa: E402
    AGNES_JUDGE_SHA256,
    AUTHOR494_PARQUET_SHA256,
    AUTHOR_COMMIT,
    DEFAULT_BASE,
    DEFAULT_ROUTEA_EXPORT_REL,
    DEFAULT_ROUTEA_SHA_REL,
    HYBRID_GRADER_SHA256,
    PAIRED_GAIN_SHA256,
    ROUTEA_STEP100_LOCK,
    STUDENT_LLAMA8B_LOCK,
    T023_JSONL_SHA256,
    T026_TRAIN_PARQUET_SHA256,
    T026_VAL_PARQUET_SHA256,
    disk_free_gb,
    format_machine_summary,
    locate_eduardo_repo,
    locate_file_by_sha256,
    resolve_local_model,
    select_free_gpus,
    sha256_file,
    snapshot_gpus,
    verify_dir_against_lock,
    verify_tutor_bfloat16,
    write_json_atomic,
)
from import_bundle import load_and_verify_bundle  # noqa: E402


def verify_packages() -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {}
    for pkg in (
        "math-verify",
        "latex2sympy2-extended",
        "latex2sympy2",
        "vllm",
        "torch",
        "transformers",
        "openai",
        "pandas",
        "pyarrow",
    ):
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = None

    if versions["math-verify"] != "0.8.0":
        raise RuntimeError(f"math-verify mismatch: actual={versions['math-verify']} expected=0.8.0")
    if versions["latex2sympy2-extended"] != "1.10.2":
        raise RuntimeError(
            f"latex2sympy2-extended mismatch: actual={versions['latex2sympy2-extended']} expected=1.10.2"
        )
    return versions


def verify_frozen_modules(repo_root: Path) -> dict[str, str]:
    checks = {
        "agnes_judge.py": (repo_root / "evaluation" / "agnes_eval_v3" / "agnes_judge.py", AGNES_JUDGE_SHA256),
        "hybrid_grader.py": (repo_root / "training" / "routeA_reward" / "hybrid_grader.py", HYBRID_GRADER_SHA256),
        "paired_gain.py": (repo_root / "evaluation" / "paired_gain" / "paired_gain.py", PAIRED_GAIN_SHA256),
    }
    out: dict[str, str] = {}
    for name, (fpath, exp) in checks.items():
        if not fpath.is_file() or fpath.is_symlink():
            raise RuntimeError(f"Frozen module missing or symlinked: {fpath}")
        got = sha256_file(fpath)
        if got != exp:
            raise RuntimeError(f"Frozen module {name} SHA-256 mismatch: actual={got} expected={exp}")
        out[name] = got
    return out


def verify_reference_artifacts(base: Path) -> dict[str, Any]:
    runs = base / "runs"
    info: dict[str, Any] = {}

    t018_csv = locate_file_by_sha256([base / "t018_bundle"], "t018_per_problem.csv", "") if False else None
    t018_cand = base / "t018_bundle" / "t018_per_problem.csv"
    info["t018_per_problem_csv"] = str(t018_cand) if t018_cand.is_file() else None

    t020_jsons = sorted(runs.glob("eval_author494_*full*/**/transfer_eval_*.json")) if runs.is_dir() else []
    info["t020_agroup_json"] = str(t020_jsons[-1]) if t020_jsons else None

    t021_dirs = sorted(runs.glob("t021_stage2_full_*")) if runs.is_dir() else []
    info["t021_agroup_dir"] = str(t021_dirs[-1]) if t021_dirs else None

    t024_dirs = sorted(runs.glob("t024_stage2_*")) if runs.is_dir() else []
    info["t024_dev_dir"] = str(t024_dirs[-1]) if t024_dirs else None

    _ = t018_csv
    return info


def run_preflight(
    *,
    base: Path,
    repo_root: Path,
    routea_export_dir: Path,
    routea_sha_file: Path,
    author494_parquet: Path,
    selfbuilt_bundle_dir: Path,
    t023_jsonl: Path,
    train_parquet: Optional[Path] = None,
    val_parquet: Optional[Path] = None,
    agnes_key_file: Path = Path.home() / ".secrets" / "agnes_api_key",
    skip_tests: bool = False,
) -> dict[str, Any]:
    versions = verify_packages()
    frozen = verify_frozen_modules(repo_root)

    eduardo_repo = locate_eduardo_repo(base)
    rev = subprocess.check_output(
        ["git", "-C", str(eduardo_repo), "rev-parse", "HEAD"],
        text=True,
        timeout=30,
    ).strip()
    if rev != AUTHOR_COMMIT:
        raise RuntimeError(f"EduardoRL commit mismatch: actual={rev} expected={AUTHOR_COMMIT}")

    # Step-100 HF bf16 export + manifest check.
    step100_verified = verify_dir_against_lock(
        routea_export_dir,
        ROUTEA_STEP100_LOCK,
        label="RouteA Step-100",
        manifest_file=routea_sha_file,
    )
    tutor_dtype = verify_tutor_bfloat16(routea_export_dir)

    # Local Student & Grader models.
    student_model = resolve_local_model(base, "Meta-Llama-3.1-8B-Instruct", ("llama-3.1-8b", "8b-instruct"))
    verify_dir_against_lock(student_model, STUDENT_LLAMA8B_LOCK, label="Student Llama-3.1-8B")
    grader_model = resolve_local_model(base, "Qwen3.6-27B", ("qwen3.6-27b", "27b"))
    if not (grader_model / "config.json").is_file():
        raise RuntimeError(f"Grader 27B config.json missing: {grader_model}")

    # Test sets & T-023 / T-026 datasets.
    if not author494_parquet.is_file():
        raise RuntimeError(f"Author 494 test.parquet missing: {author494_parquet}")
    a494_sha = sha256_file(author494_parquet)
    if a494_sha != AUTHOR494_PARQUET_SHA256:
        raise RuntimeError(f"Author 494 test.parquet SHA-256 mismatch: actual={a494_sha}")

    bundle = load_and_verify_bundle(selfbuilt_bundle_dir, require_exact_totals=True)

    if not t023_jsonl.is_file():
        raise RuntimeError(f"T-023 JSONL missing: {t023_jsonl}")
    t023_sha = sha256_file(t023_jsonl)
    if t023_sha != T023_JSONL_SHA256:
        raise RuntimeError(f"T-023 JSONL SHA-256 mismatch: actual={t023_sha}")

    search_roots = [base / "runs" / "t026_stage3_formal_20261126_140508", base / "runs", base / "data"]
    resolved_train = train_parquet or locate_file_by_sha256(search_roots, "train.parquet", T026_TRAIN_PARQUET_SHA256)
    resolved_val = val_parquet or locate_file_by_sha256(search_roots, "val.parquet", T026_VAL_PARQUET_SHA256)
    if resolved_train is None or not resolved_train.is_file() or sha256_file(resolved_train) != T026_TRAIN_PARQUET_SHA256:
        raise RuntimeError("Could not locate verified T-026 train.parquet (SHA-256 1e85fd0f...)")
    if resolved_val is None or not resolved_val.is_file() or sha256_file(resolved_val) != T026_VAL_PARQUET_SHA256:
        raise RuntimeError("Could not locate verified T-026 val.parquet (SHA-256 9a6808ea...)")

    # Secrets / env hygiene.
    if os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY must NOT be set")
    agnes_key_ok = bool(os.environ.get("AGNES_API_KEY")) or (agnes_key_file.is_file() and agnes_key_file.stat().st_size > 0)
    if not agnes_key_ok:
        raise RuntimeError(f"Agnes API key missing (env AGNES_API_KEY and {agnes_key_file})")

    # Disk & 2-snap GPUs.
    data_free_gb = disk_free_gb(Path("/data") if Path("/data").exists() else base)
    if data_free_gb < 15.0:
        raise RuntimeError(f"Low disk free space: {data_free_gb} GB < 15.0 GB")
    s1 = snapshot_gpus()
    time.sleep(2)
    s2 = snapshot_gpus()
    free_gpus = select_free_gpus(s1, s2)

    unittest_summary = "skipped"
    if not skip_tests:
        test_env = dict(os.environ, PYTHONPATH=f"{repo_root}:{eduardo_repo}", PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run(
            [sys.executable, "-B", "-m", "unittest", "discover", "-s", str(HERE), "-p", "test_*.py", "-v"],
            env=test_env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        combined = "\n".join(p for p in (proc.stdout, proc.stderr) if p)
        tail = [ln for ln in combined.splitlines() if ln.strip()][-6:]
        if proc.returncode != 0:
            raise RuntimeError(f"routeA_t033 unit tests failed rc={proc.returncode}:\n" + "\n".join(tail))
        if "skipped=" in "\n".join(tail).lower():
            raise RuntimeError("routeA_t033 unit tests must not skip inside pro5000-fast:\n" + "\n".join(tail))
        unittest_summary = " | ".join(tail[-2:])

    return {
        "status": "PREFLIGHT_OK",
        "rc": 0,
        "packages": versions,
        "frozen_modules": frozen,
        "author_commit": rev,
        "routea_export_dir": str(routea_export_dir),
        "routea_tutor_dtype": tutor_dtype,
        "routea_step100_files": len(step100_verified),
        "student_model_dir": str(student_model),
        "grader_model_dir": str(grader_model),
        "author494_parquet": str(author494_parquet),
        "author494_sha256": a494_sha,
        "selfbuilt_eligible": bundle["eligible_count"],
        "selfbuilt_skipped": bundle["skipped_count"],
        "t023_jsonl_sha256": t023_sha,
        "t026_train_parquet": str(resolved_train),
        "t026_train_sha256": T026_TRAIN_PARQUET_SHA256,
        "t026_val_parquet": str(resolved_val),
        "t026_val_sha256": T026_VAL_PARQUET_SHA256,
        "references": verify_reference_artifacts(base),
        "data_free_gb": data_free_gb,
        "idle_gpus": free_gpus,
        "unittest_summary": unittest_summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="CPU preflight for T-033 Route A Step-100 eval.")
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--repo", type=Path, default=REPO_ROOT)
    ap.add_argument("--routea-export-dir", type=Path, default=None)
    ap.add_argument("--routea-sha-file", type=Path, default=None)
    ap.add_argument("--author494-parquet", type=Path, default=None)
    ap.add_argument("--selfbuilt-bundle-dir", type=Path, default=None)
    ap.add_argument("--t023-jsonl", type=Path, default=None)
    ap.add_argument("--train-parquet", type=Path, default=None)
    ap.add_argument("--val-parquet", type=Path, default=None)
    ap.add_argument("--agnes-key-file", type=Path, default=Path.home() / ".secrets" / "agnes_api_key")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    base = args.base.resolve()
    res = run_preflight(
        base=base,
        repo_root=args.repo.resolve(),
        routea_export_dir=args.routea_export_dir or (base / DEFAULT_ROUTEA_EXPORT_REL),
        routea_sha_file=args.routea_sha_file or (base / DEFAULT_ROUTEA_SHA_REL),
        author494_parquet=args.author494_parquet or (base / "data" / "author494_eval" / "test.parquet"),
        selfbuilt_bundle_dir=args.selfbuilt_bundle_dir or (base / "t021_bundle"),
        t023_jsonl=args.t023_jsonl or (base / "runs" / "t023_stage2_20261125_163319" / "routeA_stage2_measured.jsonl"),
        train_parquet=args.train_parquet,
        val_parquet=args.val_parquet,
        agnes_key_file=args.agnes_key_file,
        skip_tests=args.skip_tests,
    )
    if args.out_json is not None:
        write_json_atomic(args.out_json, res)
    print(json.dumps(res, ensure_ascii=False, indent=2), flush=True)
    print(
        format_machine_summary(
            {
                "status": res["status"],
                "rc": res["rc"],
                "calls": 0,
                "failures": 0,
                "step100_files": res["routea_step100_files"],
                "math_verify": res["packages"]["math-verify"],
                "idle_gpus": len(res["idle_gpus"]),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
