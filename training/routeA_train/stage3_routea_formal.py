"""Stage 3 (formal 100-step Route A DPPO run with resume, anti-hacking gates, and Step-100 HF export) for Pro 5000."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent.parent
PRO5000_DIR = CODE_ROOT / "training" / "a_group_pro5000"
for p in (CODE_ROOT, PRO5000_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pro5000_config as pcfg
from pro5000_trainer_entry import T017_VAL_GATED_REF, verify_model_extra_checkpoint, write_json
from stage1_collect import find_local_model
from stage2_pilot import (
    base_subprocess_env,
    build_trainer_env,
    default_base,
    ensure_eduardo_repo,
    make_ray_tmp,
    pick_free_ports,
    terminate_pg,
    terminate_procs,
    verify_model_dir,
    wait_for_http_health,
    write_access_log_config,
    write_interaction_yaml,
)
from stage2_precheck import snapshot_gpus
from stage3_formal import (
    STAGE3_CKPT_HINT_BYTES,
    list_verified_checkpoints,
    parse_step_num,
    prepare_resume_checkpoint,
    verify_stage3_gpus_clean,
    export_hf_step,
)
from telemetry import Sampler
from training.routeA_train.prepare_parquet import sha256_file
from training.routeA_train.stage2_routea_pilot import verify_routea_prereqs

STAGE3_FRESH_DISK_FLOOR_BYTES = 70 * 10**9
STAGE3_RESUME_DISK_FLOOR_BYTES = 45 * 10**9
DEFAULT_TOTAL_STEPS = 100
DEFAULT_SAVE_FREQ = 5
DEFAULT_TEST_FREQ = 25
DEFAULT_KEEP_CKPTS = 1
DEFAULT_STUDENT_CONCURRENCY = 32
DEFAULT_JUDGE_CONCURRENCY = 16
DEFAULT_GRADER_CONCURRENCY = 32
DEFAULT_THETA_SCALE = 0.25
DEFAULT_LADDER_CLIMBS = 8


def check_stage3_disk_floor(free_bytes: int, is_resume: bool) -> tuple[bool, int]:
    """Return (ok, required_floor_bytes) for fresh (>=70 GB) vs resume (>=45 GB)."""
    floor = STAGE3_RESUME_DISK_FLOOR_BYTES if is_resume else STAGE3_FRESH_DISK_FLOOR_BYTES
    return free_bytes >= floor, floor


def find_latest_routea_stage3_run(runs_dir: Path) -> Path:
    if not runs_dir.is_dir():
        raise RuntimeError(f"找不到运行目录: {runs_dir}")
    candidates = sorted(
        (
            p
            for p in runs_dir.glob("t026_stage3_formal_*")
            if p.is_dir()
            and not p.is_symlink()
            and ((p / "run_meta.json").is_file() or any((p / "checkpoints").glob("global_step_*")))
        ),
        key=lambda p: p.name,
    )
    if not candidates:
        raise RuntimeError(f"找不到可续训的路线 A 阶段 3 历史运行目录 (t026_stage3_formal_*): {runs_dir}")
    return candidates[-1]


def _load_numbered_json_dir(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not directory.is_dir():
        return rows
    numbered: list[tuple[int, Path]] = []
    for p in directory.glob("*.json"):
        if p.stem.isascii() and p.stem.isdigit():
            numbered.append((int(p.stem), p))
    for _, p in sorted(numbered):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except ValueError:
            pass
    return rows


def write_export_sha256_manifest(export_dir: Path, out_path: Path) -> dict[str, Any]:
    """Write standard sha256sum manifest for Step-100 HF export directory."""
    if not (export_dir / "config.json").is_file():
        raise RuntimeError(f"导出目录缺少 config.json: {export_dir}")
    shards = [p for p in export_dir.glob("*.safetensors") if p.is_file() and p.stat().st_size > 0]
    if not shards:
        raise RuntimeError(f"导出目录缺少有效 *.safetensors: {export_dir}")

    files_map: dict[str, str] = {}
    lines: list[str] = []
    all_files = sorted(
        (p for p in export_dir.rglob("*") if p.is_file() and not p.is_symlink()),
        key=lambda p: p.relative_to(export_dir).as_posix(),
    )
    for fpath in all_files:
        rel = fpath.relative_to(export_dir).as_posix()
        digest = sha256_file(fpath)
        files_map[rel] = digest
        lines.append(f"{digest}  {rel}\n")

    out_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}")
    tmp_path.write_text("".join(lines), encoding="utf-8")
    os.replace(tmp_path, out_path)
    return {
        "manifest_path": str(out_path),
        "file_count": len(all_files),
        "files": files_map,
    }


def _extract_routea_brief(row: dict[str, Any]) -> dict[str, Any]:
    ram = row.get("routeA_metrics", {}).get("metrics", {})
    rgates = row.get("routeA_gates", {})
    pt = row.get("phase_timings", {})
    return {
        "step": row.get("step"),
        "step_seconds": row.get("metrics", {}).get("timing_s/step"),
        "gen_minutes": pt.get("gen", {}).get("minutes"),
        "reward_mean": ram.get("eduardo/routeA_reward_mean"),
        "theta_mean": ram.get("eduardo/routeA_theta_mean"),
        "gated_rate": ram.get("eduardo/routeA_gated_rate"),
        "gated_ma5": rgates.get("gated_ma5"),
        "tech_fail_rate": ram.get("eduardo/routeA_grader_tech_failed_rate"),
        "gates_passed": rgates.get("passed"),
        "t017_val_gated_ref": rgates.get("t017_val_gated_ref")
        if "t017_val_gated_ref" in rgates
        else T017_VAL_GATED_REF.get(row.get("step")),
    }


def summarize_stage3_routea(
    work: Path,
    layout: dict[str, Any],
    total_steps: int,
    final_info: dict[str, Any],
    export_info: dict[str, Any],
    export_manifest: dict[str, Any],
    ckpt_kept: list[int],
    shm_preserved: bool = True,
) -> dict[str, Any]:
    """Assemble and write stage3_summary.json from completed step & validation logs."""
    rows = _load_numbered_json_dir(work / "step-statistics")
    val_rows = _load_numbered_json_dir(work / "val-statistics")
    val_map: dict[str, Any] = {}
    for vr in val_rows:
        s = vr.get("step")
        if s is not None:
            vm = vr.get("metrics", {})
            val_map[str(s)] = {
                "n_rollouts": vr.get("n_rollouts", 0),
                "reward_mean": vm.get("eduardo/routeA_reward_mean"),
                "theta_mean": vm.get("eduardo/routeA_theta_mean"),
                "gated_rate": vm.get("eduardo/routeA_gated_rate"),
                "t017_val_gated_ref": vr.get("t017_val_gated_ref"),
            }

    step_secs = [
        float(r["metrics"]["timing_s/step"])
        for r in rows
        if isinstance(r.get("metrics", {}).get("timing_s/step"), (int, float))
    ]
    avg_step_s = round(sum(step_secs) / len(step_secs), 2) if step_secs else 0.0
    total_hours = round(sum(step_secs) / 3600.0, 2) if step_secs else 0.0

    by_step = {int(r["step"]): r for r in rows if isinstance(r.get("step"), int)}
    milestone_steps = [s for s in (1, 25, 50, 75, total_steps) if s in by_step]
    if rows and rows[-1].get("step") not in milestone_steps:
        milestone_steps.append(rows[-1]["step"])
    milestones = [_extract_routea_brief(by_step[s]) for s in milestone_steps if s in by_step]

    du_target = Path("/data") if Path("/data").exists() else work
    summary = {
        "status": "STAGE3_FORMAL_DONE",
        "rc": 0,
        "work_dir": str(work),
        "layout": layout,
        "total_steps": total_steps,
        "steps_completed": rows[-1]["step"] if rows else 0,
        "checkpoints_kept": ckpt_kept,
        "final_checkpoint": final_info,
        "hf_export": export_info,
        "export_step100_sha256_file": export_manifest.get("manifest_path"),
        "export_file_count": export_manifest.get("file_count"),
        "val_statistics": val_map,
        "milestones": milestones,
        "avg_step_s": avg_step_s,
        "total_training_hours": total_hours,
        "shm_models_preserved": shm_preserved,
        "post_run_disk_free_gb": round(shutil.disk_usage(du_target).free / 1e9, 2),
    }
    write_json(work / "stage3_summary.json", summary)
    return summary


def format_stage3_routea_status(
    base: Path,
    run_dir: Path | None = None,
    gpu_lines: list[str] | None = None,
    proc_lines: list[str] | None = None,
) -> str:
    work = run_dir.resolve() if run_dir is not None else find_latest_routea_stage3_run(base / "runs")
    meta_file = work / "run_meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else {}
    total_steps = int(meta.get("total_steps", DEFAULT_TOTAL_STEPS))

    rows = _load_numbered_json_dir(work / "step-statistics")
    val_rows = _load_numbered_json_dir(work / "val-statistics")
    latest_step = rows[-1]["step"] if rows else 0
    recent = rows[-5:]

    steady = [
        float(r["metrics"]["timing_s/step"])
        for r in rows[1:]
        if isinstance(r.get("metrics", {}).get("timing_s/step"), (int, float))
    ]
    if not steady and rows and isinstance(rows[0].get("metrics", {}).get("timing_s/step"), (int, float)):
        steady = [float(rows[0]["metrics"]["timing_s/step"])]
    eta_str = "N/A"
    if steady:
        avg_s = sum(steady) / len(steady)
        rem_steps = max(0, total_steps - latest_step)
        eta_h = rem_steps * avg_s / 3600.0
        eta_str = f"{eta_h:.2f}h (均步 {avg_s:.1f}s, 余 {rem_steps} 步)"

    ckpt_steps = []
    for p in work.glob("ckpt-verified-*.json"):
        s = parse_step_num(p.stem, "ckpt-verified-")
        if s is not None:
            ckpt_steps.append(s)
    latest_ckpt = f"global_step_{max(ckpt_steps)}" if ckpt_steps else "无"

    du_target = Path("/data") if Path("/data").exists() else base
    du_data = shutil.disk_usage(du_target)
    lines = [
        f"=== 路线 A 阶段 3 进度 ({work.name}) ===",
        f"进程状态: {' | '.join(proc_lines) if proc_lines else '无运行中 stage3_routea 进程'}",
        f"最新步号: {latest_step} / {total_steps} | 最近验证存盘: {latest_ckpt} | 预计剩余: {eta_str}",
        f"/data 剩余: {du_data.free / 1e9:.2f} GB ({du_data.free / 1024**3:.2f} GiB)",
    ]
    if (work / "step2-comparison.json").is_file():
        c2 = json.loads((work / "step2-comparison.json").read_text(encoding="utf-8"))
        lines.append(f"第2/1步比值: passed={c2.get('passed')} ratios={c2.get('ratios')}")
    if recent:
        lines.append("最近几步训练指标:")
        for r in recent:
            pt = r.get("phase_timings", {})
            sec = r.get("metrics", {}).get("timing_s/step")
            ram = r.get("routeA_metrics", {}).get("metrics", {})
            rg = r.get("routeA_gates", {})
            ma5 = rg.get("gated_ma5")
            ma5_s = f"{float(ma5):.3f}" if isinstance(ma5, (int, float)) else "N/A"
            sec_s = f"{sec:.1f}s({sec / 60:.2f}m)" if isinstance(sec, (int, float)) else "N/A"
            lines.append(
                f"  step {r['step']:3d}: {sec_s} | "
                f"gen={pt.get('gen', {}).get('minutes', 0):.2f}m "
                f"old={pt.get('old_log_prob', {}).get('minutes', 0):.2f}m "
                f"upd={pt.get('update_actor', {}).get('minutes', 0):.2f}m | "
                f"reward={ram.get('eduardo/routeA_reward_mean', 0.0):.3f} "
                f"theta={ram.get('eduardo/routeA_theta_mean', 0.0):.3f} "
                f"gated={ram.get('eduardo/routeA_gated_rate', 0.0):.3f} (ma5={ma5_s})"
            )
    if val_rows:
        lines.append("验证集窗口统计 (vs T-017 val gated_rate 参考):")
        for vr in val_rows:
            vm = vr.get("metrics", {})
            ref = vr.get("t017_val_gated_ref")
            ref_s = f"{float(ref):.2f}" if ref is not None else "N/A"
            lines.append(
                f"  val step {vr.get('step')}: n={vr.get('n_rollouts', 0)} "
                f"reward={vm.get('eduardo/routeA_reward_mean', 0.0):.3f} "
                f"theta={vm.get('eduardo/routeA_theta_mean', 0.0):.3f} "
                f"gated={vm.get('eduardo/routeA_gated_rate', 0.0):.3f} (t017_ref={ref_s})"
            )
    if rows:
        last_rg = rows[-1].get("routeA_gates", {})
        if last_rg and not last_rg.get("passed", True):
            lines.append(f"停机门告警 (step {rows[-1].get('step')}): {last_rg.get('failures')}")
    if gpu_lines:
        lines.append("8 卡显存与利用率:")
        lines.extend(f"  GPU {ln}" for ln in gpu_lines)
    return "\n".join(lines)


def _run_status(base: Path, run_dir: Path | None) -> int:
    proc_out: list[str] = []
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", "stage3_routea_formal.py|pro5000_trainer_entry.py"],
            text=True,
            timeout=5,
        )
        for ln in out.splitlines():
            if "--status" not in ln:
                proc_out.append(ln[:100])
    except (OSError, subprocess.CalledProcessError):
        pass
    gpu_out: list[str] = []
    try:
        gpu_csv = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
            text=True,
            timeout=5,
        )
        gpu_out = [ln.strip() for ln in gpu_csv.splitlines() if ln.strip()]
    except (OSError, subprocess.CalledProcessError):
        pass
    print(format_stage3_routea_status(base, run_dir=run_dir, gpu_lines=gpu_out, proc_lines=proc_out))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=default_base())
    parser.add_argument("--tutor-model", type=Path, default=pcfg.TEACHER_MODEL_DEFAULT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS)
    parser.add_argument("--save-freq", type=int, default=DEFAULT_SAVE_FREQ)
    parser.add_argument("--test-freq", type=int, default=DEFAULT_TEST_FREQ)
    parser.add_argument("--keep-ckpts", type=int, default=DEFAULT_KEEP_CKPTS)
    parser.add_argument("--student-concurrency", type=int, default=DEFAULT_STUDENT_CONCURRENCY)
    parser.add_argument("--judge-concurrency", type=int, default=None)
    parser.add_argument("--grader-max-concurrency", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)

    base = args.base.resolve()
    if args.status:
        return _run_status(base, args.run_dir)

    py_exe = Path(sys.executable).resolve()
    data_dir = base / "data" / "routeA_t026"
    prereq_info = verify_routea_prereqs(base, args.tutor_model, data_dir)

    du_target = Path("/data") if Path("/data").exists() else base
    free_bytes = shutil.disk_usage(du_target).free
    is_resume = bool(args.resume)
    disk_ok, floor_bytes = check_stage3_disk_floor(free_bytes, is_resume=is_resume)
    print(
        f"[preflight] /data free = {free_bytes / 1e9:.2f} GB ({free_bytes / 1024**3:.2f} GiB) "
        f"[floor={floor_bytes // 10**9} GB, resume={is_resume}]",
        flush=True,
    )
    if not disk_ok:
        print(f"STOP: /data 剩余低于 {floor_bytes // 10**9}G")
        return 3

    models_root = base / "models"
    if models_root.is_dir():
        for entry in sorted(models_root.iterdir()):
            if entry.is_symlink() and not entry.exists():
                raise RuntimeError(f"models 目录下存在失效软链接（节点若重启过请先恢复 /dev/shm）: {entry}")
    student_model = verify_model_dir(
        "Student",
        find_local_model(models_root, models_root / "Meta-Llama-3.1-8B-Instruct", ("llama-3.1-8b", "8b-instruct")),
    )
    judge_model = verify_model_dir(
        "Judge",
        find_local_model(models_root, models_root / "Qwen3.6-27B", ("qwen3.6-27b", "27b")),
    )
    print(
        f"[preflight] 模型与软链接就绪: student={student_model} -> {student_model.resolve()} ; "
        f"judge={judge_model} -> {judge_model.resolve()}",
        flush=True,
    )

    repo = ensure_eduardo_repo(base)

    if is_resume:
        work = args.run_dir.resolve() if args.run_dir else find_latest_routea_stage3_run(base / "runs")
        ckpt_dir = work / "checkpoints"
        resume_step = prepare_resume_checkpoint(work, ckpt_dir, world_size=4)
        meta_file = work / "run_meta.json"
        prior_meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else {}
        prior_layout = prior_meta.get("layout", {})
        judge_conc = (
            args.judge_concurrency
            if args.judge_concurrency is not None
            else int(prior_layout.get("judge_concurrency", DEFAULT_JUDGE_CONCURRENCY))
        )
        grader_conc = (
            args.grader_max_concurrency
            if args.grader_max_concurrency is not None
            else int(prior_layout.get("grader_max_concurrency", DEFAULT_GRADER_CONCURRENCY))
        )
        print(
            f"[resume] 从目录 {work} 的 global_step_{resume_step} 续训 "
            f"(judge_concurrency={judge_conc}, grader_max_concurrency={grader_conc})",
            flush=True,
        )
        if resume_step >= args.total_steps:
            print(f"[resume] 检查点已达目标步数 {resume_step}/{args.total_steps}，跳过训练直接核对导出", flush=True)
    else:
        if args.check_only and not args.run_dir:
            work = Path(tempfile.mkdtemp(dir=base / "tmp", prefix=".t026_s3_pf_"))
        else:
            run_name = f"t026_stage3_formal_{time.strftime('%Y%m%d_%H%M%S')}"
            work = args.run_dir.resolve() if args.run_dir else (base / "runs" / run_name)
        if work.exists() and any(work.iterdir()):
            raise RuntimeError(f"全新阶段 3 目录已存在非空文件: {work}（如要续训请加 --resume）")
        ckpt_dir = work / "checkpoints"
        resume_step = None
        judge_conc = args.judge_concurrency if args.judge_concurrency is not None else DEFAULT_JUDGE_CONCURRENCY
        grader_conc = (
            args.grader_max_concurrency if args.grader_max_concurrency is not None else DEFAULT_GRADER_CONCURRENCY
        )

    tutor_gpus = [0, 1, 2, 3]
    student_gpus = [4, 6]
    judge_gpus = [5, 7]
    all_gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    layout = {
        "mode": "8gpu",
        "tutor_gpus": tutor_gpus,
        "student_gpus": student_gpus,
        "judge_gpus": judge_gpus,
        "judge_tp": 2,
        "optimizer_offload": True,
        "student_concurrency": args.student_concurrency,
        "judge_concurrency": judge_conc,
        "grader_max_concurrency": grader_conc,
        "theta_scale": DEFAULT_THETA_SCALE,
        "ladder_num_climbs": DEFAULT_LADDER_CLIMBS,
    }
    print(f"[preflight] 固定 8 卡布局: {layout}", flush=True)

    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    (work / "queue-metrics").mkdir(mode=0o700, exist_ok=True)
    (work / "routeA-metrics").mkdir(mode=0o700, exist_ok=True)
    (work / "val-statistics").mkdir(mode=0o700, exist_ok=True)
    ckpt_dir.mkdir(mode=0o700, exist_ok=True)

    ports = pick_free_ports([18861, 18862, 18871, 18872, 18873])
    s_lb_port, j_proxy_port, s0_port, s1_port, j_port = ports
    student_api_base = f"http://127.0.0.1:{s_lb_port}/v1"
    judge_api_base = f"http://127.0.0.1:{j_proxy_port}/v1"

    token = os.environ.get("SERVING_API_KEY") or secrets.token_urlsafe(32)
    user_tag = os.environ.get("USER") or "root"
    student_name = f"Llama-3.1-8B-Instruct-{user_tag}"
    interaction = write_interaction_yaml(repo, work, student_name, student_api_base, layout["student_concurrency"])
    access_cfg = write_access_log_config(work)
    env_base = base_subprocess_env(base)
    env_base["VLLM_API_KEY"] = token
    env_base["SERVING_API_KEY"] = token
    env_base["EDUARDO_CONV_DIR"] = "/dev/null"

    trainer_entry = PRO5000_DIR / "pro5000_trainer_entry.py"
    need_step2_gate = resume_step is None or resume_step < 2

    def make_env(raytmp_path: Path) -> dict[str, str]:
        env_d = build_trainer_env(
            base=base,
            work=work,
            ckpt_dir=ckpt_dir,
            raytmp=raytmp_path,
            tutor_gpus=tutor_gpus,
            all_gpus=all_gpus,
            student_gpus=student_gpus,
            layout=layout,
            repo=repo,
            tutor_model=args.tutor_model,
            student_api_base=student_api_base,
            judge_api_base=judge_api_base,
            start_step=resume_step or 0,
            total_steps=args.total_steps,
            save_freq=args.save_freq,
            test_freq=args.test_freq,
            max_ckpt_keep=args.keep_ckpts,
            check_step1_judge=False,
            check_step2_ratio=need_step2_gate,
            experiment="routeA-t026-stage3",
            ckpt_hint=STAGE3_CKPT_HINT_BYTES,
        )
        env_d.update(
            {
                "PYTHONPATH": f"{CODE_ROOT}:{PRO5000_DIR}:{repo}",
                "PRO5000_REWARD_MODE": "ladder_rasch",
                "PRO5000_DATA_DIR": str(data_dir),
                "PRO5000_THETA_SCALE": str(DEFAULT_THETA_SCALE),
                "PRO5000_LADDER_CLIMBS": str(DEFAULT_LADDER_CLIMBS),
                "PRO5000_GRADER_CONC": str(grader_conc),
                "ROUTEA_STEP_METRICS_DIR": str(work / "routeA-metrics"),
                "PRO5000_ROUTEA_ENFORCE_GATES": "1",
            }
        )
        return env_d

    raytmp = make_ray_tmp(base)
    try:
        overrides = pcfg.build_overrides(
            base=base,
            work=work,
            ckpt_dir=ckpt_dir,
            raytmp=raytmp,
            student_name=student_name,
            interaction=interaction,
            total_steps=args.total_steps,
            save_freq=args.save_freq,
            test_freq=args.test_freq,
            tutor_gpus=len(tutor_gpus),
            optimizer_offload=True,
            student_concurrency=layout["student_concurrency"],
            judge_concurrency=layout["judge_concurrency"],
            max_actor_ckpt_to_keep=args.keep_ckpts,
            resume_step=resume_step,
            reward_mode="ladder_rasch",
            data_dir=data_dir,
            theta_scale=DEFAULT_THETA_SCALE,
            ladder_num_climbs=DEFAULT_LADDER_CLIMBS,
            grader_max_concurrency=grader_conc,
            code_root=CODE_ROOT,
        )
        trainer_env_vars = make_env(raytmp)
        pcfg.run_preflight(
            python_exe=py_exe,
            trainer_entry=trainer_entry,
            repo=repo,
            work=work,
            tutor_model=args.tutor_model,
            student_api_base=student_api_base,
            judge_api_base=judge_api_base,
            overrides=overrides,
            check_env=dict(env_base, **trainer_env_vars),
            code_root=CODE_ROOT,
        )
        print("[preflight] Hydra Route A 阶段 3 冻结配方预检通过", flush=True)
    except Exception:
        shutil.rmtree(raytmp, ignore_errors=True)
        if args.check_only and not is_resume and not args.run_dir:
            shutil.rmtree(work, ignore_errors=True)
        raise

    if args.check_only:
        shutil.rmtree(raytmp, ignore_errors=True)
        if not is_resume and not args.run_dir:
            shutil.rmtree(work, ignore_errors=True)
        print(
            f"STAGE3_PREFLIGHT_OK resume_step={resume_step} "
            f"judge_concurrency={judge_conc} grader_max_concurrency={grader_conc}"
        )
        return 0

    write_json(
        work / "run_meta.json",
        {
            "task": "T-026",
            "stage": "stage3_formal",
            "prereqs": prereq_info,
            "layout": layout,
            "total_steps": args.total_steps,
            "save_freq": args.save_freq,
            "test_freq": args.test_freq,
            "keep_ckpts": args.keep_ckpts,
            "started_at": time.time(),
            "resume_step": resume_step,
        },
    )

    managed_procs: list[tuple[str, subprocess.Popen]] = []
    sampler: Sampler | None = None
    try:
        if resume_step is None or resume_step < args.total_steps:
            first = snapshot_gpus()
            time.sleep(2)
            second = snapshot_gpus()
            verify_stage3_gpus_clean(first, second)

            def spawn(name: str, cmd: list[str], extra_env: dict[str, str], log_name: str) -> subprocess.Popen:
                log = (work / log_name).open("ab")
                try:
                    proc = subprocess.Popen(
                        cmd,
                        env=dict(env_base, **extra_env),
                        cwd=work,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                finally:
                    log.close()
                managed_procs.append((name, proc))
                return proc

            student_backends: list[str] = []
            for idx, (gpu_id, port) in enumerate(zip(student_gpus, (s0_port, s1_port))):
                cmd = [
                    str(py_exe), "-B", "-m", "vllm.entrypoints.openai.api_server",
                    "--model", str(student_model),
                    "--served-model-name", student_name,
                    "--host", "127.0.0.1",
                    "--port", str(port),
                    "--dtype", "bfloat16",
                    "--gpu-memory-utilization", "0.9",
                    "--max-model-len", "32768",
                    "--max-num-seqs", "256",
                    "--tensor-parallel-size", "1",
                    "--data-parallel-size", "1",
                    "--enable-chunked-prefill",
                    "--enable-prefix-caching",
                    "--log-config-file", str(access_cfg),
                ]
                spawn(f"student_{idx}", cmd, {"CUDA_VISIBLE_DEVICES": str(gpu_id)}, f"student_{idx}.log")
                student_backends.append(f"http://127.0.0.1:{port}")

            judge_cmd = [
                str(py_exe), "-B", "-m", "vllm.entrypoints.openai.api_server",
                "--model", str(judge_model),
                "--served-model-name", "judge-local",
                "--host", "127.0.0.1",
                "--port", str(j_port),
                "--dtype", "bfloat16",
                "--gpu-memory-utilization", "0.9",
                "--max-model-len", "32768",
                "--max-num-seqs", "256",
                "--tensor-parallel-size", "2",
                "--data-parallel-size", "1",
                "--enable-chunked-prefill",
                "--enable-prefix-caching",
                "--log-config-file", str(access_cfg),
            ]
            spawn("judge", judge_cmd, {"CUDA_VISIBLE_DEVICES": ",".join(map(str, judge_gpus))}, "judge.log")

            proxy_cmd = [
                str(py_exe), "-B", str(PRO5000_DIR / "serving_proxy.py"),
                "--work", str(work),
                "--student-port", str(s_lb_port),
                "--student-backends", *student_backends,
                "--judge-port", str(j_proxy_port),
                "--judge-backends", f"http://127.0.0.1:{j_port}",
            ]
            spawn("proxy", proxy_cmd, {"CUDA_VISIBLE_DEVICES": ""}, "proxy.log")

            for name, proc in managed_procs:
                if name.startswith("student_"):
                    port = s0_port if name == "student_0" else s1_port
                    print(f"[serving] waiting for {name} (GPU {student_gpus[int(name[-1])]})...", flush=True)
                    wait_for_http_health(f"http://127.0.0.1:{port}/health", proc, name, timeout_s=420)
                elif name == "judge":
                    print(f"[serving] waiting for judge TP=2 (GPUs {judge_gpus})...", flush=True)
                    wait_for_http_health(f"http://127.0.0.1:{j_port}/health", proc, name, timeout_s=600)
                elif name == "proxy":
                    wait_for_http_health(f"http://127.0.0.1:{s_lb_port}/health", proc, "proxy-student", timeout_s=30)
                    wait_for_http_health(f"http://127.0.0.1:{j_proxy_port}/health", proc, "proxy-judge", timeout_s=30)
            print("[serving] all serving backends & proxies healthy!", flush=True)

            sampler = Sampler(work / "tutor-telemetry.jsonl", tuple(all_gpus), lambda: (), interval=10, append=is_resume)
            sampler.start()

            cmd_train = [
                str(py_exe), "-B", str(trainer_entry),
                "--config-path", str(repo / "eduardo" / "configs"),
                "--config-name", "eduardo_base",
                *overrides,
            ]
            log_t = (work / "trainer.log").open("ab")
            proc_t = None
            try:
                proc_t = subprocess.Popen(
                    cmd_train,
                    env=dict(env_base, **trainer_env_vars),
                    cwd=work,
                    stdin=subprocess.DEVNULL,
                    stdout=log_t,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            finally:
                log_t.close()
            try:
                rc = proc_t.wait()
            finally:
                terminate_pg(proc_t.pid)
            if rc != 0:
                raise RuntimeError(f"阶段 3 路线 A 训练退出异常 rc={rc}")

            if sampler is not None:
                sampler.stop()
                sampler = None
            terminate_procs(managed_procs)
            managed_procs.clear()

        final_ckpt = ckpt_dir / f"global_step_{args.total_steps}"
        final_info = verify_model_extra_checkpoint(final_ckpt, world_size=len(tutor_gpus))
        export_dir = work / "exports" / f"step_{args.total_steps}"
        export_info = export_hf_step(py_exe, work, final_ckpt, export_dir, env_base)
        export_manifest = write_export_sha256_manifest(export_dir, work / "export_step100_sha256.txt")
        ckpt_kept = list_verified_checkpoints(work, ckpt_dir, world_size=len(tutor_gpus))
        # Preserve /dev/shm/zhujie_a_group/ intact for downstream evaluation per T-026 & Stage 2 review.
        summary = summarize_stage3_routea(
            work=work,
            layout=layout,
            total_steps=args.total_steps,
            final_info=final_info,
            export_info=export_info,
            export_manifest=export_manifest,
            ckpt_kept=ckpt_kept,
            shm_preserved=True,
        )
        print("=== STAGE 3 SUMMARY ===")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    finally:
        shutil.rmtree(raytmp, ignore_errors=True)
        if sampler is not None:
            sampler.stop()
        terminate_procs(managed_procs)


if __name__ == "__main__":
    sys.exit(main())
