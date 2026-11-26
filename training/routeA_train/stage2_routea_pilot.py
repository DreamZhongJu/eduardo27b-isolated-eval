"""Stage 2 (8-GPU 3-step Route A pilot with step-2 [model, extra] checkpoint resume) for Pro 5000."""
from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
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
from pro5000_trainer_entry import verify_model_extra_checkpoint, write_json
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
from stage3_formal import STAGE3_CKPT_HINT_BYTES, verify_stage3_gpus_clean
from telemetry import Sampler
from training.routeA_reward.hybrid_grader import verify_hybrid_grader_sha256
from training.routeA_train.prepare_parquet import sha256_file

EXPECTED_HYBRID_GRADER_SHA256 = (
    "7d2a5f46619218e92ed11efc4e7e07f190027ac0f1cca9336905c090ff62709a"
)
EXPECTED_PARQUET_SHA256 = {
    "train.parquet": "1e85fd0f2d1282b409e8617b62591fd157b165b397aa3ad514ba3a95d1adee22",
    "val.parquet": "9a6808eacdf3facfdeb39566dce26f13f6b46c8b9c332e1fd2853709f398355f",
}
T017_PILOT_STEP_SECONDS = {1: 1178.46, 2: 1261.45, 3: 1214.47}
MAX_STEP_RATIO_VS_T017 = 1.6
MAX_SERVICE_P95_MS = 300_000.0
MAX_VRAM_FRAC = 0.95
GPU_TOTAL_MIB_DEFAULT = 73415.0
PILOT_DISK_FLOOR_BYTES = 45 * 10**9


def verify_routea_prereqs(base: Path, tutor_model: Path, data_dir: Path) -> Dict[str, Any]:
    mv_ver = importlib_metadata.version("math-verify")
    l2s_ver = importlib_metadata.version("latex2sympy2-extended")
    if mv_ver != "0.8.0":
        raise RuntimeError(f"math-verify 版本不符: {mv_ver} != 0.8.0")
    if l2s_ver != "1.10.2":
        raise RuntimeError(f"latex2sympy2-extended 版本不符: {l2s_ver} != 1.10.2")

    hg_sha = verify_hybrid_grader_sha256(
        CODE_ROOT / "training" / "routeA_reward" / "hybrid_grader.py",
        EXPECTED_HYBRID_GRADER_SHA256,
    )
    receipt = base / "envs" / "pro5000-fast" / "kernel_equivalence.json"
    if not receipt.is_file():
        raise RuntimeError("缺少阶段 1 快内核等价回执 kernel_equivalence.json")
    rec_data = json.loads(receipt.read_text(encoding="utf-8"))
    if not rec_data.get("accepted") or rec_data.get("fast_kernel_verified") is not True:
        raise RuntimeError("阶段 1 等价回执未通过")

    verify_model_dir("Tutor", tutor_model)
    parquet_hashes: Dict[str, str] = {}
    for fname, exp_sha in EXPECTED_PARQUET_SHA256.items():
        fpath = data_dir / fname
        if not fpath.is_file():
            raise RuntimeError(f"缺少路线 A 数据文件: {fpath}")
        actual = sha256_file(fpath)
        if actual != exp_sha:
            raise RuntimeError(f"路线 A 数据文件 sha256 不一致 ({fname}): {actual} != {exp_sha}")
        parquet_hashes[fname] = actual
    return {
        "math-verify": mv_ver,
        "latex2sympy2-extended": l2s_ver,
        "hybrid_grader_sha256": hg_sha,
        "parquet_sha256": parquet_hashes,
    }


def _find_latest_pilot_run(runs_dir: Path) -> Optional[Path]:
    if not runs_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in runs_dir.glob("t026_stage2_pilot_*") if p.is_dir() and not p.is_symlink()),
        key=lambda p: p.name,
    )
    return candidates[-1] if candidates else None


def format_stage2_status(base: Path, run_dir: Path | None = None) -> str:
    work = run_dir or _find_latest_pilot_run(base / "runs")
    if work is None or not work.is_dir():
        return "暂无 t026_stage2_pilot 运行目录"

    proc_lines: List[str] = []
    if (work / "stage2.pid").is_file():
        pid_txt = (work / "stage2.pid").read_text(encoding="utf-8").strip()
        if pid_txt.isdigit():
            try:
                os.kill(int(pid_txt), 0)
                proc_lines.append(f"orchestrator pid={pid_txt} RUNNING")
            except OSError:
                proc_lines.append(f"orchestrator pid={pid_txt} STOPPED")
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", "stage2_routea_pilot.py|pro5000_trainer_entry.py"],
            text=True,
            timeout=5,
        )
        for ln in out.splitlines():
            if "--status" not in ln:
                proc_lines.append(ln[:110])
    except subprocess.CalledProcessError:
        pass

    steps_dir = work / "step-statistics"
    step_rows: List[Dict[str, Any]] = []
    if steps_dir.is_dir():
        for p in sorted(steps_dir.glob("*.json"), key=lambda x: int(x.stem) if x.stem.isdigit() else 0):
            if p.stem.isdigit():
                try:
                    step_rows.append(json.loads(p.read_text(encoding="utf-8")))
                except ValueError:
                    pass

    du = shutil.disk_usage("/data" if Path("/data").exists() else base)
    lines = [
        f"=== T-026 阶段 2 试跑进度 ({work.name}) ===",
        f"进程状态: {' | '.join(proc_lines) if proc_lines else '无运行中进程'}",
        f"已完成步数: {len(step_rows)} / 3 | restore_step_2={'YES' if (work / 'restore_step_2.json').is_file() else 'NO'} | summary={'READY' if (work / 'stage2_summary.json').is_file() else 'PENDING'}",
        f"/data 剩余: {du.free / 1e9:.2f} GB ({du.free / 1024**3:.2f} GiB)",
    ]
    for r in step_rows:
        m = r.get("metrics", {})
        pt = r.get("phase_timings", {})
        ram = r.get("routeA_metrics", {}).get("metrics", {})
        sec = m.get("timing_s/step")
        lines.append(
            f"  step {r.get('step')}: {sec:.1f}s ({sec / 60:.2f}m) | "
            f"gen={pt.get('gen', {}).get('minutes', 0):.2f}m "
            f"old={pt.get('old_log_prob', {}).get('minutes', 0):.2f}m "
            f"upd={pt.get('update_actor', {}).get('minutes', 0):.2f}m | "
            f"reward={ram.get('eduardo/routeA_reward_mean', 0):.3f} "
            f"theta={ram.get('eduardo/routeA_theta_mean', 0):.3f} "
            f"gated={ram.get('eduardo/routeA_gated_rate', 0):.3f}"
            if isinstance(sec, (int, float))
            else f"  step {r.get('step')}: timing N/A"
        )
    try:
        gpu_csv = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
            text=True,
            timeout=5,
        )
        lines.append("8 卡显存与利用率:")
        lines.extend(f"  GPU {ln.strip()}" for ln in gpu_csv.splitlines() if ln.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    bg_log = work / "stage2_bg.log"
    if bg_log.is_file():
        tail = bg_log.read_text(encoding="utf-8", errors="replace").splitlines()[-10:]
        if tail:
            lines.append("最近日志:")
            lines.extend(f"  {ln}" for ln in tail)
    return "\n".join(lines)


def _bottleneck_hint(steps_summary: List[Dict[str, Any]]) -> str:
    slow_steps = [s for s in steps_summary if s.get("ratio_vs_t017", 0.0) > MAX_STEP_RATIO_VS_T017]
    if not slow_steps:
        return "none"
    s = slow_steps[0]
    judge_wall = float(s.get("judge_27b_busy_wall_s") or 0.0)
    stu_wall = float(s.get("student_combined_busy_wall_s") or 0.0)
    tutor_wall = float(s.get("tutor_gen_busy_est_s") or 0.0)
    gen_s = float(s.get("gen_seconds") or 0.0)
    role_walls = {
        "27B_judge_plus_grader": judge_wall,
        "student_8b": stu_wall,
        "tutor_4b_gen": tutor_wall,
    }
    top_role = max(role_walls, key=role_walls.get)  # type: ignore[arg-type]
    return f"top_busy_role={top_role} (gen_s={gen_s:.1f}, walls={role_walls})"


def summarize_stage2_routea(
    work: Path,
    layout: Dict[str, Any],
    step2_ckpt_bytes: int,
    *,
    gpu_totals_mib: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    steps_dir = work / "step-statistics"
    loaded: Dict[int, Dict[str, Any]] = {}
    for step_num in (1, 2, 3):
        sp = steps_dir / f"{step_num}.json"
        if sp.is_file():
            loaded[step_num] = json.loads(sp.read_text(encoding="utf-8"))

    restore_path = work / "restore_step_2.json"
    restore = json.loads(restore_path.read_text(encoding="utf-8")) if restore_path.is_file() else {}
    cmp2_path = work / "step2-comparison.json"
    cmp2 = json.loads(cmp2_path.read_text(encoding="utf-8")) if cmp2_path.is_file() else {}

    totals_mib = dict(gpu_totals_mib or {})
    steps_summary: List[Dict[str, Any]] = []
    gpu_peak_overall: Dict[str, float] = {}

    for step_num in sorted(loaded):
        s = loaded[step_num]
        m = s.get("metrics", {})
        pt = s.get("phase_timings", {})
        telem = s.get("telemetry", {})
        per_gpu_peak: Dict[str, float] = {}
        for phase in ("gen", "old_log_prob", "update_actor"):
            for gid, gvals in telem.get(phase, {}).get("gpus", {}).items():
                mx = gvals.get("used_mib", {}).get("max")
                if mx is not None:
                    per_gpu_peak[gid] = max(per_gpu_peak.get(gid, 0.0), float(mx))
                    gpu_peak_overall[gid] = max(gpu_peak_overall.get(gid, 0.0), float(mx))

        jp = s.get("judge_proxy", {})
        sp = s.get("student_proxy", {})
        qm = s.get("queues", {}).get("groups", {})
        gate_qm = qm.get("judge", {})
        grader_qm = qm.get("hybrid_grader", {})
        stu_int_qm = qm.get("student_interaction", {})
        stu_fin_qm = qm.get("student_final", {})

        vote = s.get("vote_log_counts", {})
        gate_parse = int(vote.get("parse", 0))
        gate_recov = int(vote.get("recovery", 0))
        gate_calls = int(gate_qm.get("calls") or vote.get("requests") or 0)
        gate_parse_rate = (gate_parse / gate_calls) if gate_calls > 0 else 0.0

        ram = s.get("routeA_metrics", {}).get("metrics", {})
        grader_llm_calls = int(round(ram.get("eduardo/routeA_grader_llm_calls_total", 0.0)))
        grader_llm_unknown = int(round(ram.get("eduardo/routeA_grader_llm_unknown_total", 0.0)))
        grader_tech_failed = int(round(ram.get("eduardo/routeA_grader_tech_failed_total", 0.0)))
        grader_qm_calls = int(grader_qm.get("calls", 0))

        gen_sec = float(m.get("timing_s/gen") or 0.0)
        gen_gpus = telem.get("gen", {}).get("gpus", {})
        tutor_utils = [
            float(gen_gpus[str(g)]["util_pct"]["mean"])
            for g in layout["tutor_gpus"]
            if str(g) in gen_gpus and gen_gpus[str(g)].get("util_pct", {}).get("mean") is not None
        ]
        tutor_busy_s = round(gen_sec * (sum(tutor_utils) / len(tutor_utils)) / 100.0, 2) if tutor_utils and gen_sec > 0 else None

        step_s = m.get("timing_s/step")
        t017_s = T017_PILOT_STEP_SECONDS.get(step_num)
        ratio_t017 = round(float(step_s) / t017_s, 4) if (step_s is not None and t017_s) else None

        student_timeouts = int(stu_int_qm.get("attempt_errors", 0)) + int(stu_fin_qm.get("attempt_errors", 0))
        student_failed_rollouts = int(round(ram.get("eduardo/routeA_student_failed_total", 0.0)))

        steps_summary.append(
            {
                "step": step_num,
                "total_num_tokens": m.get("perf/total_num_tokens"),
                "step_seconds": step_s,
                "t017_step_seconds": t017_s,
                "ratio_vs_t017": ratio_t017,
                "gen_seconds": round(gen_sec, 2),
                "phases": {
                    p: {
                        "minutes": round(pt[p]["minutes"], 3) if p in pt else None,
                        "ms_per_token": round(pt[p]["ms_per_token"], 4) if p in pt else None,
                    }
                    for p in ("gen", "old_log_prob", "update_actor")
                },
                "gpu_peak_mib": per_gpu_peak,
                "judge_27b_proxy_requests": int(jp.get("total_chat_requests") or 0),
                "judge_27b_non_2xx": int(jp.get("non_2xx", 0)),
                "judge_27b_proxy_errors": int(jp.get("errors", 0)),
                "judge_27b_busy_wall_s": jp.get("combined_busy_wall_s"),
                "gate_calls": gate_calls,
                "gate_wait_p50_ms": gate_qm.get("wait_p50_ms"),
                "gate_wait_p95_ms": gate_qm.get("wait_p95_ms"),
                "gate_service_p50_ms": gate_qm.get("service_p50_ms"),
                "gate_service_p95_ms": gate_qm.get("service_p95_ms"),
                "gate_attempt_errors": int(gate_qm.get("attempt_errors", 0)),
                "gate_parse_failures": gate_parse,
                "gate_parse_failure_rate": round(gate_parse_rate, 6),
                "gate_regex_recoveries": gate_recov,
                "hybrid_grader_qm_calls": grader_qm_calls,
                "hybrid_grader_wait_p50_ms": grader_qm.get("wait_p50_ms"),
                "hybrid_grader_wait_p95_ms": grader_qm.get("wait_p95_ms"),
                "hybrid_grader_service_p50_ms": grader_qm.get("service_p50_ms"),
                "hybrid_grader_service_p95_ms": grader_qm.get("service_p95_ms"),
                "hybrid_grader_llm_calls": grader_llm_calls,
                "hybrid_grader_llm_unknown": grader_llm_unknown,
                "hybrid_grader_llm_unknown_rate": round(ram.get("eduardo/routeA_grader_llm_unknown_rate", 0.0), 6),
                "hybrid_grader_tech_failed": grader_tech_failed,
                "hybrid_grader_tech_fail_rate": round(ram.get("eduardo/routeA_grader_tech_failed_rate", 0.0), 6),
                "student_proxy_requests": int(sp.get("total_chat_requests") or 0),
                "student_non_2xx": int(sp.get("non_2xx", 0)),
                "student_attempt_errors": student_timeouts,
                "student_rollout_failed_total": student_failed_rollouts,
                "student_combined_busy_wall_s": sp.get("combined_busy_wall_s"),
                "student_backends": sp.get("backends", {}),
                "tutor_gen_busy_est_s": tutor_busy_s,
                "tutor_gen_mean_util_pct": round(sum(tutor_utils) / len(tutor_utils), 2) if tutor_utils else None,
                "routeA_reward_metrics": ram,
                "actor_grad_norm": m.get("actor/grad_norm"),
                "actor_pg_loss": m.get("actor/pg_loss"),
            }
        )

    # Evaluate the 4 Stage 2 acceptance gates
    gate_failures: List[str] = []

    steps_ok = set(loaded.keys()) == {1, 2, 3}
    resume_ok = restore.get("global_step") == 2 and restore.get("data_batches") == 2
    if not steps_ok:
        gate_failures.append(f"missing_steps={sorted({1,2,3} - set(loaded.keys()))}")
    if not resume_ok:
        gate_failures.append(f"resume_mismatch={restore}")

    vram_violations: List[str] = []
    vram_peak_pct: Dict[str, float] = {}
    for gid_str, mx in sorted(gpu_peak_overall.items(), key=lambda kv: int(kv[0])):
        tot = float(totals_mib.get(int(gid_str), GPU_TOTAL_MIB_DEFAULT))
        pct = mx / tot if tot > 0 else 1.0
        vram_peak_pct[gid_str] = round(100.0 * pct, 2)
        if pct > MAX_VRAM_FRAC:
            vram_violations.append(f"GPU {gid_str}: {mx:.0f}/{tot:.0f} MiB ({100*pct:.2f}% > 95%)")
    if vram_violations:
        gate_failures.extend(vram_violations)

    serving_violations: List[str] = []
    for row in steps_summary:
        st = row["step"]
        if row["judge_27b_non_2xx"] != 0 or row["judge_27b_proxy_errors"] != 0:
            serving_violations.append(f"step {st}: 27B non_2xx={row['judge_27b_non_2xx']} errors={row['judge_27b_proxy_errors']}")
        gc = row["gate_calls"]
        gp = row["gate_parse_failures"]
        if gc <= 0 or 100 * gp > gc:
            serving_violations.append(f"step {st}: gate_parse_fail={gp}/{gc} (>1% or 0 calls)")
        hc = row["hybrid_grader_llm_calls"]
        hf = row["hybrid_grader_tech_failed"]
        if hc > 0 and 100 * hf > hc:
            serving_violations.append(f"step {st}: hybrid_tech_fail={hf}/{hc} (>1%)")
        gp95 = row["gate_service_p95_ms"]
        if gp95 is None or gp95 >= MAX_SERVICE_P95_MS:
            serving_violations.append(f"step {st}: gate_service_p95_ms={gp95} >= 300000")
        hp95 = row["hybrid_grader_service_p95_ms"]
        if row["hybrid_grader_qm_calls"] > 0 and (hp95 is None or hp95 >= MAX_SERVICE_P95_MS):
            serving_violations.append(f"step {st}: hybrid_service_p95_ms={hp95} >= 300000")
    if serving_violations:
        gate_failures.extend(serving_violations)

    timing_violations: List[str] = []
    for row in steps_summary:
        ratio = row.get("ratio_vs_t017")
        if ratio is None or ratio > MAX_STEP_RATIO_VS_T017:
            timing_violations.append(
                f"step {row['step']}: {row.get('step_seconds')}s vs T-017 {row.get('t017_step_seconds')}s (ratio={ratio} > 1.6)"
            )
    if timing_violations:
        gate_failures.extend(timing_violations)

    gates_passed = len(gate_failures) == 0
    all_secs = [float(r["step_seconds"]) for r in steps_summary if r.get("step_seconds") is not None]
    steady_secs = [float(r["step_seconds"]) for r in steps_summary[1:] if r.get("step_seconds") is not None] or all_secs
    avg_all3_s = (sum(all_secs) / len(all_secs)) if all_secs else 0.0
    avg_steady_s = (sum(steady_secs) / len(steady_secs)) if steady_secs else 0.0

    return {
        "status": "STAGE2_PILOT_DONE",
        "rc": 0 if gates_passed else 4,
        "gates_passed": gates_passed,
        "gate_failures": gate_failures,
        "per_gate": {
            "steps_and_resume": steps_ok and resume_ok,
            "vram_under_95pct": not vram_violations,
            "serving_quality": not serving_violations,
            "step_time_within_1_6x": not timing_violations,
        },
        "bottleneck_hint": _bottleneck_hint(steps_summary),
        "layout": layout,
        "restore": restore,
        "step2_comparison": cmp2,
        "step2_ckpt_gb": round(step2_ckpt_bytes / 1e9, 2),
        "gpu_peak_overall_mib": gpu_peak_overall,
        "gpu_peak_pct_of_capacity": vram_peak_pct,
        "steps": steps_summary,
        "all3_avg_step_s": round(avg_all3_s, 2),
        "steady_avg_step_s": round(avg_steady_s, 2),
        "estimated_100_steps_hours_all3": round(100.0 * avg_all3_s / 3600.0, 2),
        "estimated_100_steps_hours_steady": round(100.0 * avg_steady_s / 3600.0, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=default_base())
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--tutor-model", type=Path, default=pcfg.TEACHER_MODEL_DEFAULT)
    parser.add_argument("--judge-concurrency", type=int, default=16)
    parser.add_argument("--grader-max-concurrency", type=int, default=32)
    parser.add_argument("--student-concurrency", type=int, default=32)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)

    base = args.base.resolve()
    if args.status:
        print(format_stage2_status(base, run_dir=args.run_dir))
        return 0

    py_exe = Path(sys.executable).resolve()
    data_dir = base / "data" / "routeA_t026"
    prereq_info = verify_routea_prereqs(base, args.tutor_model, data_dir)
    repo = ensure_eduardo_repo(base)

    free_bytes = shutil.disk_usage("/data").free
    print(f"[preflight] /data free = {free_bytes / 1e9:.2f} GB ({free_bytes / 1024**3:.2f} GiB)", flush=True)
    if free_bytes < PILOT_DISK_FLOOR_BYTES:
        print("STOP: /data 剩余低于 45G")
        return 3

    first = snapshot_gpus()
    time.sleep(2)
    second = snapshot_gpus()
    verify_stage3_gpus_clean(first, second)
    gpu_totals_mib = {idx: float(info["total_mib"]) for idx, info in second.items()}

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
        "judge_concurrency": args.judge_concurrency,
        "grader_max_concurrency": args.grader_max_concurrency,
        "theta_scale": 0.25,
        "ladder_num_climbs": 8,
    }
    print(f"[preflight] 固定 8 卡布局: {layout}", flush=True)

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
    print(f"[preflight] 模型就绪: student={student_model} -> {student_model.resolve()} ; judge={judge_model} -> {judge_model.resolve()}", flush=True)

    if args.preflight_only and args.run_dir is None:
        work = Path(tempfile.mkdtemp(dir=base / "tmp", prefix=".t026_s2_pf_"))
    else:
        run_name = f"t026_stage2_pilot_{time.strftime('%Y%m%d_%H%M%S')}"
        work = args.run_dir.resolve() if args.run_dir else (base / "runs" / run_name)
    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    (work / "queue-metrics").mkdir(mode=0o700, exist_ok=True)
    (work / "routeA-metrics").mkdir(mode=0o700, exist_ok=True)
    ckpt_dir = work / "checkpoints"
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

    def make_trainer_env(raytmp: Path, start_step: int, total_steps: int, save_freq: int, ckpt_hint: int | None = None) -> dict[str, str]:
        env_d = build_trainer_env(
            base=base,
            work=work,
            ckpt_dir=ckpt_dir,
            raytmp=raytmp,
            tutor_gpus=tutor_gpus,
            all_gpus=all_gpus,
            student_gpus=student_gpus,
            layout=layout,
            repo=repo,
            tutor_model=args.tutor_model,
            student_api_base=student_api_base,
            judge_api_base=judge_api_base,
            start_step=start_step,
            total_steps=total_steps,
            save_freq=save_freq,
            test_freq=25,
            max_ckpt_keep=None,
            check_step1_judge=False,
            check_step2_ratio=(start_step == 0),
            experiment="routeA-t026-stage2",
            ckpt_hint=ckpt_hint,
        )
        env_d.update(
            {
                "PYTHONPATH": f"{CODE_ROOT}:{PRO5000_DIR}:{repo}",
                "PRO5000_REWARD_MODE": "ladder_rasch",
                "PRO5000_DATA_DIR": str(data_dir),
                "PRO5000_THETA_SCALE": "0.25",
                "PRO5000_LADDER_CLIMBS": "8",
                "PRO5000_GRADER_CONC": str(args.grader_max_concurrency),
                "ROUTEA_STEP_METRICS_DIR": str(work / "routeA-metrics"),
                "PRO5000_ROUTEA_ENFORCE_GATES": "0",
            }
        )
        return env_d

    raytmp_a = make_ray_tmp(base)
    try:
        overrides_a = pcfg.build_overrides(
            base=base,
            work=work,
            ckpt_dir=ckpt_dir,
            raytmp=raytmp_a,
            student_name=student_name,
            interaction=interaction,
            total_steps=2,
            save_freq=2,
            test_freq=25,
            tutor_gpus=len(tutor_gpus),
            optimizer_offload=True,
            student_concurrency=layout["student_concurrency"],
            judge_concurrency=layout["judge_concurrency"],
            reward_mode="ladder_rasch",
            data_dir=data_dir,
            theta_scale=0.25,
            ladder_num_climbs=8,
            grader_max_concurrency=args.grader_max_concurrency,
            code_root=CODE_ROOT,
        )
        env_a = make_trainer_env(raytmp_a, 0, 2, 2)
        pcfg.run_preflight(
            python_exe=py_exe,
            trainer_entry=trainer_entry,
            repo=repo,
            work=work,
            tutor_model=args.tutor_model,
            student_api_base=student_api_base,
            judge_api_base=judge_api_base,
            overrides=overrides_a,
            check_env=dict(env_base, **env_a),
            code_root=CODE_ROOT,
        )
    except Exception:
        shutil.rmtree(raytmp_a, ignore_errors=True)
        if args.preflight_only and args.run_dir is None:
            shutil.rmtree(work, ignore_errors=True)
        raise

    print("[preflight] Hydra Route A 阶段 2 预检通过", flush=True)
    if args.preflight_only:
        shutil.rmtree(raytmp_a, ignore_errors=True)
        if args.run_dir is None:
            shutil.rmtree(work, ignore_errors=True)
        print("STAGE2_PREFLIGHT_OK")
        return 0

    write_json(
        work / "run_meta.json",
        {
            "task": "T-026",
            "stage": "stage2_pilot",
            "prereqs": prereq_info,
            "layout": layout,
            "started_at": time.time(),
        },
    )

    managed_procs: List[tuple[str, subprocess.Popen]] = []
    sampler: Sampler | None = None
    step2_ckpt_bytes = 0
    try:
        def spawn(name: str, cmd: List[str], extra_env: Dict[str, str], log_name: str) -> subprocess.Popen:
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

        student_backends: List[str] = []
        for idx, (gpu_id, port) in enumerate(zip(student_gpus, (s0_port, s1_port))):
            cmd = [
                str(py_exe), "-B", "-m", "vllm.entrypoints.openai.api_server",
                "--model", str(student_model), "--served-model-name", student_name,
                "--host", "127.0.0.1", "--port", str(port),
                "--dtype", "bfloat16", "--gpu-memory-utilization", "0.9",
                "--max-model-len", "32768", "--max-num-seqs", "256",
                "--tensor-parallel-size", "1", "--data-parallel-size", "1",
                "--enable-chunked-prefill", "--enable-prefix-caching",
                "--log-config-file", str(access_cfg),
            ]
            spawn(f"student_{idx}", cmd, {"CUDA_VISIBLE_DEVICES": str(gpu_id)}, f"student_{idx}.log")
            student_backends.append(f"http://127.0.0.1:{port}")

        judge_cmd = [
            str(py_exe), "-B", "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(judge_model), "--served-model-name", "judge-local",
            "--host", "127.0.0.1", "--port", str(j_port),
            "--dtype", "bfloat16", "--gpu-memory-utilization", "0.9",
            "--max-model-len", "32768", "--max-num-seqs", "256",
            "--tensor-parallel-size", "2", "--data-parallel-size", "1",
            "--enable-chunked-prefill", "--enable-prefix-caching",
            "--log-config-file", str(access_cfg),
        ]
        spawn("judge", judge_cmd, {"CUDA_VISIBLE_DEVICES": ",".join(map(str, judge_gpus))}, "judge.log")

        proxy_cmd = [
            str(py_exe), "-B", str(PRO5000_DIR / "serving_proxy.py"),
            "--work", str(work),
            "--student-port", str(s_lb_port), "--student-backends", *student_backends,
            "--judge-port", str(j_proxy_port), "--judge-backends", f"http://127.0.0.1:{j_port}",
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

        sampler = Sampler(work / "tutor-telemetry.jsonl", tuple(all_gpus), lambda: (), interval=10)
        sampler.start()

        # Phase A: steps 1-2 + save [model, extra]
        print("[phase A] starting steps 1-2 (will save step-2 model+extra checkpoint and stop)...", flush=True)
        cmd_a = [
            str(py_exe), "-B", str(trainer_entry),
            "--config-path", str(repo / "eduardo" / "configs"), "--config-name", "eduardo_base",
            *overrides_a,
        ]
        log_a = (work / "trainer_phaseA.log").open("xb")
        proc_a = None
        try:
            proc_a = subprocess.Popen(
                cmd_a,
                env=dict(env_base, **env_a),
                cwd=work,
                stdin=subprocess.DEVNULL,
                stdout=log_a,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_a.close()
        try:
            rc_a = proc_a.wait()
        finally:
            terminate_pg(proc_a.pid)
            shutil.rmtree(raytmp_a, ignore_errors=True)
        if rc_a != 0:
            raise RuntimeError(f"阶段 2 前两步训练退出异常 rc={rc_a}")

        ckpt_step2 = ckpt_dir / "global_step_2"
        ckpt_info = verify_model_extra_checkpoint(ckpt_step2, world_size=len(tutor_gpus))
        step2_ckpt_bytes = ckpt_info["size_bytes"]
        print(f"[phase A] step-2 [model,extra] checkpoint verified: {ckpt_info}", flush=True)

        # Phase B: fresh trainer resume from global_step_2 -> step 3
        print("[phase B] starting fresh trainer to resume from global_step_2 -> step 3...", flush=True)
        raytmp_b = make_ray_tmp(base)
        try:
            overrides_b = pcfg.build_overrides(
                base=base,
                work=work,
                ckpt_dir=ckpt_dir,
                raytmp=raytmp_b,
                student_name=student_name,
                interaction=interaction,
                total_steps=3,
                save_freq=25,
                test_freq=25,
                tutor_gpus=len(tutor_gpus),
                optimizer_offload=True,
                student_concurrency=layout["student_concurrency"],
                judge_concurrency=layout["judge_concurrency"],
                resume_step=2,
                reward_mode="ladder_rasch",
                data_dir=data_dir,
                theta_scale=0.25,
                ladder_num_climbs=8,
                grader_max_concurrency=args.grader_max_concurrency,
                code_root=CODE_ROOT,
            )
            env_b = make_trainer_env(raytmp_b, 2, 3, 25, ckpt_hint=step2_ckpt_bytes)
            pcfg.run_preflight(
                python_exe=py_exe,
                trainer_entry=trainer_entry,
                repo=repo,
                work=work,
                tutor_model=args.tutor_model,
                student_api_base=student_api_base,
                judge_api_base=judge_api_base,
                overrides=overrides_b,
                check_env=dict(env_base, **env_b),
                code_root=CODE_ROOT,
            )
            cmd_b = [
                str(py_exe), "-B", str(trainer_entry),
                "--config-path", str(repo / "eduardo" / "configs"), "--config-name", "eduardo_base",
                *overrides_b,
            ]
            log_b = (work / "trainer_phaseB.log").open("xb")
            proc_b = None
            try:
                proc_b = subprocess.Popen(
                    cmd_b,
                    env=dict(env_base, **env_b),
                    cwd=work,
                    stdin=subprocess.DEVNULL,
                    stdout=log_b,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            finally:
                log_b.close()
            rc_b = proc_b.wait()
        finally:
            if "proc_b" in locals() and proc_b is not None:
                terminate_pg(proc_b.pid)
            shutil.rmtree(raytmp_b, ignore_errors=True)
        if rc_b != 0:
            raise RuntimeError(f"阶段 2 第 3 步续训退出异常 rc={rc_b}")

        if sampler is not None:
            sampler.stop()
            sampler = None
        terminate_procs(managed_procs)
        managed_procs.clear()

        summary = summarize_stage2_routea(
            work,
            layout,
            step2_ckpt_bytes,
            gpu_totals_mib=gpu_totals_mib,
        )
        if summary["gates_passed"]:
            for gdir in sorted(ckpt_dir.glob("global_step_*")):
                if gdir.is_dir():
                    shutil.rmtree(gdir)
            summary["pilot_ckpts_deleted"] = not any(ckpt_dir.glob("global_step_*"))
        else:
            summary["pilot_ckpts_deleted"] = False

        summary["post_run_disk_free_gb"] = round(shutil.disk_usage("/data").free / 1e9, 2)
        summary["work_dir"] = str(work)
        write_json(work / "stage2_summary.json", summary)

        print("=== STAGE 2 SUMMARY ===")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"STAGE2_PILOT_DONE={summary['rc']}")
        return int(summary["rc"])
    finally:
        if sampler is not None:
            sampler.stop()
        terminate_procs(managed_procs)


if __name__ == "__main__":
    raise SystemExit(main())
