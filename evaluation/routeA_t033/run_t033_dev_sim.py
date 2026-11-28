#!/usr/bin/env python3
"""T-033 Pipeline 3: Dev-set Route A reward simulation for Route A Step-100 only.

Reuses training/routeA_reward/stage2_dev_sim.py (rollout_one_dialogue, climb_16_recorded,
RoundRobinClientPool), untouched hybrid_grader.py (7d2a5f46...), and EduardoRL's 27B
majority_vote gate (no_leak & teaching) + T-026 reward formula (theta_scale=0.25).
Runs 16 ladder climbs unconditionally on every dialogue so theta_hat is directly
comparable to T-024 base (-0.1825) and a_group (+0.3949).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import secrets
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    DEFAULT_BASE,
    DEFAULT_ROUTEA_EXPORT_REL,
    DEFAULT_ROUTEA_SHA_REL,
    HYBRID_GRADER_SHA256,
    ROUTEA_STEP100_LOCK,
    build_clean_env,
    ensure_base_cache_dirs,
    format_machine_summary,
    locate_eduardo_repo,
    pick_free_ports,
    resolve_local_model,
    select_free_gpus,
    sha256_file,
    snapshot_gpus,
    source_key,
    terminate_procs,
    verify_dir_against_lock,
    verify_tutor_bfloat16,
    wait_for_health,
    write_json_atomic,
)
from training.routeA_reward.rasch import rasch_map_estimate  # noqa: E402
from training.routeA_reward.stage2_dev_sim import (  # noqa: E402
    EXPECTED_T023_JSONL_SHA256,
    NUM_CLIMBS_PER_DIALOGUE,
    NUM_DIALOGUES_PER_SOURCE,
    RoundRobinClientPool,
    TUTOR_VLLM_MAX_MODEL_LEN,
    climb_16_recorded,
    load_dev_problems,
    rollout_one_dialogue,
)

DEFAULT_THETA_SCALE = 0.25
DEFAULT_GATE_PENALTY = 0.2


def _extract_observations(climb_res: dict[str, Any], max_climbs: int) -> list[tuple[float, bool]]:
    obs: list[tuple[float, bool]] = []
    for climb in climb_res["climbs"][:max_climbs]:
        for step in climb["steps"]:
            if not step.get("tech_failed", False):
                obs.append((float(step["b"]), bool(step["correct"])))
    return obs


def compute_dialogue_routea_metrics(
    *,
    transcript: list[dict[str, str]],
    climb_res: dict[str, Any],
    no_leak: str,
    teaching: str,
    tokenizer: Any,
    theta_scale: float = DEFAULT_THETA_SCALE,
    gate_penalty: float = DEFAULT_GATE_PENALTY,
) -> dict[str, Any]:
    from eduardo.utils import strip_teacher_controls
    from training.routeA_reward.reward_function import _compute_teacher_stats

    leaked = int(no_leak == "REJECT")
    teaching_rejected = int(teaching == "REJECT")
    gated = int(bool(leaked or teaching_rejected))

    teacher_msgs = [m for m in transcript if m.get("role") == "assistant"]
    n_turns = len(teacher_msgs)
    blanked_turns = sum(
        1 for m in teacher_msgs if not strip_teacher_controls(m.get("content") or "", assume_thinking=True).strip()
    )
    unterm_turns = sum(1 for m in teacher_msgs if "</think>" not in (m.get("content") or ""))

    tstats = _compute_teacher_stats(
        transcript,
        tokenizer=tokenizer,
        enable_thinking=True,
        target_tokens=150.0,
        think_target_tokens=512.0,
    )
    len_decay = float(tstats["length_decay"])
    think_decay = float(tstats["thinking_decay"])
    thinking_tokens_sum = int(tstats["thinking_tokens_total"])

    def _score_k(k: int) -> dict[str, Any]:
        obs = _extract_observations(climb_res, k)
        est = rasch_map_estimate(obs)
        theta = float(est.theta) if (est.valid and est.theta is not None) else None
        if gated:
            reward = -gate_penalty
        elif theta is None:
            reward = 0.0
        else:
            gain = theta * theta_scale
            if gain > 0.0:
                reward = gain * len_decay * think_decay
            else:
                reward = gain
        return {
            "valid": bool(est.valid and theta is not None),
            "theta_hat": theta,
            "reward": float(reward),
            "prior_pulled": int(est.prior_pulled) if est.valid else 0,
        }

    k8 = _score_k(8)
    k16 = _score_k(16)
    return {
        "no_leak": no_leak,
        "teaching": teaching,
        "leaked": leaked,
        "teaching_rejected": teaching_rejected,
        "gated": gated,
        "teacher_turns": n_turns,
        "blanked_turns": blanked_turns,
        "unterminated_think_turns": unterm_turns,
        "thinking_tokens_sum": thinking_tokens_sum,
        "length_decay": float(len_decay),
        "thinking_decay": float(think_decay),
        "k8": k8,
        "k16": k16,
    }


def wait_4_or_5_gpus(*, poll_seconds: int = 60, max_gpus: int = 5) -> dict[str, Any]:
    polls = 0
    while True:
        s1 = snapshot_gpus()
        time.sleep(2)
        s2 = snapshot_gpus()
        free = select_free_gpus(s1, s2, min_free_mib=65000)
        if max_gpus >= 5 and len(free) >= 5:
            picked = free[:5]
            return {"num_gpus": 5, "tutor_gpu": picked[0], "student_gpus": picked[1:3], "grader_gpus": picked[3:5]}
        if len(free) >= 4:
            picked = free[:4]
            return {"num_gpus": 4, "tutor_gpu": picked[0], "student_gpus": picked[1:2], "grader_gpus": picked[2:4]}
        polls += 1
        print(f"[t033_dev] poll={polls} free={free} (<4), sleeping {poll_seconds}s...", flush=True)
        time.sleep(poll_seconds)


async def run_dev_eval(
    *,
    dev_rows: list[dict[str, Any]],
    tutor_path: Path,
    tutor_client: Any,
    student_client: Any,
    grader_27b_client: Any,
    grader_27b_model: str,
    run_dir: Path,
    tutor_conc: int = 16,
    student_conc: int = 32,
    gate_conc: int = 32,
) -> list[dict[str, Any]]:
    from eduardo.metrics import judge_quality as author_judge_quality
    from eduardo.utils import load_prompt, strip_teacher_controls
    from transformers import AutoTokenizer
    from training.routeA_reward.hybrid_grader import (
        AsyncAdaptiveWindow,
        HybridGrader,
        check_technical_failure_rate,
    )

    teacher_sys = load_prompt("teacher_prompt")
    student_sys = load_prompt("student_prompt")
    tokenizer = AutoTokenizer.from_pretrained(str(tutor_path), trust_remote_code=True, local_files_only=True)

    def _count_chat_tokens(msgs: list[dict[str, str]]) -> int:
        return len(
            tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=True)
        )

    grader = HybridGrader(
        client=grader_27b_client,
        model=grader_27b_model,
        window=AsyncAdaptiveWindow(max_window=16),
        use_vllm_no_thinking=True,
    )

    ckpt_dir = run_dir / "checkpoints" / "routeA_step100"
    ckpt_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    priv_dir = run_dir / "private_dialogues" / "routeA_step100"
    priv_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    tutor_sem = asyncio.Semaphore(tutor_conc)
    student_sem = asyncio.Semaphore(student_conc)
    gate_sem = asyncio.Semaphore(gate_conc)
    item_sem = asyncio.Semaphore(tutor_conc)

    retry_kwargs = {"num_retries": 3, "backoff_base": 2.0, "backoff_cap": 30.0, "jitter": True}
    done = 0
    total = len(dev_rows) * NUM_DIALOGUES_PER_SOURCE
    cum_llm = 0
    cum_fail = 0
    t0 = time.monotonic()

    async def _one(row: dict[str, Any], d_idx: int) -> dict[str, Any]:
        nonlocal done, cum_llm, cum_fail
        skey = source_key(row["source_id"])
        ckpt_p = ckpt_dir / f"{skey}_d{d_idx}.json"
        if ckpt_p.is_file():
            done += 1
            return json.loads(ckpt_p.read_text(encoding="utf-8"))

        async with item_sem:
            ts = time.monotonic()
            transcript, dmeta = await rollout_one_dialogue(
                problem=row["problem"],
                tutor_client=tutor_client,
                tutor_model="tutor-routeA-step100",
                student_client=student_client,
                student_model="student-local",
                teacher_system_prompt=teacher_sys,
                student_system_prompt=student_sys,
                strip_teacher_controls_fn=strip_teacher_controls,
                tutor_semaphore=tutor_sem,
                student_semaphore=student_sem,
                count_chat_tokens_fn=_count_chat_tokens,
            )
            write_json_atomic(
                priv_dir / f"{skey}_d{d_idx}.json",
                {"source_key": skey, "dialogue_idx": d_idx, "messages": transcript},
            )

            judge_msgs = [
                {**m, "content": strip_teacher_controls(m.get("content") or "", assume_thinking=True)}
                if m.get("role") == "assistant"
                else m
                for m in transcript
            ]
            (no_leak, teaching, _), climb_res = await asyncio.gather(
                author_judge_quality.majority_vote(
                    messages=judge_msgs,
                    problem=row["problem"],
                    answer=row["answer"],
                    client=grader_27b_client,
                    model=grader_27b_model,
                    num_votes=3,
                    temperature=0.6,
                    max_tokens=4096,
                    retry_kwargs=retry_kwargs,
                    semaphore=gate_sem,
                ),
                climb_16_recorded(
                    transcript=transcript,
                    problem=row["problem"],
                    variants=row["variants"],
                    student_client=student_client,
                    student_model="student-local",
                    grader=grader,
                    student_semaphore=student_sem,
                    num_climbs=NUM_CLIMBS_PER_DIALOGUE,
                ),
            )
            metrics = compute_dialogue_routea_metrics(
                transcript=transcript,
                climb_res=climb_res,
                no_leak=no_leak,
                teaching=teaching,
                tokenizer=tokenizer,
            )
            rec = {
                "source_id": row["source_id"],
                "source_key": skey,
                "dialogue_idx": d_idx,
                "dialogue_meta": dmeta,
                "wall_s": round(time.monotonic() - ts, 2),
                "metrics": metrics,
                "student_calls": climb_res["student_calls"],
                "student_failed": climb_res["student_failed"],
                "grader_llm_calls": climb_res["grader_llm_calls"],
                "grader_llm_unknown": climb_res["grader_llm_unknown"],
                "grader_tech_failed": climb_res["grader_tech_failed"],
            }
            write_json_atomic(ckpt_p, rec)
            done += 1
            cum_llm += int(climb_res["grader_llm_calls"])
            cum_fail += int(climb_res["grader_tech_failed"])
            if done % 10 == 0 or done == total:
                print(
                    f"[t033_dev] progress={done}/{total} elapsed={time.monotonic()-t0:.1f}s "
                    f"llm_calls={cum_llm} tech_fail={cum_fail}",
                    flush=True,
                )
            if cum_llm >= 50 and check_technical_failure_rate(cum_fail, cum_llm):
                raise RuntimeError(f"27B grader technical failure rate > 1%: {cum_fail}/{cum_llm}")
            return rec

    return list(
        await asyncio.gather(*[_one(row, d_idx) for row in dev_rows for d_idx in range(NUM_DIALOGUES_PER_SOURCE)])
    )


def summarize_dev_records(records: list[dict[str, Any]], *, out_dir: Path, elapsed_s: float) -> dict[str, Any]:
    by_sid: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_sid.setdefault(rec["source_id"], []).append(rec)

    csv_rows: list[dict[str, Any]] = []
    th_k8_all, th_k16_all, rw_k8_all, rw_k16_all = [], [], [], []
    gated_srcs = []

    for sid in sorted(by_sid):
        recs = by_sid[sid]
        k8_th = [r["metrics"]["k8"]["theta_hat"] for r in recs if r["metrics"]["k8"]["theta_hat"] is not None]
        k16_th = [r["metrics"]["k16"]["theta_hat"] for r in recs if r["metrics"]["k16"]["theta_hat"] is not None]
        k8_rw = [r["metrics"]["k8"]["reward"] for r in recs]
        k16_rw = [r["metrics"]["k16"]["reward"] for r in recs]
        g_rate = statistics.fmean(r["metrics"]["gated"] for r in recs)
        l_rate = statistics.fmean(r["metrics"]["leaked"] for r in recs)
        t_rate = statistics.fmean(r["metrics"]["teaching_rejected"] for r in recs)
        m_k8 = statistics.fmean(k8_th) if k8_th else 0.0
        m_k16 = statistics.fmean(k16_th) if k16_th else 0.0
        m_rw8 = statistics.fmean(k8_rw)
        m_rw16 = statistics.fmean(k16_rw)
        th_k8_all.append(m_k8)
        th_k16_all.append(m_k16)
        rw_k8_all.append(m_rw8)
        rw_k16_all.append(m_rw16)
        gated_srcs.append(g_rate)
        csv_rows.append(
            {
                "source_id": sid,
                "source_key": source_key(sid),
                "n_dialogues": len(recs),
                "theta_hat_k8": f"{m_k8:.6f}",
                "theta_hat_k16": f"{m_k16:.6f}",
                "reward_k8": f"{m_rw8:.6f}",
                "reward_k16": f"{m_rw16:.6f}",
                "gated_rate": f"{g_rate:.6f}",
                "leak_rate": f"{l_rate:.6f}",
                "teaching_reject_rate": f"{t_rate:.6f}",
            }
        )

    csv_path = out_dir / "dev_per_problem.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    comp_th16 = [
        r["metrics"]["k16"]["theta_hat"]
        for r in records
        if r["metrics"]["gated"] == 0 and r["metrics"]["k16"]["theta_hat"] is not None
    ]
    comp_th8 = [
        r["metrics"]["k8"]["theta_hat"]
        for r in records
        if r["metrics"]["gated"] == 0 and r["metrics"]["k8"]["theta_hat"] is not None
    ]
    tot_turns = sum(r["metrics"]["teacher_turns"] for r in records)
    tot_blank = sum(r["metrics"]["blanked_turns"] for r in records)
    tot_unterm = sum(r["metrics"]["unterminated_think_turns"] for r in records)
    tot_tok = sum(r["metrics"]["thinking_tokens_sum"] for r in records)
    tot_eoc = sum(int(bool(r["dialogue_meta"].get("eoc_used"))) for r in records)
    tot_trunc = sum(int(bool(r["dialogue_meta"].get("truncated_seq"))) for r in records)
    tot_stu_calls = sum(int(r["student_calls"]) for r in records)
    tot_stu_fail = sum(int(r["student_failed"]) + int(bool(r["dialogue_meta"].get("student_interaction_failed"))) for r in records)
    tot_llm = sum(int(r["grader_llm_calls"]) for r in records)
    tot_llm_unk = sum(int(r["grader_llm_unknown"]) for r in records)
    tot_tech_fail = sum(int(r["grader_tech_failed"]) for r in records)

    ok = tot_stu_fail == 0 and (tot_llm == 0 or (tot_tech_fail / tot_llm) <= 0.01)
    summary = {
        "status": "DEV_SIM_DONE" if ok else "DEV_SIM_FAIL",
        "rc": 0 if ok else 2,
        "elapsed_s": round(elapsed_s, 2),
        "n_sources": len(by_sid),
        "n_dialogues": len(records),
        "per_problem_csv": csv_path.name,
        "theta_hat_k16_mean": round(statistics.fmean(th_k16_all), 6),
        "theta_hat_k8_mean": round(statistics.fmean(th_k8_all), 6),
        "theta_hat_k16_compliant_mean": round(statistics.fmean(comp_th16), 6) if comp_th16 else None,
        "theta_hat_k8_compliant_mean": round(statistics.fmean(comp_th8), 6) if comp_th8 else None,
        "reward_k8_mean": round(statistics.fmean(rw_k8_all), 6),
        "reward_k16_mean": round(statistics.fmean(rw_k16_all), 6),
        "gated_rate": round(statistics.fmean(gated_srcs), 6),
        "leak_rate": round(statistics.fmean(r["metrics"]["leaked"] for r in records), 6),
        "teaching_reject_rate": round(statistics.fmean(r["metrics"]["teaching_rejected"] for r in records), 6),
        "student_calls": tot_stu_calls,
        "student_failed": tot_stu_fail,
        "grader_llm_calls": tot_llm,
        "grader_llm_unknown": tot_llm_unk,
        "grader_tech_failed": tot_tech_fail,
        "diagnostics": {
            "teacher_turns": tot_turns,
            "blanked_turns": tot_blank,
            "empty_rate": (tot_blank / tot_turns) if tot_turns else 0.0,
            "unterminated_think_turns": tot_unterm,
            "unterminated_think_rate": (tot_unterm / tot_turns) if tot_turns else 0.0,
            "eoc_rate": (tot_eoc / len(records)) if records else 0.0,
            "seq_trunc_rate": (tot_trunc / len(records)) if records else 0.0,
            "thinking_tokens_avg_per_turn": (tot_tok / tot_turns) if tot_turns else 0.0,
        },
    }
    write_json_atomic(out_dir / "dev_summary.json", summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="T-033 Pipeline 3: Dev 100 Route A simulation.")
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--tutor-model", type=Path, default=None)
    ap.add_argument("--step100-sha-file", type=Path, default=None)
    ap.add_argument("--t023-jsonl", type=Path, default=None)
    ap.add_argument("--max-sources", type=int, default=0, help="0 means all 100 dev sources")
    ap.add_argument("--max-gpus", type=int, default=4)
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--tutor-conc", type=int, default=16)
    ap.add_argument("--student-conc", type=int, default=32)
    args = ap.parse_args()

    os.umask(0o077)
    base = args.base.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    ensure_base_cache_dirs(base)

    eduardo_repo = locate_eduardo_repo(base)
    if str(eduardo_repo) not in sys.path:
        sys.path.insert(0, str(eduardo_repo))

    tutor_path = (args.tutor_model or (base / DEFAULT_ROUTEA_EXPORT_REL)).resolve()
    sha_file = (args.step100_sha_file or (base / DEFAULT_ROUTEA_SHA_REL)).resolve()
    t023_jsonl = (args.t023_jsonl or (base / "runs/t023_stage2_20261125_163319/routeA_stage2_measured.jsonl")).resolve()

    if sha256_file(REPO_ROOT / "training/routeA_reward/hybrid_grader.py") != HYBRID_GRADER_SHA256:
        raise RuntimeError("hybrid_grader.py SHA-256 mismatch")
    if sha256_file(t023_jsonl) != EXPECTED_T023_JSONL_SHA256:
        raise RuntimeError("T-023 JSONL SHA-256 mismatch")
    verify_dir_against_lock(tutor_path, ROUTEA_STEP100_LOCK, label="RouteA Step-100", manifest_file=sha_file)
    verify_tutor_bfloat16(tutor_path)

    student_model = resolve_local_model(base, "Meta-Llama-3.1-8B-Instruct", ("llama-3.1-8b", "8b-instruct"))
    grader_model = resolve_local_model(base, "Qwen3.6-27B", ("qwen3.6-27b", "27b"))

    dev_rows = load_dev_problems(t023_jsonl)
    if args.max_sources > 0:
        dev_rows = dev_rows[: args.max_sources]

    layout = wait_4_or_5_gpus(poll_seconds=args.poll_seconds, max_gpus=args.max_gpus)
    write_json_atomic(run_dir / "gpu_layout.json", layout)

    import openai

    token = secrets.token_urlsafe(32)
    env_base = build_clean_env(base, token=token, eduardo_repo=eduardo_repo)
    ports = pick_free_ports([19261, 19262, 19263, 19264])
    t_port, s0_port, s1_port, g_port = ports
    managed: list[tuple[str, subprocess.Popen]] = []

    def spawn(name: str, cmd: list[str], cuda_devs: str, log_name: str) -> subprocess.Popen:
        with (run_dir / log_name).open("ab") as log_f:
            proc = subprocess.Popen(
                cmd,
                env=dict(env_base, CUDA_VISIBLE_DEVICES=cuda_devs),
                cwd=run_dir,
                stdin=subprocess.DEVNULL,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        managed.append((name, proc))
        return proc

    t0 = time.monotonic()
    try:
        s_ports = [s0_port, s1_port][: len(layout["student_gpus"])]
        for idx, (gid, sp) in enumerate(zip(layout["student_gpus"], s_ports)):
            spawn(
                f"student_{idx}",
                [
                    sys.executable, "-B", "-m", "vllm.entrypoints.openai.api_server",
                    "--model", str(student_model), "--served-model-name", "student-local",
                    "--host", "127.0.0.1", "--port", str(sp),
                    "--dtype", "bfloat16", "--gpu-memory-utilization", "0.9",
                    "--max-model-len", "32768", "--max-num-seqs", "256",
                    "--tensor-parallel-size", "1", "--enable-chunked-prefill", "--enable-prefix-caching",
                ],
                str(gid),
                f"vllm_student_{idx}.log",
            )
        spawn(
            "grader_27b",
            [
                sys.executable, "-B", "-m", "vllm.entrypoints.openai.api_server",
                "--model", str(grader_model), "--served-model-name", "grader-27b",
                "--host", "127.0.0.1", "--port", str(g_port),
                "--dtype", "bfloat16", "--gpu-memory-utilization", "0.9",
                "--max-model-len", "32768", "--max-num-seqs", "256",
                "--tensor-parallel-size", "2", "--enable-chunked-prefill", "--enable-prefix-caching",
            ],
            ",".join(map(str, layout["grader_gpus"])),
            "vllm_grader_27b.log",
        )
        spawn(
            "tutor_routea",
            [
                sys.executable, "-B", "-m", "vllm.entrypoints.openai.api_server",
                "--model", str(tutor_path), "--served-model-name", "tutor-routeA-step100",
                "--host", "127.0.0.1", "--port", str(t_port),
                "--dtype", "bfloat16", "--gpu-memory-utilization", "0.88",
                "--max-model-len", str(TUTOR_VLLM_MAX_MODEL_LEN), "--max-num-seqs", "64",
                "--tensor-parallel-size", "1", "--enable-chunked-prefill", "--enable-prefix-caching",
            ],
            str(layout["tutor_gpu"]),
            "vllm_tutor_routea.log",
        )

        for name, proc in managed:
            if name.startswith("student_"):
                wait_for_health(f"http://127.0.0.1:{s_ports[int(name[-1])]}/health", proc, name, timeout_s=420)
            elif name == "grader_27b":
                wait_for_health(f"http://127.0.0.1:{g_port}/health", proc, name, timeout_s=600)
            elif name == "tutor_routea":
                wait_for_health(f"http://127.0.0.1:{t_port}/health", proc, name, timeout_s=420)

        async def _exec() -> list[dict[str, Any]]:
            tc = openai.AsyncClient(api_key=token, base_url=f"http://127.0.0.1:{t_port}/v1", timeout=300.0, max_retries=0)
            scs = [openai.AsyncClient(api_key=token, base_url=f"http://127.0.0.1:{p}/v1", timeout=300.0, max_retries=0) for p in s_ports]
            gc = openai.AsyncClient(api_key=token, base_url=f"http://127.0.0.1:{g_port}/v1", timeout=300.0, max_retries=0)
            try:
                return await run_dev_eval(
                    dev_rows=dev_rows,
                    tutor_path=tutor_path,
                    tutor_client=tc,
                    student_client=RoundRobinClientPool(scs),
                    grader_27b_client=gc,
                    grader_27b_model="grader-27b",
                    run_dir=run_dir,
                    tutor_conc=args.tutor_conc,
                    student_conc=args.student_conc,
                )
            finally:
                await tc.close()
                for c in scs:
                    await c.close()
                await gc.close()

        records = asyncio.run(_exec())
    finally:
        terminate_procs(managed)

    summary = summarize_dev_records(records, out_dir=run_dir, elapsed_s=time.monotonic() - t0)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(
        format_machine_summary(
            {
                "status": summary["status"],
                "rc": summary["rc"],
                "calls": summary["grader_llm_calls"],
                "failures": summary["grader_tech_failed"] + summary["student_failed"],
                "n_sources": summary["n_sources"],
                "theta_hat_k16": f"{summary['theta_hat_k16_mean']:+.6f}",
                "theta_hat_k8": f"{summary['theta_hat_k8_mean']:+.6f}",
                "reward_k8": f"{summary['reward_k8_mean']:+.6f}",
                "gated_rate": f"{summary['gated_rate']:.4f}",
            }
        ),
        flush=True,
    )
    return int(summary["rc"])


if __name__ == "__main__":
    raise SystemExit(main())
