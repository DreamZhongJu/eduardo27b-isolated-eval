#!/usr/bin/env python3
"""T-023 Stage 2 runner (Pro 5000): bf16 student pre-solve + hybrid grading + dev Agnes check + summary.

Phases (all resumable, atomic per-item checkpointing):
  - gpu       : Generate k=16 student rollouts per variant (exact author _solve_rate prompt,
                temp=0.6, max_tokens=1024, bf16) + run hybrid grading (author rule math-verify==0.8.0
                first, local Qwen3.6-27B TP=2 fallback using verbatim T-024 hybrid_grader.py
                sha256=7d2a5f46...). On dev split ONLY, also records 27B verdict on rule-pass
                responses (`dev_audit_27b`) so Step 6 can report exact 3-way Rule x 27B x Agnes
                consistency including rule false-positives.
  - agnes     : CPU-only (0 GPUs). Grade all non-failed dev-set responses with Agnes
                (`agnes-3.0-flash`, exact prompt from agnes_eval_v3/agnes_judge.py sha256=2ffe7464...,
                AIMD congestion window 1..16, >1% technical failure abort gate, atomic put_once).
  - summarize : Pure local aggregation. Computes smoothed p_new=(n_correct_hybrid+0.5)/(n_scored+1),
                Rasch b=ln((1-p)/p), ladder_rank (tie-break <1 count by generation_tier),
                writes private `routeA_stage2_measured.jsonl`, `stage2_manifest.json`, and
                sanitized `stage2_summary.json` / `stage2_summary.txt`.

PRIVACY GUARANTEE:
- Never prints or logs problem statements, reference answers, or student responses.
- Raw student rollouts and `routeA_stage2_measured.jsonl` remain strictly on Pro 5000 (`0700`).
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import importlib.metadata
import json
import math
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from routeA_data.hybrid_grader import (
    AGNES_JUDGE_REFERENCE_SHA256,
    DEFAULT_MAX_WINDOW,
    EQUIV_JUDGE_SYSTEM_PROMPT,
    AsyncAdaptiveWindow,
    GradeResult,
    HybridGrader,
    build_equiv_messages,
    canonical_json,
    check_technical_failure_rate,
    parse_equiv_verdict,
    strict_json,
)
from routeA_data.hybrid_grader_caller import (
    CANONICAL_HYBRID_GRADER_SHA256,
    TechnicalFailureGateError,
    TechnicalFailureTracker,
    assert_canonical_hybrid_grader,
)
from routeA_data.rasch_ladder import (
    GENERATION_TIER_ORDER,
    compute_smoothed_p_and_b,
    sort_ladder_variants,
)

# Exact student prompt from author's eduardo/create_transfer_dataset.py lines 66-68.
AUTHOR_SOLVE_PROMPT = (
    "Solve the following math problem step by step. Put your final answer in \\boxed{{}}.\n\n"
    "Problem: {problem}"
)

AGNES_ENDPOINT = "https://api.agnes-ai.cn/v1/chat/completions"
AGNES_MODEL = "agnes-3.0-flash"
AGNES_MAX_RESPONSE_BYTES = 262144
AGNES_MAX_ATTEMPTS = 4

REQUIRED_GRADER_PINS: Dict[str, str] = {
    "math-verify": "0.8.0",
    "latex2sympy2_extended": "1.10.2",
}

EXPECTED_STAGE1_V2_SHA256 = (
    "3bab0292bb6d2393a4f64cf8bf4dad282499b0bf395905c52dac9559b5f4f333"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = (json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def put_once_json(path: Path, obj: Any) -> bool:
    """Write JSON atomically without overwriting an existing receipt (returns True if newly written)."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        return False
    payload = (canonical_json(obj) + "\n").encode("utf-8")
    fd, tmp = tempfile.mkstemp(prefix=".receipt-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
            return True
        except FileExistsError:
            return False
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def verify_required_pins() -> Dict[str, str]:
    found: Dict[str, str] = {}
    bad: List[str] = []
    for pkg, want in REQUIRED_GRADER_PINS.items():
        try:
            got = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            got = "NOT_INSTALLED"
        found[pkg] = got
        if got != want:
            bad.append(f"{pkg}: got={got} want={want}")
    if bad:
        raise RuntimeError("Package version pin mismatch: " + "; ".join(bad))
    return found


def classify_answer_form(reference_answer: str) -> str:
    """Classify reference answer into coarse structural buckets without exposing text."""
    s = (reference_answer or "").strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    if re.fullmatch(r"[+-]?\d+", s):
        return "integer"
    if re.fullmatch(r"[+-]?\d+\.\d+", s):
        return "decimal"
    if re.fullmatch(r"[+-]?\\d?frac\{[+-]?\d+\}\{[+-]?\d+\}|[+-]?\d+/\d+", s):
        return "simple_fraction"
    if any(tok in s for tok in ("\\sqrt", "\\pi", "\\sin", "\\cos", "\\tan", "\\log", "\\ln", "^", "e^")):
        return "symbolic_or_radical"
    if any(ch in s for ch in (",", ";", "(", "[", "{", "\\cup", "\\infty", "\\le", "\\ge", "<", ">")):
        return "tuple_interval_or_set"
    if re.search(r"[a-zA-Z]{3,}", re.sub(r"\\[a-zA-Z]+", "", s)):
        return "text_or_units"
    return "other_expression"


def variant_key_hash(source_id: str, variant_id: str) -> str:
    return sha256_bytes(f"{source_id}::{variant_id}".encode("utf-8"))[:24]


def load_stage1_rows(path: Path, expected_sha256: Optional[str] = None) -> Tuple[List[Dict[str, Any]], str]:
    actual_sha = sha256_file(path)
    if expected_sha256 and actual_sha != expected_sha256.strip().lower():
        raise ValueError(f"Input JSONL SHA-256 mismatch: actual={actual_sha} expected={expected_sha256}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows, actual_sha


# ---------------------------------------------------------------------------
# Phase 1+2: Student rollouts (bf16, exact author _solve_rate) + Hybrid Grading
# ---------------------------------------------------------------------------


async def _sample_student_once(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.6,
    max_tokens: int = 1024,
    max_attempts: int = 4,
    sem: Optional[asyncio.Semaphore] = None,
    sleep_fn: Callable[[float], Any] = asyncio.sleep,
) -> Tuple[Optional[str], Optional[str], int]:
    """Sample one student rollout matching author's `_solve_rate`.

    Returns (text_or_none, finish_reason_or_error, attempts).
    """
    last_err = "unknown_error"
    for attempt in range(1, max_attempts + 1):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }
            if sem is not None:
                async with sem:
                    resp = await client.chat.completions.create(**kwargs)
            else:
                resp = await client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            text = choice.message.content if choice.message.content is not None else ""
            finish = getattr(choice, "finish_reason", "stop")
            return text, str(finish), attempt
        except Exception as exc:
            last_err = f"{type(exc).__name__}"
            if attempt < max_attempts:
                await sleep_fn(min(16.0, float(2 ** (attempt - 1))))
    return None, last_err, max_attempts


async def _judge_27b_direct_for_dev_audit(
    grader: HybridGrader,
    problem: str,
    reference_answer: str,
    student_response: str,
) -> GradeResult:
    """Invoke the same HybridGrader's 27B fallback directly (for dev rule-pass audit ONLY)."""
    # Pass a dummy rule_grader returning acc=0 so HybridGrader runs the EXACT same
    # 27B prompt, parser, retries, and AIMD window without touching hybrid_grader.py.
    audit_grader = HybridGrader(
        client=grader._client,
        model=grader._model,
        llm_caller=grader._llm_caller,
        rule_grader=lambda _s, _g: {"score": 0.0, "acc": 0.0, "pred": "", "incorrect_format": 0},
        max_attempts=grader._max_attempts,
        temperature=grader._temperature,
        max_tokens=grader._max_tokens,
        extra_body=grader._extra_body,
        use_vllm_no_thinking=False,
        window=grader._window,
        semaphore=grader._semaphore,
        sleep_fn=grader._sleep,
        backoff_base=grader._backoff_base,
    )
    return await audit_grader.grade(problem, reference_answer, student_response)


async def run_gpu_phase(
    rows: Sequence[Dict[str, Any]],
    *,
    run_dir: Path,
    student_client: Any,
    student_model: str,
    judge_client: Any,
    judge_model: str,
    k_samples: int = 16,
    student_temperature: float = 0.6,
    student_max_tokens: int = 1024,
    student_concurrency: int = 64,
    judge_max_window: int = DEFAULT_MAX_WINDOW,
    variant_concurrency: int = 16,
    limit_sources: Optional[int] = None,
    rule_grader: Optional[Callable[[str, str], Dict[str, Any]]] = None,
    llm_caller: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Generate k=16 student rollouts per variant and grade with canonical HybridGrader."""
    assert_canonical_hybrid_grader()

    resp_dir = run_dir / "responses"
    grade_dir = run_dir / "hybrid_grades"
    resp_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    grade_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    selected_rows = list(rows[:limit_sources]) if limit_sources is not None else list(rows)
    tasks_list: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for row in selected_rows:
        for var in row["variants"]:
            tasks_list.append((row, var))

    total_variants = len(tasks_list)
    student_sem = asyncio.Semaphore(student_concurrency)
    variant_sem = asyncio.Semaphore(variant_concurrency)
    judge_window = AsyncAdaptiveWindow(max_window=judge_max_window)

    grader_kwargs: Dict[str, Any] = {
        "client": judge_client if llm_caller is None else None,
        "model": judge_model,
        "llm_caller": llm_caller,
        "window": judge_window,
    }
    if rule_grader is not None:
        grader_kwargs["rule_grader"] = rule_grader
    grader = HybridGrader(**grader_kwargs)

    student_gate = TechnicalFailureTracker(max_percent=1, min_samples=200)
    judge_gate = TechnicalFailureTracker(max_percent=1, min_samples=100)
    state_lock = asyncio.Lock()

    stats = {
        "total_variants": total_variants,
        "variants_done": 0,
        "variants_resumed_resp": 0,
        "variants_resumed_grade": 0,
        "student_calls": 0,
        "student_failed": 0,
        "llm_calls": 0,
        "llm_tech_failed": 0,
        "llm_unknown": 0,
        "dev_rule_pass_27b_audit_calls": 0,
        "aborted": False,
        "abort_reason": None,
        "started_at": time.time(),
    }
    abort_event = asyncio.Event()

    async def process_one_variant(row: Dict[str, Any], var: Dict[str, Any]) -> None:
        if abort_event.is_set():
            return
        async with variant_sem:
            if abort_event.is_set():
                return
            sid = row["source_id"]
            vid = var["variant_id"]
            split = row["split"]
            vkey = variant_key_hash(sid, vid)
            resp_path = resp_dir / f"{vkey}.json"
            grade_path = grade_dir / f"{vkey}.json"

            # 1. Student rollouts (resumable)
            if resp_path.is_file():
                resp_doc = json.loads(resp_path.read_text(encoding="utf-8"))
                rollouts = resp_doc["rollouts"]
                async with state_lock:
                    stats["variants_resumed_resp"] += 1
            else:
                msgs = [
                    {
                        "role": "user",
                        "content": AUTHOR_SOLVE_PROMPT.format(problem=var["problem"]),
                    }
                ]
                raw_results = await asyncio.gather(
                    *[
                        _sample_student_once(
                            student_client,
                            student_model,
                            msgs,
                            temperature=student_temperature,
                            max_tokens=student_max_tokens,
                            sem=student_sem,
                        )
                        for _ in range(k_samples)
                    ]
                )
                rollouts = []
                n_fail_local = 0
                for idx, (text, finish, att) in enumerate(raw_results):
                    failed = text is None
                    if failed:
                        n_fail_local += 1
                    rollouts.append(
                        {
                            "rollout_index": idx,
                            "failed": failed,
                            "finish_reason": finish,
                            "attempts": att,
                            "text": text,
                        }
                    )
                async with state_lock:
                    for r in rollouts:
                        stats["student_calls"] += 1
                        if r["failed"]:
                            stats["student_failed"] += 1
                        try:
                            student_gate.record(technical_failure=bool(r["failed"]))
                        except TechnicalFailureGateError as exc:
                            stats["aborted"] = True
                            stats["abort_reason"] = f"student_failure_gate: {exc}"
                            abort_event.set()
                            return

                resp_doc = {
                    "source_id": sid,
                    "variant_id": vid,
                    "split": split,
                    "k_samples": k_samples,
                    "student_model": student_model,
                    "temperature": student_temperature,
                    "max_tokens": student_max_tokens,
                    "n_failed": n_fail_local,
                    "rollouts": rollouts,
                }
                atomic_write_json(resp_path, resp_doc)

            # 2. Hybrid grading (resumable)
            if grade_path.is_file():
                gdoc = json.loads(grade_path.read_text(encoding="utf-8"))
                # Verify dev audit is present if split == dev
                if split != "dev" or gdoc.get("dev_audit_27b_complete", False):
                    async with state_lock:
                        stats["variants_resumed_grade"] += 1
                        stats["variants_done"] += 1
                    return

            per_rollout: List[Dict[str, Any]] = []
            for r in rollouts:
                if abort_event.is_set():
                    return
                if r["failed"] or r["text"] is None:
                    per_rollout.append(
                        {
                            "rollout_index": r["rollout_index"],
                            "student_failed": True,
                            "rule_correct": None,
                            "hybrid_correct": None,
                            "llm_called": False,
                            "llm_unknown": False,
                            "llm_technical_failure": False,
                            "judge_27b_verdict": None,
                            "judge_27b_status": "student_failed",
                        }
                    )
                    continue

                gres = await grader.grade(var["problem"], var["answer"], r["text"])
                if gres.llm_called:
                    async with state_lock:
                        stats["llm_calls"] += 1
                        if gres.technical_failure:
                            stats["llm_tech_failed"] += 1
                        if gres.is_unknown:
                            stats["llm_unknown"] += 1
                        try:
                            judge_gate.record(technical_failure=gres.technical_failure)
                        except TechnicalFailureGateError as exc:
                            stats["aborted"] = True
                            stats["abort_reason"] = f"judge_27b_failure_gate: {exc}"
                            abort_event.set()
                            return

                # Determine 27B verdict for 3-way dev audit:
                # - If LLM was called by HybridGrader, we already have 27B's verdict!
                # - If split == "dev" and rule passed (so HybridGrader skipped 27B),
                #   run one audit call to 27B so Step 6 can check Rule-True vs 27B-False.
                j27_verdict: Optional[bool]
                j27_status: str
                if gres.llm_called:
                    j27_verdict = (
                        None
                        if (gres.is_unknown or gres.technical_failure)
                        else bool(gres.correct)
                    )
                    j27_status = gres.status
                elif split == "dev":
                    aud = await _judge_27b_direct_for_dev_audit(
                        grader, var["problem"], var["answer"], r["text"]
                    )
                    async with state_lock:
                        stats["dev_rule_pass_27b_audit_calls"] += 1
                        stats["llm_calls"] += 1
                        if aud.technical_failure:
                            stats["llm_tech_failed"] += 1
                        try:
                            judge_gate.record(technical_failure=aud.technical_failure)
                        except TechnicalFailureGateError as exc:
                            stats["aborted"] = True
                            stats["abort_reason"] = f"judge_27b_audit_failure_gate: {exc}"
                            abort_event.set()
                            return
                    j27_verdict = (
                        None
                        if (aud.is_unknown or aud.technical_failure)
                        else bool(aud.correct)
                    )
                    j27_status = aud.status
                else:
                    j27_verdict = None
                    j27_status = "skipped_rule_pass"

                per_rollout.append(
                    {
                        "rollout_index": r["rollout_index"],
                        "student_failed": False,
                        "rule_correct": bool(gres.rule_correct),
                        "rule_incorrect_format": int(gres.incorrect_format),
                        "hybrid_correct": None if gres.technical_failure else bool(gres.correct),
                        "hybrid_status": gres.status,
                        "hybrid_source": gres.source,
                        "llm_called": bool(gres.llm_called),
                        "llm_unknown": bool(gres.is_unknown),
                        "llm_technical_failure": bool(gres.technical_failure),
                        "attempts": int(gres.attempts),
                        "judge_27b_verdict": j27_verdict,
                        "judge_27b_status": j27_status,
                    }
                )

            n_failed = sum(
                1 for x in per_rollout if x["student_failed"] or x["llm_technical_failure"]
            )
            n_student_failed = sum(1 for x in per_rollout if x["student_failed"])
            n_llm_tech_failed = sum(1 for x in per_rollout if x["llm_technical_failure"])
            n_scored = len(per_rollout) - n_failed
            n_correct_rule = sum(
                1
                for x in per_rollout
                if (not x["student_failed"]) and (not x["llm_technical_failure"]) and x["rule_correct"]
            )
            n_correct_hybrid = sum(
                1
                for x in per_rollout
                if (not x["student_failed"]) and (not x["llm_technical_failure"]) and x["hybrid_correct"]
            )
            n_llm_called = sum(1 for x in per_rollout if x["llm_called"])
            n_unknown = sum(1 for x in per_rollout if x["llm_unknown"])

            grade_doc = {
                "source_id": sid,
                "variant_id": vid,
                "split": split,
                "hybrid_grader_sha256": CANONICAL_HYBRID_GRADER_SHA256,
                "n_samples": len(per_rollout),
                "n_scored": n_scored,
                "n_failed": n_failed,
                "n_student_failed": n_student_failed,
                "n_llm_tech_failed": n_llm_tech_failed,
                "n_correct_rule": n_correct_rule,
                "n_correct_hybrid": n_correct_hybrid,
                "n_llm_called": n_llm_called,
                "n_unknown": n_unknown,
                "dev_audit_27b_complete": (split == "dev"),
                "rollouts": per_rollout,
            }
            atomic_write_json(grade_path, grade_doc)

            async with state_lock:
                stats["variants_done"] += 1
                done = stats["variants_done"]
                if done % 25 == 0 or done == total_variants:
                    elapsed = max(0.001, time.time() - stats["started_at"])
                    rate = done / elapsed
                    eta_min = (total_variants - done) / rate / 60.0 if rate > 0 else 0.0
                    progress = {
                        **stats,
                        "judge_window": judge_window.window,
                        "elapsed_sec": round(elapsed, 1),
                        "variants_per_min": round(rate * 60.0, 2),
                        "eta_min": round(eta_min, 1),
                    }
                    atomic_write_json(run_dir / "gpu_phase_status.json", progress)
                    print(
                        f"[gpu_phase] {done}/{total_variants} variants "
                        f"({100.0 * done / total_variants:.1f}%) "
                        f"stu_fail={stats['student_failed']}/{stats['student_calls']} "
                        f"llm_fail={stats['llm_tech_failed']}/{stats['llm_calls']} "
                        f"win={judge_window.window} eta={eta_min:.1f}m",
                        flush=True,
                    )

    await asyncio.gather(*[process_one_variant(row, var) for row, var in tasks_list])

    if stats["aborted"]:
        atomic_write_json(run_dir / "gpu_phase_status.json", stats)
        raise TechnicalFailureGateError(str(stats["abort_reason"]))

    # Final strict <=1% technical failure checks
    student_gate.check_final()
    judge_gate.check_final()

    stats["completed_at"] = time.time()
    stats["elapsed_sec"] = round(stats["completed_at"] - stats["started_at"], 2)
    stats["status"] = "ok"
    atomic_write_json(run_dir / "gpu_phase_status.json", stats)
    (run_dir / "gpu_phase_done.flag").write_text("ok\n", encoding="utf-8")
    return stats


# ---------------------------------------------------------------------------
# Phase 3: Dev-set Agnes consistency check (CPU-only, 0 GPUs, AIMD 1..16)
# ---------------------------------------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _agnes_http_call_sync(body: Dict[str, Any], api_key: str) -> Tuple[int, Optional[Dict[str, Any]]]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    req = urllib.request.Request(
        AGNES_ENDPOINT,
        data=canonical_json(body).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with opener.open(req, timeout=120) as resp:
            raw = resp.read(AGNES_MAX_RESPONSE_BYTES + 1)
            if len(raw) > AGNES_MAX_RESPONSE_BYTES:
                return resp.status, None
            try:
                return resp.status, strict_json(raw.decode("utf-8"))
            except (ValueError, UnicodeError):
                return resp.status, None
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, None


async def run_agnes_dev_phase(
    rows: Sequence[Dict[str, Any]],
    *,
    run_dir: Path,
    api_key: str,
    max_window: int = 16,
    agnes_caller: Optional[Callable[[Dict[str, Any], str], Tuple[int, Optional[Dict[str, Any]]]]] = None,
    sleep_fn: Callable[[float], Any] = asyncio.sleep,
) -> Dict[str, Any]:
    """Run Agnes consistency grading on ALL non-failed dev-set responses ONLY."""
    caller = agnes_caller or _agnes_http_call_sync
    resp_dir = run_dir / "responses"
    rec_dir = run_dir / "agnes_dev_receipts"
    rec_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    # Collect ONLY dev split items
    dev_items: List[Tuple[str, str, int, str, str, str]] = []
    for row in rows:
        if row["split"] != "dev":
            continue
        sid = row["source_id"]
        for var in row["variants"]:
            vid = var["variant_id"]
            vkey = variant_key_hash(sid, vid)
            rpath = resp_dir / f"{vkey}.json"
            if not rpath.is_file():
                raise FileNotFoundError(f"Missing dev response file for {sid}/{vid}: {rpath}")
            rdoc = json.loads(rpath.read_text(encoding="utf-8"))
            for r in rdoc["rollouts"]:
                if r["failed"] or r["text"] is None:
                    continue
                dev_items.append(
                    (sid, vid, int(r["rollout_index"]), var["problem"], var["answer"], r["text"])
                )

    window = AsyncAdaptiveWindow(max_window=max_window)
    gate = TechnicalFailureTracker(max_percent=1, min_samples=100)
    lock = asyncio.Lock()
    abort_event = asyncio.Event()

    stats = {
        "total_dev_items": len(dev_items),
        "done": 0,
        "resumed": 0,
        "api_calls": 0,
        "http_429": 0,
        "tech_failed": 0,
        "semantic_unknown": 0,
        "max_window_reached": 1,
        "aborted": False,
        "abort_reason": None,
        "started_at": time.time(),
    }

    async def grade_one_dev(item: Tuple[str, str, int, str, str, str]) -> None:
        if abort_event.is_set():
            return
        sid, vid, ridx, prob, ref, resp_txt = item
        rid = sha256_bytes(f"dev::{sid}::{vid}::{ridx}".encode("utf-8"))[:28]
        rpath = rec_dir / f"{rid}.json"
        if rpath.is_file():
            async with lock:
                stats["resumed"] += 1
                stats["done"] += 1
            return

        body = {
            "model": AGNES_MODEL,
            "messages": build_equiv_messages(prob, ref, resp_txt),
            "temperature": 0,
            "max_tokens": 1024,
        }
        verdict: Optional[bool] = None
        status_tag = "technical_error"
        tech_failed = True
        attempts_used = 0

        for attempt in range(1, AGNES_MAX_ATTEMPTS + 1):
            if abort_event.is_set():
                return
            attempts_used = attempt
            await window.acquire()
            win_ok: Optional[bool] = False
            rate_limited = False
            retryable = False
            try:
                http_code, raw_obj = await asyncio.to_thread(caller, body, api_key)
                async with lock:
                    stats["api_calls"] += 1
                    if http_code == 429:
                        stats["http_429"] += 1
                if http_code == 200 and isinstance(raw_obj, dict):
                    if raw_obj.get("model") != AGNES_MODEL:
                        verdict, issue = None, "model_mismatch"
                    else:
                        try:
                            ch = raw_obj["choices"][0]
                            content = ch["message"]["content"]
                            finish = ch.get("finish_reason", "stop")
                        except (KeyError, IndexError, TypeError):
                            verdict, issue = None, "response_structure_invalid"
                        else:
                            verdict, issue = parse_equiv_verdict(content, finish)
                    if type(verdict) is bool:
                        win_ok = True
                        tech_failed = False
                        status_tag = "correct" if verdict else "incorrect"
                    elif issue is None:
                        win_ok = None
                        tech_failed = False
                        status_tag = "unknown"
                    else:
                        win_ok = None
                        retryable = True
                        status_tag = issue
                else:
                    win_ok = False
                    rate_limited = http_code == 429
                    retryable = http_code in (0, 408, 429) or (500 <= http_code <= 599)
                    status_tag = f"http_{http_code}"
            finally:
                await window.release(win_ok, rate_limited=rate_limited)
                async with lock:
                    if window.window > stats["max_window_reached"]:
                        stats["max_window_reached"] = window.window

            if not tech_failed:
                break
            if not retryable:
                break
            if attempt < AGNES_MAX_ATTEMPTS:
                await sleep_fn(min(30.0, float(2 ** (attempt - 1))))

        # If retries exhausted on content/format issue (HTTP 200), treat as unknown (wrong), not transport error
        if tech_failed and status_tag.startswith("response_"):
            tech_failed = False
            verdict = None
            status_tag = "unknown"

        receipt = {
            "source_id": sid,
            "variant_id": vid,
            "rollout_index": ridx,
            "model": AGNES_MODEL,
            "verdict": verdict,
            "status": status_tag,
            "technical_failure": tech_failed,
            "attempts": attempts_used,
        }
        put_once_json(rpath, receipt)

        async with lock:
            stats["done"] += 1
            if tech_failed:
                stats["tech_failed"] += 1
            elif verdict is None:
                stats["semantic_unknown"] += 1
            try:
                gate.record(technical_failure=tech_failed)
            except TechnicalFailureGateError as exc:
                stats["aborted"] = True
                stats["abort_reason"] = f"agnes_failure_gate: {exc}"
                abort_event.set()
                return
            if stats["done"] % 100 == 0 or stats["done"] == stats["total_dev_items"]:
                elapsed = max(0.001, time.time() - stats["started_at"])
                atomic_write_json(
                    run_dir / "agnes_phase_status.json",
                    {**stats, "current_window": window.window, "elapsed_sec": round(elapsed, 1)},
                )
                print(
                    f"[agnes_dev] {stats['done']}/{stats['total_dev_items']} "
                    f"tech_fail={stats['tech_failed']} unk={stats['semantic_unknown']} "
                    f"429={stats['http_429']} win={window.window}",
                    flush=True,
                )

    sem = asyncio.Semaphore(max_window)

    async def bounded(item: Tuple[str, str, int, str, str, str]) -> None:
        async with sem:
            await grade_one_dev(item)

    await asyncio.gather(*[bounded(it) for it in dev_items])

    if stats["aborted"]:
        atomic_write_json(run_dir / "agnes_phase_status.json", stats)
        raise TechnicalFailureGateError(str(stats["abort_reason"]))

    gate.check_final()
    stats["completed_at"] = time.time()
    stats["elapsed_sec"] = round(stats["completed_at"] - stats["started_at"], 2)
    stats["status"] = "ok"
    atomic_write_json(run_dir / "agnes_phase_status.json", stats)
    (run_dir / "agnes_phase_done.flag").write_text("ok\n", encoding="utf-8")
    return stats


# ---------------------------------------------------------------------------
# Phase 4: Summarize, write back enriched JSONL, compute all T-023 Stage 2 metrics
# ---------------------------------------------------------------------------


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0.0 or dy == 0.0:
        return None
    return num / (dx * dy)


def _ranks(vals: Sequence[float]) -> List[float]:
    order = sorted(enumerate(vals), key=lambda kv: kv[1])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][1] == order[i][1]:
            j += 1
        avg_r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k][0]] = avg_r
        i = j + 1
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    return _pearson(_ranks(xs), _ranks(ys))


def _quantiles(vals: Sequence[float]) -> Dict[str, Optional[float]]:
    if not vals:
        return {"count": 0, "mean": None, "std": None, "min": None, "p10": None, "p25": None, "p50": None, "p75": None, "p90": None, "max": None}
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / n

    def q(prob: float) -> float:
        idx = prob * (n - 1)
        lo = int(math.floor(idx))
        hi = min(n - 1, lo + 1)
        frac = idx - lo
        return s[lo] * (1.0 - frac) + s[hi] * frac

    return {
        "count": n,
        "mean": round(mean, 6),
        "std": round(math.sqrt(var), 6),
        "min": round(s[0], 6),
        "p10": round(q(0.10), 6),
        "p25": round(q(0.25), 6),
        "p50": round(q(0.50), 6),
        "p75": round(q(0.75), 6),
        "p90": round(q(0.90), 6),
        "max": round(s[-1], 6),
    }


def _cohen_kappa_binary(a: Sequence[bool], b: Sequence[bool]) -> Optional[float]:
    n = len(a)
    if n == 0:
        return None
    agree = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(1 for x in a if x) / n
    pb1 = sum(1 for y in b if y) / n
    pe = pa1 * pb1 + (1.0 - pa1) * (1.0 - pb1)
    if pe >= 1.0:
        return 1.0 if agree == 1.0 else 0.0
    return (agree - pe) / (1.0 - pe)


def run_summarize_phase(
    rows: Sequence[Dict[str, Any]],
    *,
    run_dir: Path,
    input_sha256: str,
    student_model_desc: str = "Meta-Llama-3.1-8B-Instruct (bf16)",
    judge_model_desc: str = "Qwen3.6-27B (bf16, TP=2)",
    require_agnes: bool = True,
) -> Dict[str, Any]:
    """Assemble enriched JSONL, manifest, and full sanitized Stage 2 report."""
    hybrid_sha = assert_canonical_hybrid_grader()
    grade_dir = run_dir / "hybrid_grades"
    agnes_dir = run_dir / "agnes_dev_receipts"

    # Load all Agnes dev receipts if present
    agnes_map: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    if agnes_dir.is_dir():
        for p in agnes_dir.glob("*.json"):
            rec = json.loads(p.read_text(encoding="utf-8"))
            agnes_map[(rec["source_id"], rec["variant_id"], int(rec["rollout_index"]))] = rec

    enriched_rows: List[Dict[str, Any]] = []
    p_all: List[float] = []
    p_train: List[float] = []
    p_dev: List[float] = []
    p_self: List[float] = []
    p_paper: List[float] = []
    b_all: List[float] = []
    old_p_pairs: List[Tuple[float, float]] = []
    paper_p_pairs: List[Tuple[float, float]] = []

    tot_rollouts_scored = 0
    tot_rule_correct = 0
    tot_hybrid_correct = 0
    tot_llm_called = 0
    tot_unknown = 0
    tot_failed = 0
    variants_promoted = 0

    multi_sources_total = 0
    sources_tie_broken_by_tag = 0
    tag_comparable_pairs = 0
    tag_inverted_pairs = 0
    low_disc_all_easy = 0  # all variants n_correct_hybrid >= 15
    low_disc_all_hard = 0  # all variants n_correct_hybrid <= 1
    low_disc_small_b_span = 0  # max(b) - min(b) < 0.5 (among multi-variant)
    low_disc_small_cnt_span = 0  # max(n_correct) - min(n_correct) <= 1

    # Dev 3-way consistency accumulators
    dev_total = 0
    dev_rule_false = 0
    dev_rule_false_27b_or_agnes_true = 0
    dev_rule_false_27b_true = 0
    dev_rule_false_agnes_true = 0
    dev_rule_false_by_form: Dict[str, Dict[str, int]] = collections.defaultdict(
        lambda: {"rule_false": 0, "either_true": 0, "q27b_true": 0, "agnes_true": 0}
    )
    dev_rule_true = 0
    dev_rule_true_both_false = 0
    dev_rule_true_27b_false = 0
    dev_rule_true_agnes_false = 0
    dev_pair_27b_agnes_strict: List[Tuple[bool, bool]] = []
    dev_pair_27b_agnes_unk_as_false: List[Tuple[bool, bool]] = []
    dev_27b_agnes_matrix: collections.Counter[str] = collections.Counter()
    dev_agnes_missing = 0
    dev_agnes_tech_failed = 0

    for row in rows:
        sid = row["source_id"]
        split = row["split"]
        enriched_variants: List[Dict[str, Any]] = []
        for var in row["variants"]:
            vid = var["variant_id"]
            vkey = variant_key_hash(sid, vid)
            gpath = grade_dir / f"{vkey}.json"
            if not gpath.is_file():
                raise FileNotFoundError(f"Missing hybrid grade file for {sid}/{vid}: {gpath}")
            gdoc = json.loads(gpath.read_text(encoding="utf-8"))

            n_scored = int(gdoc["n_scored"])
            n_failed = int(gdoc["n_failed"])
            n_rule = int(gdoc["n_correct_rule"])
            n_hyb = int(gdoc["n_correct_hybrid"])
            n_llm = int(gdoc["n_llm_called"])
            n_unk = int(gdoc["n_unknown"])

            p_new, b_val = compute_smoothed_p_and_b(n_hyb, n_scored)
            p_all.append(p_new)
            b_all.append(b_val)
            (p_train if split == "train" else p_dev).append(p_new)
            (p_self if var.get("origin") == "self_generated" else p_paper).append(p_new)

            old_pre = var.get("old_pre_solve") or {}
            if old_pre.get("pass_rate") is not None:
                old_p_pairs.append((float(old_pre["pass_rate"]), p_new))
            if old_pre.get("paper_rate") is not None:
                paper_p_pairs.append((float(old_pre["paper_rate"]), p_new))

            tot_rollouts_scored += n_scored
            tot_rule_correct += n_rule
            tot_hybrid_correct += n_hyb
            tot_llm_called += n_llm
            tot_unknown += n_unk
            tot_failed += n_failed
            if n_hyb > n_rule:
                variants_promoted += 1

            # Dev 3-way consistency check
            if split == "dev":
                form = classify_answer_form(var["answer"])
                for r in gdoc["rollouts"]:
                    if r["student_failed"] or r["llm_technical_failure"]:
                        continue
                    arec = agnes_map.get((sid, vid, int(r["rollout_index"])))
                    if arec is None:
                        dev_agnes_missing += 1
                        continue
                    if arec.get("technical_failure"):
                        dev_agnes_tech_failed += 1
                        continue
                    dev_total += 1
                    r_ok = bool(r["rule_correct"])
                    j27_raw = r["judge_27b_verdict"]  # True / False / None
                    ag_raw = arec["verdict"]  # True / False / None
                    j27_eff = bool(j27_raw is True)  # unknown -> False
                    ag_eff = bool(ag_raw is True)  # unknown -> False

                    dev_27b_agnes_matrix[f"27b={j27_raw}|agnes={ag_raw}"] += 1
                    dev_pair_27b_agnes_unk_as_false.append((j27_eff, ag_eff))
                    if isinstance(j27_raw, bool) and isinstance(ag_raw, bool):
                        dev_pair_27b_agnes_strict.append((j27_raw, ag_raw))

                    if not r_ok:
                        dev_rule_false += 1
                        dev_rule_false_by_form[form]["rule_false"] += 1
                        if j27_eff:
                            dev_rule_false_27b_true += 1
                            dev_rule_false_by_form[form]["q27b_true"] += 1
                        if ag_eff:
                            dev_rule_false_agnes_true += 1
                            dev_rule_false_by_form[form]["agnes_true"] += 1
                        if j27_eff or ag_eff:
                            dev_rule_false_27b_or_agnes_true += 1
                            dev_rule_false_by_form[form]["either_true"] += 1
                    else:
                        dev_rule_true += 1
                        if not j27_eff:
                            dev_rule_true_27b_false += 1
                        if not ag_eff:
                            dev_rule_true_agnes_false += 1
                        if (not j27_eff) and (not ag_eff):
                            dev_rule_true_both_false += 1

            ev = dict(var)
            ev.update(
                {
                    "n_correct_rule": n_rule,
                    "n_correct_hybrid": n_hyb,
                    "n_llm_called": n_llm,
                    "n_unknown": n_unk,
                    "n_failed": n_failed,
                    "n_scored": n_scored,
                    "p_new": round(p_new, 8),
                    "b": round(b_val, 8),
                }
            )
            enriched_variants.append(ev)

        sorted_vars, tie_broken = sort_ladder_variants(enriched_variants)
        if tie_broken:
            sources_tie_broken_by_tag += 1

        if len(sorted_vars) >= 2:
            multi_sources_total += 1
            bs = [float(v["b"]) for v in sorted_vars]
            cnts = [int(v["n_correct_hybrid"]) for v in sorted_vars]
            if all(c >= 15 for c in cnts):
                low_disc_all_easy += 1
            if all(c <= 1 for c in cnts):
                low_disc_all_hard += 1
            if (max(bs) - min(bs)) < 0.5:
                low_disc_small_b_span += 1
            if (max(cnts) - min(cnts)) <= 1:
                low_disc_small_cnt_span += 1

            # Check pairwise order disagreement vs generation_tier (only pairs with strictly ordered tiers & non-tied p)
            for i in range(len(sorted_vars)):
                for j in range(i + 1, len(sorted_vars)):
                    vi, vj = sorted_vars[i], sorted_vars[j]
                    ti = GENERATION_TIER_ORDER.get(str(vi.get("generation_tier")), 99)
                    tj = GENERATION_TIER_ORDER.get(str(vj.get("generation_tier")), 99)
                    if ti != tj and ti < 99 and tj < 99:
                        tag_comparable_pairs += 1
                        # sorted_vars[i] has lower ladder_rank (easier), so tag expects ti < tj
                        if ti > tj:
                            tag_inverted_pairs += 1

        erow = dict(row)
        erow["variants"] = sorted_vars
        erow["ladder_tie_broken_by_tag"] = tie_broken
        enriched_rows.append(erow)

    if require_agnes and dev_agnes_missing > 0:
        raise RuntimeError(f"Agnes dev receipts incomplete: missing={dev_agnes_missing}")

    # Write enriched private JSONL
    out_jsonl = run_dir / "routeA_stage2_measured.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for erow in enriched_rows:
            f.write(json.dumps(erow, ensure_ascii=False) + "\n")
    os.chmod(out_jsonl, 0o600)
    out_jsonl_sha = sha256_file(out_jsonl)

    # 27B vs Agnes agreement stats
    n_pair_all = len(dev_pair_27b_agnes_unk_as_false)
    agree_all = sum(1 for a, b in dev_pair_27b_agnes_unk_as_false if a == b)
    q27_yes_ag_no = sum(1 for a, b in dev_pair_27b_agnes_unk_as_false if a and (not b))
    q27_no_ag_yes = sum(1 for a, b in dev_pair_27b_agnes_unk_as_false if (not a) and b)

    n_pair_strict = len(dev_pair_27b_agnes_strict)
    agree_strict = sum(1 for a, b in dev_pair_27b_agnes_strict if a == b)

    summary = {
        "task": "T-023",
        "stage": "stage2",
        "card_version": 3,
        "inbox_read_up_to": 3,
        "student_model": student_model_desc,
        "judge_model": judge_model_desc,
        "hybrid_grader_sha256": hybrid_sha,
        "agnes_judge_prompt_reference_sha256": AGNES_JUDGE_REFERENCE_SHA256,
        "input_dataset_sha256": input_sha256,
        "output_measured_jsonl_sha256": out_jsonl_sha,
        "output_measured_jsonl_bytes": out_jsonl.stat().st_size,
        "counts": {
            "n_sources": len(enriched_rows),
            "n_variants": len(p_all),
            "n_rollouts_scored": tot_rollouts_scored,
            "n_rollouts_failed": tot_failed,
            "n_correct_rule": tot_rule_correct,
            "n_correct_hybrid": tot_hybrid_correct,
            "n_promoted_rollouts_hybrid_over_rule": tot_hybrid_correct - tot_rule_correct,
            "hybrid_gain_pct_points": round(
                100.0 * (tot_hybrid_correct - tot_rule_correct) / tot_rollouts_scored, 4
            )
            if tot_rollouts_scored
            else 0.0,
            "variants_with_hybrid_promotion": variants_promoted,
            "n_llm_called": tot_llm_called,
            "n_unknown": tot_unknown,
        },
        "p_new_distribution": {
            "all": _quantiles(p_all),
            "train": _quantiles(p_train),
            "dev": _quantiles(p_dev),
            "self_generated": _quantiles(p_self),
            "paper_transfer": _quantiles(p_paper),
        },
        "b_distribution": _quantiles(b_all),
        "correlation_with_old_pre_solve": {
            "n_pairs_old_fp16": len(old_p_pairs),
            "pearson_old_fp16": round(_pearson([x for x, _ in old_p_pairs], [y for _, y in old_p_pairs]), 6)
            if _pearson([x for x, _ in old_p_pairs], [y for _, y in old_p_pairs]) is not None
            else None,
            "spearman_old_fp16": round(_spearman([x for x, _ in old_p_pairs], [y for _, y in old_p_pairs]), 6)
            if _spearman([x for x, _ in old_p_pairs], [y for _, y in old_p_pairs]) is not None
            else None,
            "n_pairs_paper_rate": len(paper_p_pairs),
            "pearson_paper_rate": round(_pearson([x for x, _ in paper_p_pairs], [y for _, y in paper_p_pairs]), 6)
            if _pearson([x for x, _ in paper_p_pairs], [y for _, y in paper_p_pairs]) is not None
            else None,
            "spearman_paper_rate": round(_spearman([x for x, _ in paper_p_pairs], [y for _, y in paper_p_pairs]), 6)
            if _spearman([x for x, _ in paper_p_pairs], [y for _, y in paper_p_pairs]) is not None
            else None,
        },
        "dev_three_way_consistency": {
            "dev_rollouts_evaluated": dev_total,
            "dev_agnes_missing": dev_agnes_missing,
            "dev_agnes_tech_failed": dev_agnes_tech_failed,
            "rule_false_negatives": {
                "n_rule_false": dev_rule_false,
                "n_27b_or_agnes_true": dev_rule_false_27b_or_agnes_true,
                "rate_27b_or_agnes_true": round(dev_rule_false_27b_or_agnes_true / dev_rule_false, 6)
                if dev_rule_false
                else 0.0,
                "n_27b_true": dev_rule_false_27b_true,
                "n_agnes_true": dev_rule_false_agnes_true,
                "by_answer_form": dict(sorted(dev_rule_false_by_form.items())),
            },
            "rule_false_positives": {
                "n_rule_true": dev_rule_true,
                "n_both_27b_and_agnes_false": dev_rule_true_both_false,
                "rate_both_false": round(dev_rule_true_both_false / dev_rule_true, 6)
                if dev_rule_true
                else 0.0,
                "n_27b_false": dev_rule_true_27b_false,
                "n_agnes_false": dev_rule_true_agnes_false,
            },
            "qwen27b_vs_agnes": {
                "n_total_unk_as_false": n_pair_all,
                "agreement_rate_unk_as_false": round(agree_all / n_pair_all, 6) if n_pair_all else None,
                "cohen_kappa_unk_as_false": round(
                    _cohen_kappa_binary(
                        [a for a, _ in dev_pair_27b_agnes_unk_as_false],
                        [b for _, b in dev_pair_27b_agnes_unk_as_false],
                    ),
                    6,
                )
                if n_pair_all
                else None,
                "n_27b_true_agnes_false": q27_yes_ag_no,
                "n_27b_false_agnes_true": q27_no_ag_yes,
                "n_strict_both_bool": n_pair_strict,
                "agreement_rate_strict_bool": round(agree_strict / n_pair_strict, 6)
                if n_pair_strict
                else None,
                "confusion_matrix": dict(sorted(dev_27b_agnes_matrix.items())),
            },
        },
        "ladder_diagnostics": {
            "multi_variant_sources": multi_sources_total,
            "sources_tie_broken_by_tag": sources_tie_broken_by_tag,
            "tag_comparable_pairs": tag_comparable_pairs,
            "tag_inverted_pairs": tag_inverted_pairs,
            "tag_inversion_rate": round(tag_inverted_pairs / tag_comparable_pairs, 6)
            if tag_comparable_pairs
            else 0.0,
            "low_discriminability_all_easy_ge15": low_disc_all_easy,
            "low_discriminability_all_hard_le1": low_disc_all_hard,
            "low_discriminability_b_span_lt_0_5": low_disc_small_b_span,
            "low_discriminability_count_span_le_1": low_disc_small_cnt_span,
        },
    }

    atomic_write_json(run_dir / "stage2_summary.json", summary)
    manifest = {
        "task": "T-023",
        "stage": "stage2",
        "input_dataset_sha256": input_sha256,
        "output_measured_jsonl_sha256": out_jsonl_sha,
        "output_measured_jsonl_bytes": out_jsonl.stat().st_size,
        "hybrid_grader_sha256": hybrid_sha,
        "stage2_summary_sha256": sha256_file(run_dir / "stage2_summary.json"),
    }
    atomic_write_json(run_dir / "stage2_manifest.json", manifest)

    lines = [
        "=== T-023 STAGE 2 SANITIZED SUMMARY (ZERO TEXT) ===",
        f"hybrid_grader_sha256={hybrid_sha}",
        f"input_sha256={input_sha256}",
        f"output_measured_jsonl_sha256={out_jsonl_sha}",
        f"sources={summary['counts']['n_sources']} variants={summary['counts']['n_variants']} scored_rollouts={tot_rollouts_scored} failed={tot_failed}",
        f"rule_correct={tot_rule_correct} hybrid_correct={tot_hybrid_correct} promoted=+{tot_hybrid_correct - tot_rule_correct} (+{summary['counts']['hybrid_gain_pct_points']:.2f}pp) llm_called={tot_llm_called} unknown={tot_unknown}",
        f"p_new_all={summary['p_new_distribution']['all']}",
        f"corr_old_fp16={summary['correlation_with_old_pre_solve']}",
        f"dev_3way_fn={summary['dev_three_way_consistency']['rule_false_negatives']}",
        f"dev_3way_fp={summary['dev_three_way_consistency']['rule_false_positives']}",
        f"dev_27b_vs_agnes={summary['dev_three_way_consistency']['qwen27b_vs_agnes']}",
        f"ladder={summary['ladder_diagnostics']}",
    ]
    (run_dir / "stage2_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "STAGE2_DONE.flag").write_text("ok\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="T-023 Stage 2 runner (gpu / agnes / summarize).",
        allow_abbrev=False,
    )
    parser.add_argument("--phase", choices=["gpu", "agnes", "summarize"], required=True)
    parser.add_argument("--data-jsonl", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", type=str, default=EXPECTED_STAGE1_V2_SHA256)
    parser.add_argument("--student-api-base", type=str, default="http://127.0.0.1:18981/v1")
    parser.add_argument("--student-model", type=str, default="Llama-3.1-8B-Instruct-t023")
    parser.add_argument("--judge-api-base", type=str, default="http://127.0.0.1:18982/v1")
    parser.add_argument("--judge-model", type=str, default="judge-local")
    parser.add_argument("--serving-api-key", type=str, default=os.environ.get("SERVING_API_KEY", "EMPTY"))
    parser.add_argument("--k-samples", type=int, default=16)
    parser.add_argument("--student-concurrency", type=int, default=64)
    parser.add_argument("--judge-max-window", type=int, default=16)
    parser.add_argument("--agnes-max-window", type=int, default=16)
    parser.add_argument("--variant-concurrency", type=int, default=16)
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--skip-pin-check", action="store_true")
    args = parser.parse_args()

    args.run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    rows, input_sha = load_stage1_rows(args.data_jsonl, args.expected_input_sha256)

    if args.phase == "gpu":
        if not args.skip_pin_check:
            pins = verify_required_pins()
            print(f"[stage2] verified pins={pins}", flush=True)
        import httpx
        import openai

        http_client = httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(180.0))
        student_client = openai.AsyncOpenAI(
            base_url=args.student_api_base,
            api_key=args.serving_api_key,
            http_client=http_client,
        )
        judge_client = openai.AsyncOpenAI(
            base_url=args.judge_api_base,
            api_key=args.serving_api_key,
            http_client=http_client,
        )

        async def _run() -> Dict[str, Any]:
            try:
                return await run_gpu_phase(
                    rows,
                    run_dir=args.run_dir,
                    student_client=student_client,
                    student_model=args.student_model,
                    judge_client=judge_client,
                    judge_model=args.judge_model,
                    k_samples=args.k_samples,
                    student_concurrency=args.student_concurrency,
                    judge_max_window=args.judge_max_window,
                    variant_concurrency=args.variant_concurrency,
                    limit_sources=args.limit_sources,
                )
            finally:
                await http_client.aclose()

        res = asyncio.run(_run())
        print(json.dumps(res, ensure_ascii=False, sort_keys=True), flush=True)
        return 0

    if args.phase == "agnes":
        api_key = os.environ.get("AGNES_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("ERROR: AGNES_API_KEY not set")
        res = asyncio.run(
            run_agnes_dev_phase(
                rows,
                run_dir=args.run_dir,
                api_key=api_key,
                max_window=args.agnes_max_window,
            )
        )
        print(json.dumps(res, ensure_ascii=False, sort_keys=True), flush=True)
        return 0

    if args.phase == "summarize":
        res = run_summarize_phase(
            rows,
            run_dir=args.run_dir,
            input_sha256=input_sha,
        )
        print((args.run_dir / "stage2_summary.txt").read_text(encoding="utf-8"), flush=True)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
