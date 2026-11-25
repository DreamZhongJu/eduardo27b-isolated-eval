"""Unit tests for T-023 Route A data pipeline using synthetic fake data only.

Covers every item in T-023 Acceptance Criterion 1:
1. Problem text normalization and hash deduplication (whitespace, NFKC full/half-width, LaTeX spacing).
2. Test-set exclusion removing entire source group when source OR any variant matches test set.
3. Reproducible stratified sampling (same seed -> identical result; balanced strata).
4. Zero overlap between train and dev splits.
5. Rasch smoothed pass rate p and difficulty b calculation (including boundary & n_failed).
6. Ladder sorting by p descending with generation-tag fallback when counts differ by < 1.
7. Hybrid grading: rule pass skips LLM, rule fail calls LLM, null/malformed counts as wrong+unknown,
   technical failure retries and is NOT counted as a verdict, AIMD window & >1% failure gate.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import pytest

from routeA_data.hybrid_grader import (
    AIMDWindow,
    FailureRateGate,
    HybridGrader,
    TechnicalFailureError,
    TechnicalFailureGateError,
    build_judge_payload,
    parse_judge_response,
)
from routeA_data.normalize import normalize_problem_text, problem_text_hash
from routeA_data.rasch_ladder import (
    compute_smoothed_p_and_b,
    infer_generation_tier,
    sort_ladder_variants,
)
from routeA_data.stage1_select import (
    build_test_hash_index,
    deduplicate_sources,
    exclude_test_overlaps,
    stratified_train_dev_split,
)


def _fake_group(
    sid: str,
    direction: str,
    n_variants: int,
    *,
    source_problem: Optional[str] = None,
    variant_problems: Optional[List[str]] = None,
) -> Dict[str, Any]:
    sp = source_problem or f"Compute synthetic value {sid}: x + 1 = 2."
    variants = []
    for idx in range(n_variants):
        vp = (
            variant_problems[idx]
            if variant_problems and idx < len(variant_problems)
            else f"Synthetic variant {sid}_v{idx}: compute 2x + {idx}."
        )
        variants.append(
            {
                "variant_id": f"{sid}_v{idx}",
                "origin": "self_generated" if idx == 0 else "paper_transfer",
                "problem": vp,
                "answer": str(idx + 1),
                "problem_text_sha256": problem_text_hash(vp),
                "generation_tag": "reduce non-core preparation" if idx == 0 else "paper_transfer",
                "generation_tier": "easier" if idx == 0 else "paper_transfer",
                "old_pre_solve": {"correct": 8, "known_n": 16, "pass_rate": 0.5},
            }
        )
    return {
        "source_id": sid,
        "station": "one",
        "direction": direction,
        "source_problem": sp,
        "source_answer": "1",
        "source_text_sha256": problem_text_hash(sp),
        "source_pre_solve_rate_old": 0.25,
        "source_llama8b_solve_rate_old": 0.25,
        "variants": variants,
    }


# ---------------------------------------------------------------------------
# 1. Problem normalization & hash deduplication
# ---------------------------------------------------------------------------


def test_normalize_whitespace_fullwidth_and_latex_spaces():
    a = r"Find $x\, + \; y\quad = \qquad 10$."
    b = "Find $x+y=10$."
    c = "Ｆｉｎｄ　＄ｘ＋ｙ＝１０＄．"
    d = r"Find $x \hspace{2em} + ~ y \  = \thinspace 10$."

    assert normalize_problem_text(a) == normalize_problem_text(b)
    assert normalize_problem_text(b) == normalize_problem_text(c)
    assert normalize_problem_text(b) == normalize_problem_text(d)
    assert problem_text_hash(a) == problem_text_hash(b) == problem_text_hash(c) == problem_text_hash(d)

    different = "Find $x+y=11$."
    assert problem_text_hash(a) != problem_text_hash(different)


def test_deduplicate_sources_merges_overlay_without_duplicating_candidates():
    def _make_transfer(cid: str, q: str, origin: str = "self_generated") -> Dict[str, Any]:
        return {
            "candidate_id": cid,
            "origin": origin,
            "question": q,
            "answer": "42",
            "complete": True,
            "known_n": 16,
            "correct": 10,
            "pass_rate": 10 / 16,
            "direction": "reduce non-core preparation" if origin == "self_generated" else None,
            "quality": {
                "quality_status": "provisional_pass",
                "model_review": {
                    "solvability": "solvable",
                    "core_relation": "related",
                    "direct_repeat": "no",
                },
                "answer_equivalence": {"status": "correct"},
            },
        }

    raw = [
        {
            "_station": "one",
            "_member": "one/old/shard_000/COMPLETE_TRAIN.jsonl",
            "source": {
                "source_id": "S1",
                "direction_band": "easier",
                "problem": "Fake source S1",
                "answer": "1",
                "pre_solve_rate": 0.125,
                "llama8b_solve_rate": 0.125,
            },
            "transfers": [_make_transfer("C1", "Fake var 1")],
        },
        {
            "_station": "one",
            "_member": "one/legacy_overlay/shard_040/COMPLETE_TRAIN.jsonl",
            "source": {
                "source_id": "S1",
                "direction_band": "easier",
                "problem": "Fake source S1",
                "answer": "1",
                "pre_solve_rate": 0.125,
                "llama8b_solve_rate": 0.125,
            },
            "transfers": [
                _make_transfer("C1", "Fake var 1"),
                _make_transfer("C2", "Fake var 2", origin="paper_transfer"),
            ],
        },
    ]
    groups, stats = deduplicate_sources(raw)
    assert len(groups) == 1
    assert len(groups[0]["variants"]) == 2
    assert stats["duplicate_source_rows"] == 1
    assert stats["duplicate_candidate_rows"] == 1


# ---------------------------------------------------------------------------
# 2. Test-set exclusion removes entire source group on any hit
# ---------------------------------------------------------------------------


def test_exclude_test_overlaps_removes_whole_group():
    self_test_rows = [
        {
            "source_problem": r"Self test source $a\,+\,b=3$",
            "transfer_questions": [{"problem": r"Self test variant $2a + 2b = 6$"}],
        }
    ]
    author_test_rows = [
        {
            "problem": r"Author test source $m \quad = 5$",
            "transfer_problem": r"Author test transfer $n \; = 7$",
        }
    ]
    test_index = build_test_hash_index(self_test_rows, author_test_rows)

    groups = [
        # Hits self499 source (with full-width / spacing difference)
        _fake_group("G_hit_src_self", "easier", 2, source_problem="Self test source $a+b=３$"),
        # Source is clean, but 2nd variant hits self499 variant -> ENTIRE group must be dropped
        _fake_group(
            "G_hit_var_self",
            "mixed",
            2,
            variant_problems=["Clean variant A", "Self test variant $2a+2b=6$"],
        ),
        # Source hits author transfer
        _fake_group("G_hit_src_auth_t", "harder", 1, source_problem="Author test transfer $n=7$"),
        # Variant hits author source
        _fake_group(
            "G_hit_var_auth_s",
            "easier",
            2,
            variant_problems=["Author test source $m=5$", "Other clean variant"],
        ),
        # Completely clean group
        _fake_group("G_clean_1", "easier", 2),
        _fake_group("G_clean_2", "harder", 1),
    ]

    clean, summary = exclude_test_overlaps(groups, test_index)
    assert [g["source_id"] for g in clean] == ["G_clean_1", "G_clean_2"]
    assert summary["excluded_groups"] == 4
    assert summary["remaining_clean_groups"] == 2
    assert summary["individual_reason_counts"]["train_source_hits_self499_source"] == 1
    assert summary["individual_reason_counts"]["train_variant_hits_self499_variant"] == 1
    assert summary["individual_reason_counts"]["train_source_hits_author_test_variant"] == 1
    assert summary["individual_reason_counts"]["train_variant_hits_author_test_source"] == 1


# ---------------------------------------------------------------------------
# 3 & 4. Reproducible stratified sampling + zero train/dev overlap
# ---------------------------------------------------------------------------


def test_stratified_split_reproducible_and_disjoint():
    pool: List[Dict[str, Any]] = []
    idx = 0
    for nv, direction, count in [
        (1, "easier", 40),
        (2, "easier", 60),
        (1, "mixed", 20),
        (2, "mixed", 30),
        (1, "harder", 10),
        (2, "harder", 20),
        (3, "easier", 10),
    ]:
        for _ in range(count):
            idx += 1
            pool.append(_fake_group(f"S{idx:04d}", direction, nv))

    tr1, dv1, m1 = stratified_train_dev_split(pool, train_size=120, dev_size=30, seed=42)
    tr2, dv2, m2 = stratified_train_dev_split(pool, train_size=120, dev_size=30, seed=42)
    tr3, dv3, _ = stratified_train_dev_split(pool, train_size=120, dev_size=30, seed=99)

    assert [g["source_id"] for g in tr1] == [g["source_id"] for g in tr2]
    assert [g["source_id"] for g in dv1] == [g["source_id"] for g in dv2]
    assert m1["train_allocation_by_stratum"] == m2["train_allocation_by_stratum"]
    assert [g["source_id"] for g in tr1] != [g["source_id"] for g in tr3]

    # Disjointness
    tr_ids = {g["source_id"] for g in tr1}
    dv_ids = {g["source_id"] for g in dv1}
    assert len(tr_ids) == 120
    assert len(dv_ids) == 30
    assert tr_ids.isdisjoint(dv_ids)

    # Check stratification proportions preserved roughly
    assert m1["train_allocation_by_stratum"]["2|easier"] > m1["train_allocation_by_stratum"]["1|easier"]
    assert m1["dev_allocation_by_stratum"]["2|easier"] > m1["dev_allocation_by_stratum"]["1|harder"]


# ---------------------------------------------------------------------------
# 5. Rasch smoothed p and difficulty b
# ---------------------------------------------------------------------------


def test_rasch_smoothed_p_and_b():
    # Exact formula: p = (n + 0.5) / 17, b = ln((1 - p) / p)
    p0, b0 = compute_smoothed_p_and_b(0, 16)
    p8, b8 = compute_smoothed_p_and_b(8, 16)
    p16, b16 = compute_smoothed_p_and_b(16, 16)

    assert math.isclose(p0, 0.5 / 17.0, rel_tol=1e-12)
    assert math.isclose(p8, 8.5 / 17.0, rel_tol=1e-12)
    assert math.isclose(p16, 16.5 / 17.0, rel_tol=1e-12)

    assert math.isfinite(b0) and math.isfinite(b16)
    assert math.isclose(b8, 0.0, abs_tol=1e-12)
    assert math.isclose(b0, -b16, rel_tol=1e-12)
    assert b0 > b8 > b16  # 0/16 is hardest (largest b), 16/16 is easiest (most negative b)

    # Failed rollouts excluded from denominator (e.g., 1 failed -> n_scored=15)
    p_fail, b_fail = compute_smoothed_p_and_b(10, 15)
    assert math.isclose(p_fail, 10.5 / 16.0, rel_tol=1e-12)
    assert math.isclose(b_fail, math.log((1.0 - p_fail) / p_fail), rel_tol=1e-12)

    with pytest.raises(ValueError):
        compute_smoothed_p_and_b(17, 16)
    with pytest.raises(ValueError):
        compute_smoothed_p_and_b(0, 0)


# ---------------------------------------------------------------------------
# 6. Ladder sorting and generation-tag tie-breaking
# ---------------------------------------------------------------------------


def test_ladder_sorting_strict_and_tag_fallback():
    # Case A: clear separation (>= 1 rollout difference) -> pure p_new descending
    v_easy = {"variant_id": "v1", "n_correct_hybrid": 14, "p_new": 14.5 / 17, "generation_tier": "harder"}
    v_mid = {"variant_id": "v2", "n_correct_hybrid": 8, "p_new": 8.5 / 17, "generation_tier": "easier"}
    v_hard = {"variant_id": "v3", "n_correct_hybrid": 2, "p_new": 2.5 / 17, "generation_tier": "nearby"}
    ordered, tie_used = sort_ladder_variants([v_hard, v_easy, v_mid])
    assert [v["variant_id"] for v in ordered] == ["v1", "v2", "v3"]
    assert [v["ladder_rank"] for v in ordered] == [1, 2, 3]
    assert tie_used is False

    # Case B: exact tie (difference 0 < 1) -> broken by generation_tier (easier < nearby < harder)
    t_harder = {"variant_id": "a_harder", "n_correct_hybrid": 6, "p_new": 6.5 / 17, "generation_tier": "harder"}
    t_easier = {"variant_id": "z_easier", "n_correct_hybrid": 6, "p_new": 6.5 / 17, "generation_tier": "easier"}
    t_nearby = {"variant_id": "m_nearby", "n_correct_hybrid": 6, "p_new": 6.5 / 17, "generation_tier": "nearby"}
    ordered2, tie_used2 = sort_ladder_variants([t_harder, t_nearby, t_easier])
    assert [v["variant_id"] for v in ordered2] == ["z_easier", "m_nearby", "a_harder"]
    assert [v["ladder_rank"] for v in ordered2] == [1, 2, 3]
    assert tie_used2 is True

    # Case C: sub-1-rollout difference due to n_failed (8/16 = 8.0 vs 8/15 -> eff 8.53, diff 0.53 < 1)
    f_harder = {"variant_id": "h", "n_correct_hybrid": 8, "n_scored": 15, "p_new": 8.5 / 16, "generation_tier": "harder"}
    f_easier = {"variant_id": "e", "n_correct_hybrid": 8, "n_scored": 16, "p_new": 8.5 / 17, "generation_tier": "easier"}
    ordered3, tie_used3 = sort_ladder_variants([f_harder, f_easier])
    assert [v["variant_id"] for v in ordered3] == ["e", "h"]
    assert tie_used3 is True

    assert infer_generation_tier("self_generated", "reduce non-core preparation") == "easier"
    assert infer_generation_tier("paper_transfer", None) == "paper_transfer"


# ---------------------------------------------------------------------------
# 7. Hybrid grader: rule-first, LLM fallback, unknown=wrong, tech failure!=verdict
# ---------------------------------------------------------------------------


def test_hybrid_grader_rule_pass_skips_llm():
    llm_calls = 0

    def fake_rule(resp: str, ref: str) -> Dict[str, Any]:
        return {"score": 1.0, "acc": 1.0, "pred": ref, "incorrect_format": 0}

    def fake_llm(prob: str, ref: str, resp: str) -> Tuple[Optional[bool], str]:
        nonlocal llm_calls
        llm_calls += 1
        return False, "ok"

    grader = HybridGrader(rule_grader=fake_rule, llm_judge=fake_llm, sleep_fn=lambda _: None)
    res = grader.grade(problem="1+1?", reference_answer="2", student_response=r"\boxed{2}")
    assert res.correct is True
    assert res.rule_correct is True
    assert res.llm_called is False
    assert res.unknown is False
    assert res.technical_failure is False
    assert res.source == "rule"
    assert llm_calls == 0


def test_hybrid_grader_rule_fail_falls_back_to_llm_true_false_unknown():
    def fake_rule_fail(resp: str, ref: str) -> Dict[str, Any]:
        return {"score": 0.0, "acc": 0.0, "pred": "", "incorrect_format": 1}

    # LLM returns True -> hybrid correct
    g_true = HybridGrader(
        rule_grader=fake_rule_fail,
        llm_judge=lambda p, r, s: (True, "ok"),
        sleep_fn=lambda _: None,
    )
    res_t = g_true.grade(problem="p", reference_answer="1/2", student_response="0.5")
    assert res_t.correct is True
    assert res_t.rule_correct is False
    assert res_t.llm_called is True
    assert res_t.unknown is False
    assert res_t.technical_failure is False

    # LLM returns False -> hybrid incorrect
    g_false = HybridGrader(
        rule_grader=fake_rule_fail,
        llm_judge=lambda p, r, s: (False, "ok"),
        sleep_fn=lambda _: None,
    )
    res_f = g_false.grade(problem="p", reference_answer="2", student_response="3")
    assert res_f.correct is False
    assert res_f.llm_called is True
    assert res_f.unknown is False

    # LLM returns None (explicit null or bad format) -> counted as False AND unknown=True
    g_unk = HybridGrader(
        rule_grader=fake_rule_fail,
        llm_judge=lambda p, r, s: (None, "explicit_null"),
        sleep_fn=lambda _: None,
    )
    res_u = g_unk.grade(problem="p", reference_answer="2", student_response="maybe")
    assert res_u.correct is False
    assert res_u.llm_called is True
    assert res_u.unknown is True
    assert res_u.technical_failure is False
    assert res_u.source == "llm_unknown"


def test_hybrid_grader_technical_failure_retries_and_not_counted_as_verdict():
    attempts = 0

    def flaky_llm_recovers(p: str, r: str, s: str) -> Tuple[Optional[bool], str]:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TechnicalFailureError("HTTP 503")
        return True, "ok"

    win = AIMDWindow(max_window=16, base_backoff_sec=0.0, rate_limit_backoff_sec=0.0)
    g_recover = HybridGrader(
        rule_grader=lambda s, r: {"score": 0.0, "acc": 0.0, "pred": "", "incorrect_format": 0},
        llm_judge=flaky_llm_recovers,
        max_retries=3,
        window=win,
        retry_sleep_sec=0.0,
        sleep_fn=lambda _: None,
    )
    res_ok = g_recover.grade(problem="p", reference_answer="2", student_response="2")
    assert res_ok.correct is True
    assert res_ok.technical_failure is False
    assert res_ok.attempts == 3
    assert win.window == 2  # started at 1, failures stayed at 1, then success bumped to 2

    # Persistent technical failure -> NOT a verdict (correct=None, technical_failure=True)
    def always_fail(p: str, r: str, s: str) -> Tuple[Optional[bool], str]:
        raise TechnicalFailureError("connection reset")

    g_fail = HybridGrader(
        rule_grader=lambda s, r: {"score": 0.0, "acc": 0.0, "pred": "", "incorrect_format": 0},
        llm_judge=always_fail,
        max_retries=2,
        retry_sleep_sec=0.0,
        sleep_fn=lambda _: None,
    )
    res_tf = g_fail.grade(problem="p", reference_answer="2", student_response="2")
    assert res_tf.correct is None
    assert res_tf.technical_failure is True
    assert res_tf.unknown is False
    assert res_tf.attempts == 3
    assert res_tf.source == "technical_failure"


def test_aimd_window_and_failure_rate_gate_and_response_parser():
    # AIMD: 1 -> +1 -> +1 -> +1 (4) -> error -> //2 (2) -> error -> 1
    win = AIMDWindow(max_window=16, base_backoff_sec=0.0, rate_limit_backoff_sec=0.0)
    assert win.window == 1
    for _ in range(3):
        win.acquire()
        win.release(True)
    assert win.window == 4
    win.acquire()
    win.release(False)
    assert win.window == 2
    win.acquire()
    win.release(False)
    assert win.window == 1

    # Failure rate gate (>1% after min_samples)
    gate = FailureRateGate(max_failure_rate=0.01, min_samples=100)
    for _ in range(99):
        gate.record(technical_failure=False)
    gate.record(technical_failure=True)  # exactly 1/100 == 1.0% -> still allowed
    assert math.isclose(gate.failure_rate, 0.01)
    with pytest.raises(TechnicalFailureGateError):
        gate.record(technical_failure=True)  # 2/101 > 1% -> abort

    # Judge payload & strict response parser
    body = build_judge_payload(
        model="Qwen3.6-27B",
        problem="p",
        reference_answer="2",
        student_response="2",
    )
    assert body["model"] == "Qwen3.6-27B"
    assert body["temperature"] == 0

    def _resp(content: str, finish: str = "stop", model: str = "Qwen3.6-27B") -> Dict[str, Any]:
        return {
            "model": model,
            "choices": [{"finish_reason": finish, "message": {"content": content}}],
        }

    assert parse_judge_response(_resp('{"correct": true}'), "Qwen3.6-27B") == (True, "ok")
    assert parse_judge_response(_resp('```json\n{"correct": false}\n```'), "Qwen3.6-27B") == (False, "ok")
    assert parse_judge_response(_resp('{"correct": null}'), "Qwen3.6-27B") == (None, "explicit_null")
    assert parse_judge_response(_resp('not json'), "Qwen3.6-27B") == (None, "response_not_json")
    assert parse_judge_response(_resp('{"correct": true}', finish="length"), "Qwen3.6-27B") == (
        None,
        "response_truncated",
    )
    assert parse_judge_response(_resp('{"correct": true, "extra": 1}'), "Qwen3.6-27B") == (
        None,
        "response_schema_invalid",
    )
