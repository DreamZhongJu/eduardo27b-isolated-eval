"""Unit tests for T-023 Route A data pipeline (Card v3 / Inbox #3), using fake data ONLY.

Runnable via BOTH `python -m unittest discover -s tests` (Pro 5000 pro5000-fast)
and `pytest` (cu04).

Covers:
1. Problem text normalization & hash deduplication (whitespace, NFKC, LaTeX spacing).
2. Test-set exclusion (self499 + author test + author all_test) removing whole source group.
3. Reproducible Stage 1 v3 sampling (multi-variant all-in proportional split + single-variant stratified fill).
4. Zero overlap between train & dev splits (source_id & normalized text hashes).
5. Rasch smoothed pass rate p and difficulty b calculation (boundaries & n_failed).
6. Ladder sorting by p descending + generation-tag tie-break (<1 rollout difference).
7. Canonical T-024 hybrid_grader.py (SHA-256 7d2a5f46...) + caller >1% failure gate + sync wrapper:
   - Rule pass skips 27B
   - Rule fail calls 27B (True/False/null/malformed)
   - Technical failure retries and is NOT counted as a verdict
   - AIMD window 1..16 + >1% abort gate
   - Prompt matches agnes_eval_v3/agnes_judge.py byte-for-byte
   - Real author grade_math_answer check when math_verify==0.8.0 is installed (Pro 5000)
8. Pro 5000 GPU waiter (GPU 6 sensor quirk + 4->3 fallback + TP=2/DP=1..2 split).
9. End-to-end synthetic Stage 2 runner (gpu -> agnes -> summarize) with zero text leakage in summary.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure `eduardo.utils.grade_math_answer` is available on cu04 (where EduardoRL is not installed)
# while using the REAL `eduardo.utils.grade_math_answer` whenever `eduardo` IS installed (e.g. Pro 5000).
_USING_REAL_EDUARDO = False
try:
    import eduardo.utils  # type: ignore  # noqa: F401
    _USING_REAL_EDUARDO = True
except ImportError:
    _edu_pkg = types.ModuleType("eduardo")
    _edu_utils = types.ModuleType("eduardo.utils")

    def _stub_grade_math_answer(solution_str: str, ground_truth: str) -> Dict[str, Any]:
        marker = f"\\boxed{{{ground_truth}}}"
        if marker in (solution_str or ""):
            return {"score": 1.0, "acc": 1.0, "pred": ground_truth, "incorrect_format": 0, "truncated": 0}
        inc_fmt = 0 if "\\boxed{" in (solution_str or "") else 1
        return {"score": 0.0, "acc": 0.0, "pred": "", "incorrect_format": inc_fmt, "truncated": 0}

    _edu_utils.grade_math_answer = _stub_grade_math_answer  # type: ignore[attr-defined]
    _edu_pkg.utils = _edu_utils  # type: ignore[attr-defined]
    sys.modules["eduardo"] = _edu_pkg
    sys.modules["eduardo.utils"] = _edu_utils

from routeA_data.gpu_wait_pro5000 import (
    is_gpu_eligible,
    parse_gpu_csv,
    select_confirmed_gpus,
    split_judge_and_student_gpus,
)
from routeA_data.hybrid_grader import (
    AGNES_JUDGE_REFERENCE_SHA256,
    EQUIV_JUDGE_SYSTEM_PROMPT,
    AsyncAdaptiveWindow,
    HybridGrader,
    check_technical_failure_rate,
    parse_equiv_verdict,
)
from routeA_data.hybrid_grader_caller import (
    CANONICAL_HYBRID_GRADER_SHA256,
    TechnicalFailureGateError,
    TechnicalFailureTracker,
    assert_canonical_hybrid_grader,
    grade_sync,
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
from routeA_data.stage2_runner import (
    classify_answer_form,
    run_agnes_dev_phase,
    run_gpu_phase,
    run_summarize_phase,
    sha256_file,
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


class TestNormalizationAndStage1(unittest.TestCase):
    def test_normalize_whitespace_fullwidth_and_latex_spaces(self) -> None:
        a = r"Find $x\, + \; y\quad = \qquad 10$."
        b = "Find $x+y=10$."
        c = "Ｆｉｎｄ　＄ｘ＋ｙ＝１０＄．"
        d = r"Find $x \hspace{2em} + ~ y \  = \thinspace 10$."
        self.assertEqual(normalize_problem_text(a), normalize_problem_text(b))
        self.assertEqual(normalize_problem_text(b), normalize_problem_text(c))
        self.assertEqual(normalize_problem_text(b), normalize_problem_text(d))
        self.assertEqual(problem_text_hash(a), problem_text_hash(c))
        self.assertNotEqual(problem_text_hash(a), problem_text_hash("Find $x+y=11$."))

    def test_deduplicate_sources_merges_overlay(self) -> None:
        def _mk(cid: str, q: str, origin: str = "self_generated") -> Dict[str, Any]:
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
                "transfers": [_mk("C1", "Fake var 1")],
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
                "transfers": [_mk("C1", "Fake var 1"), _mk("C2", "Fake var 2", origin="paper_transfer")],
            },
        ]
        groups, stats = deduplicate_sources(raw)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["variants"]), 2)
        self.assertEqual(stats["duplicate_source_rows"], 1)
        self.assertEqual(stats["duplicate_candidate_rows"], 1)

    def test_exclude_test_and_all_test_overlaps_removes_whole_group(self) -> None:
        self_test = [
            {
                "source_problem": r"Self test source $a\,+\,b=3$",
                "transfer_questions": [{"problem": r"Self test variant $2a + 2b = 6$"}],
            }
        ]
        author_test = [
            {
                "problem": r"Author test source $m \quad = 5$",
                "transfer_problem": r"Author test transfer $n \; = 7$",
            }
        ]
        author_all_test = [
            {
                "problem": r"Author all_test extra $z = 99$",
                "transfer_problem": r"Author all_test transfer $w = 100$",
            }
        ]
        idx = build_test_hash_index(self_test, author_test, author_all_test)
        groups = [
            _fake_group("G_hit_self_s", "easier", 2, source_problem="Self test source $a+b=３$"),
            _fake_group("G_hit_self_v", "mixed", 2, variant_problems=["Clean A", "Self test variant $2a+2b=6$"]),
            _fake_group("G_hit_auth_t", "harder", 1, source_problem="Author test transfer $n=7$"),
            _fake_group("G_hit_alltest_s", "easier", 2, source_problem="Author all_test extra $z=99$"),
            _fake_group("G_hit_alltest_v", "mixed", 2, variant_problems=["Clean B", "Author all_test transfer $w=100$"]),
            _fake_group("G_clean_1", "easier", 2),
            _fake_group("G_clean_2", "harder", 1),
        ]
        clean, summary = exclude_test_overlaps(groups, idx)
        self.assertEqual([g["source_id"] for g in clean], ["G_clean_1", "G_clean_2"])
        self.assertEqual(summary["excluded_groups"], 5)
        self.assertEqual(summary["remaining_clean_groups"], 2)

    def test_multi_variant_all_in_split_reproducible_and_disjoint(self) -> None:
        pool: List[Dict[str, Any]] = []
        idx = 0
        for nv, direction, count in [
            (1, "easier", 50),
            (1, "mixed", 30),
            (1, "harder", 20),
            (2, "easier", 60),
            (2, "mixed", 30),
            (2, "harder", 15),
            (3, "easier", 10),
            (4, "easier", 1),
        ]:
            for _ in range(count):
                idx += 1
                pool.append(_fake_group(f"S{idx:04d}", direction, nv))

        # Total pool = 216 (116 multi-variant + 100 single-variant); pick 140 train + 20 dev = 160
        tr1, dv1, m1 = stratified_train_dev_split(pool, train_size=140, dev_size=20, seed=42, multi_variant_all_in=True)
        tr2, dv2, _ = stratified_train_dev_split(pool, train_size=140, dev_size=20, seed=42, multi_variant_all_in=True)
        tr3, _, _ = stratified_train_dev_split(pool, train_size=140, dev_size=20, seed=99, multi_variant_all_in=True)

        self.assertEqual([g["source_id"] for g in tr1], [g["source_id"] for g in tr2])
        self.assertEqual([g["source_id"] for g in dv1], [g["source_id"] for g in dv2])
        self.assertNotEqual([g["source_id"] for g in tr1], [g["source_id"] for g in tr3])

        tr_ids = {g["source_id"] for g in tr1}
        dv_ids = {g["source_id"] for g in dv1}
        self.assertTrue(tr_ids.isdisjoint(dv_ids))
        self.assertEqual(len(tr_ids), 140)
        self.assertEqual(len(dv_ids), 20)

        # ALL 116 multi-variant sources must be included across train + dev!
        n_multi_selected = m1["train_summary"]["n_multi_variant_sources"] + m1["dev_summary"]["n_multi_variant_sources"]
        self.assertEqual(n_multi_selected, 116)
        # Multi-variant ratio in train and dev should match closely (~116/160 = 0.725)
        self.assertAlmostEqual(m1["train_summary"]["multi_variant_ratio"], m1["dev_summary"]["multi_variant_ratio"], delta=0.05)


class TestRaschAndLadder(unittest.TestCase):
    def test_rasch_smoothed_p_and_b(self) -> None:
        p0, b0 = compute_smoothed_p_and_b(0, 16)
        p8, b8 = compute_smoothed_p_and_b(8, 16)
        p16, b16 = compute_smoothed_p_and_b(16, 16)
        self.assertAlmostEqual(p0, 0.5 / 17.0, places=12)
        self.assertAlmostEqual(p8, 8.5 / 17.0, places=12)
        self.assertAlmostEqual(p16, 16.5 / 17.0, places=12)
        self.assertTrue(math.isfinite(b0) and math.isfinite(b16))
        self.assertAlmostEqual(b8, 0.0, places=12)
        self.assertAlmostEqual(b0, -b16, places=12)
        self.assertGreater(b0, b8)
        self.assertGreater(b8, b16)

        p_fail, b_fail = compute_smoothed_p_and_b(10, 15)
        self.assertAlmostEqual(p_fail, 10.5 / 16.0, places=12)
        self.assertAlmostEqual(b_fail, math.log((1.0 - p_fail) / p_fail), places=12)

        with self.assertRaises(ValueError):
            compute_smoothed_p_and_b(17, 16)
        with self.assertRaises(ValueError):
            compute_smoothed_p_and_b(0, 0)

    def test_ladder_sorting_strict_and_tag_fallback(self) -> None:
        v_easy = {"variant_id": "v1", "n_correct_hybrid": 14, "p_new": 14.5 / 17, "generation_tier": "harder"}
        v_mid = {"variant_id": "v2", "n_correct_hybrid": 8, "p_new": 8.5 / 17, "generation_tier": "easier"}
        v_hard = {"variant_id": "v3", "n_correct_hybrid": 2, "p_new": 2.5 / 17, "generation_tier": "nearby"}
        ordered, tie_used = sort_ladder_variants([v_hard, v_easy, v_mid])
        self.assertEqual([v["variant_id"] for v in ordered], ["v1", "v2", "v3"])
        self.assertEqual([v["ladder_rank"] for v in ordered], [1, 2, 3])
        self.assertFalse(tie_used)

        t_harder = {"variant_id": "a_harder", "n_correct_hybrid": 6, "p_new": 6.5 / 17, "generation_tier": "harder"}
        t_easier = {"variant_id": "z_easier", "n_correct_hybrid": 6, "p_new": 6.5 / 17, "generation_tier": "easier"}
        t_nearby = {"variant_id": "m_nearby", "n_correct_hybrid": 6, "p_new": 6.5 / 17, "generation_tier": "nearby"}
        ordered2, tie_used2 = sort_ladder_variants([t_harder, t_nearby, t_easier])
        self.assertEqual([v["variant_id"] for v in ordered2], ["z_easier", "m_nearby", "a_harder"])
        self.assertTrue(tie_used2)

        self.assertEqual(infer_generation_tier("self_generated", "reduce non-core preparation"), "easier")
        self.assertEqual(infer_generation_tier("paper_transfer", None), "paper_transfer")


class TestCanonicalHybridGraderAndCaller(unittest.TestCase):
    def test_canonical_sha256_and_agnes_prompt_parity(self) -> None:
        got = assert_canonical_hybrid_grader()
        self.assertEqual(got, CANONICAL_HYBRID_GRADER_SHA256)
        agnes_path = Path(__file__).resolve().parent.parent / "evaluation" / "agnes_eval_v3" / "agnes_judge.py"
        if agnes_path.is_file():
            self.assertEqual(sha256_file(agnes_path), AGNES_JUDGE_REFERENCE_SHA256)
            spec = importlib.util.spec_from_file_location("_agnes_v3_ref", agnes_path)
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)
            self.assertEqual(EQUIV_JUDGE_SYSTEM_PROMPT, mod.SYSTEM)

    def test_rule_pass_skips_llm_and_rule_fail_calls_llm(self) -> None:
        calls = 0

        async def fake_llm(messages: List[Dict[str, str]]) -> Tuple[Optional[str], Optional[str]]:
            nonlocal calls
            calls += 1
            return '{"correct": true}', "stop"

        grader = HybridGrader(
            llm_caller=fake_llm,
            rule_grader=lambda s, g: {"score": 1.0, "acc": 1.0, "incorrect_format": 0},
            backoff_base=0.0,
        )
        res = grade_sync(grader, problem="1+1?", reference_answer="2", student_response=r"\boxed{2}")
        self.assertTrue(res.correct)
        self.assertTrue(res.rule_correct)
        self.assertFalse(res.llm_called)
        self.assertEqual(calls, 0)

        # Rule fails -> LLM returns true / false / null / malformed
        for content, exp_correct, exp_unk in [
            ('{"correct": true}', True, False),
            ('{"correct": false}', False, False),
            ('{"correct": null}', False, True),
            ('not json at all', False, True),
        ]:
            async def _llm(_m: List[Dict[str, str]], c: str = content) -> Tuple[Optional[str], Optional[str]]:
                return c, "stop"

            g2 = HybridGrader(
                llm_caller=_llm,
                rule_grader=lambda s, g: {"score": 0.0, "acc": 0.0, "incorrect_format": 1},
                max_attempts=2,
                backoff_base=0.0,
            )
            r2 = grade_sync(g2, problem="p", reference_answer="2", student_response="0.5")
            self.assertEqual(r2.correct, exp_correct)
            self.assertEqual(r2.is_unknown, exp_unk)
            self.assertTrue(r2.llm_called)
            self.assertFalse(r2.technical_failure)

    def test_technical_failure_retries_not_verdict_and_caller_1pct_gate(self) -> None:
        attempts = 0

        async def always_timeout(_m: List[Dict[str, str]]) -> Tuple[Optional[str], Optional[str]]:
            nonlocal attempts
            attempts += 1
            raise TimeoutError("simulated timeout")

        win = AsyncAdaptiveWindow(max_window=16, max_cooldown=0.0)
        grader = HybridGrader(
            llm_caller=always_timeout,
            rule_grader=lambda s, g: {"score": 0.0, "acc": 0.0, "incorrect_format": 0},
            max_attempts=3,
            window=win,
            backoff_base=0.0,
        )
        res = grade_sync(grader, problem="p", reference_answer="2", student_response="2")
        self.assertEqual(attempts, 3)
        self.assertTrue(res.technical_failure)
        self.assertEqual(res.status, "technical_error")
        self.assertFalse(res.is_unknown)

        # AIMD window + exact 1% gate in caller layer
        async def _check_win() -> None:
            w = AsyncAdaptiveWindow(max_window=16, max_cooldown=0.0)
            self.assertEqual(w.window, 1)
            for _ in range(3):
                await w.acquire()
                await w.release(True)
            self.assertEqual(w.window, 4)
            await w.acquire()
            await w.release(False)
            self.assertEqual(w.window, 2)

        asyncio.run(_check_win())
        self.assertFalse(check_technical_failure_rate(1, 100, max_percent=1))
        self.assertTrue(check_technical_failure_rate(2, 100, max_percent=1))

        tracker = TechnicalFailureTracker(max_percent=1, min_samples=100)
        for _ in range(99):
            tracker.record(technical_failure=False)
        tracker.record(technical_failure=True)  # 1/100 == 1% ok
        with self.assertRaises(TechnicalFailureGateError):
            tracker.record(technical_failure=True)  # 2/101 > 1% abort

    def test_real_math_verify_when_available(self) -> None:
        if not _USING_REAL_EDUARDO:
            return
        from eduardo.utils import grade_math_answer  # type: ignore

        r_exact = grade_math_answer(r"Therefore the answer is \boxed{4}.", "4")
        self.assertEqual(r_exact["acc"], 1.0)
        r_bad = grade_math_answer(r"Therefore the answer is \boxed{7}.", "8")
        self.assertEqual(r_bad["acc"], 0.0)
        if importlib.util.find_spec("math_verify") is not None:
            r_sym = grade_math_answer(r"Therefore the answer is \boxed{\frac{1}{2}}.", "0.5")
            self.assertEqual(r_sym["acc"], 1.0)
            r_box = grade_math_answer(r"Answer: \boxed{2x+2}", "2(x+1)")
            self.assertEqual(r_box["acc"], 1.0)

    def test_parse_equiv_verdict(self) -> None:
        self.assertEqual(parse_equiv_verdict('{"correct": true}', "stop"), (True, None))
        self.assertEqual(parse_equiv_verdict('```json\n{"correct": false}\n```', "stop"), (False, None))
        self.assertEqual(parse_equiv_verdict('{"correct": null}', "stop"), (None, None))
        self.assertEqual(parse_equiv_verdict('{"correct": true}', "length"), (None, "response_truncated"))
        self.assertEqual(parse_equiv_verdict('bad', "stop"), (None, "response_not_json"))


class TestPro5000GpuWaitAndSyntheticStage2(unittest.TestCase):
    def test_gpu_wait_quirk_and_4_to_3_fallback(self) -> None:
        # GPU 0-4 busy (T-022 using 5 GPUs), GPU 5, 6 (util=100 quirk, used=20MiB), 7 idle -> 3 GPUs!
        gpu_csv = "\n".join(
            [
                "0,GPU-0,10000,73415,85",
                "1,GPU-1,10000,73415,85",
                "2,GPU-2,10000,73415,85",
                "3,GPU-3,10000,73415,85",
                "4,GPU-4,10000,73415,85",
                "5,GPU-5,73390,73415,0",
                "6,GPU-6,73395,73415,100",
                "7,GPU-7,73390,73415,0",
            ]
        )
        apps_csv = "GPU-0,101,60000\nGPU-1,102,60000\nGPU-2,103,60000\nGPU-3,104,60000\nGPU-4,105,60000\n"
        snap = parse_gpu_csv(gpu_csv, apps_csv)
        self.assertTrue(is_gpu_eligible(snap[6], 70000))
        sel = select_confirmed_gpus(snap, snap, num_gpus=4, fallback_min_gpus=3, min_free_mib=70000)
        self.assertEqual(sel, [5, 7, 6])  # non-quirk GPUs preferred first
        j_gpus, s_gpus = split_judge_and_student_gpus(sel)
        self.assertEqual(j_gpus, [5, 7])
        self.assertEqual(s_gpus, [6])

    def test_end_to_end_synthetic_stage2_gpu_agnes_summarize(self) -> None:
        rows = [
            {**_fake_group("S_TR", "easier", 2), "split": "train"},
            {**_fake_group("S_DV", "mixed", 2), "split": "dev"},
        ]

        class _FakeChoice:
            def __init__(self, text: str) -> None:
                self.message = types.SimpleNamespace(content=text)
                self.finish_reason = "stop"

        class _FakeStudentCompletions:
            def __init__(self) -> None:
                self.n = 0

            async def create(self, **kwargs: Any) -> Any:
                self.n += 1
                # Half boxed "1", half plain "1" (rule misses, 27B/Agnes pass!)
                txt = r"\boxed{1}" if (self.n % 2 == 0) else "Final answer is 1"
                return types.SimpleNamespace(choices=[_FakeChoice(txt)])

        fake_student = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=_FakeStudentCompletions())
        )

        async def fake_27b_llm(messages: List[Dict[str, str]]) -> Tuple[Optional[str], Optional[str]]:
            payload = json.loads(messages[1]["content"])
            ok = payload["reference_answer"] in payload["student_response"]
            return json.dumps({"correct": ok}), "stop"

        def fake_agnes_caller(body: Dict[str, Any], _key: str) -> Tuple[int, Optional[Dict[str, Any]]]:
            payload = json.loads(body["messages"][1]["content"])
            ok = payload["reference_answer"] in payload["student_response"]
            return 200, {
                "model": "agnes-3.0-flash",
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"correct": ok})}}],
            }

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            gpu_stats = asyncio.run(
                run_gpu_phase(
                    rows,
                    run_dir=run_dir,
                    student_client=fake_student,
                    student_model="fake-student",
                    judge_client=None,
                    judge_model="judge-local",
                    k_samples=4,
                    llm_caller=fake_27b_llm,
                )
            )
            self.assertEqual(gpu_stats["variants_done"], 4)
            self.assertGreater(gpu_stats["dev_rule_pass_27b_audit_calls"], 0)

            agnes_stats = asyncio.run(
                run_agnes_dev_phase(
                    rows,
                    run_dir=run_dir,
                    api_key="fake-key",
                    max_window=4,
                    agnes_caller=fake_agnes_caller,
                )
            )
            # Only dev variants (2 variants * 4 rollouts = 8) sent to Agnes, NEVER train!
            self.assertEqual(agnes_stats["total_dev_items"], 8)
            self.assertEqual(agnes_stats["tech_failed"], 0)

            summary = run_summarize_phase(rows, run_dir=run_dir, input_sha256="deadbeef")
            self.assertEqual(summary["counts"]["n_sources"], 2)
            self.assertEqual(summary["counts"]["n_variants"], 4)
            self.assertEqual(summary["dev_three_way_consistency"]["dev_rollouts_evaluated"], 8)
            self.assertEqual(summary["dev_three_way_consistency"]["qwen27b_vs_agnes"]["agreement_rate_unk_as_false"], 1.0)
            self.assertGreater(summary["counts"]["n_promoted_rollouts_hybrid_over_rule"], 0)

            # Verify zero raw problem text leaked into stage2_summary.txt
            txt = (run_dir / "stage2_summary.txt").read_text(encoding="utf-8")
            self.assertNotIn("Synthetic variant", txt)
            self.assertEqual(classify_answer_form("42"), "integer")
            self.assertEqual(classify_answer_form(r"\frac{3}{4}"), "simple_fraction")


if __name__ == "__main__":
    unittest.main()
