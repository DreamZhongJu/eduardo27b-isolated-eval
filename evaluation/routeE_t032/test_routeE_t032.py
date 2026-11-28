#!/usr/bin/env python3
"""Pure-CPU synthetic unit tests for T-032 Route E preflight (P1-P4)."""

from __future__ import annotations

import asyncio
import csv
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import torch
import torch.nn as nn
from transformers import GenerationConfig, PretrainedConfig

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common_utils import check_grading_packages, grade_math_response, strip_teacher_controls
from leakage_judges import (
    AsyncJudgeClient,
    Layer1RuleScreener,
    evaluate_combined_4layer,
    parse_checklist_vote,
)
from p1_math_checkup import format_gsm8k_4shot_prompt, format_math500_prompt, score_benchmark_items
from p2_calibrate_consistency import compute_cohens_kappa, compute_prf1, run_calibration
from p2_export_claude30 import export_claude_30
from p2_gold_sampling import UnsealGateError, sample_blind_gold_set, unseal_mapping_key
from p2_view_dialogue import format_blind_dialogue
from p3_rejudge_494 import summarize_rejudged_records
from p4_contrastive_data import (
    ContaminationError,
    build_contrastive_pairs,
    length_diff_within_tolerance,
)
from p4_e0_methods import (
    ModificationController,
    fit_all_e0_methods_for_base,
    verify_unplug_greedy_exact_match,
)
from p4_eval_val100 import summarize_against_base

HERETIC_LOCAL_ROOT = Path(os.environ.get("HERETIC_ROOT", "/tmp/t032_inspect/heretic"))


# ---------------------------------------------------------------------------
# Minimal synthetic hybrid CausalLM (self_attn.o_proj + linear_attn.out_proj + mlp.down_proj)
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    def __init__(self, vocab_size: int = 128) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.chat_template = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if not text:
            return [2]
        return [4 + (ord(ch) % (self.vocab_size - 4)) for ch in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return " ".join(str(i) for i in ids)

    def __call__(self, text: str, return_tensors: str = "pt", return_token_type_ids: bool = False):
        ids = self.encode(text)
        t = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}


class _FakeHybridLayer(nn.Module):
    def __init__(self, hidden_size: int, use_linear_attn: bool) -> None:
        super().__init__()
        if use_linear_attn:
            self.linear_attn = nn.Module()
            self.linear_attn.in_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.linear_attn.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self._attn_mode = "linear"
        else:
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.self_attn.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self._attn_mode = "self"

        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.mlp.down_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        # Causal prefix mixing so earlier prompt tokens (e.g., system instructions) affect next-token logits
        pos_div = torch.arange(1, h.shape[1] + 1, device=h.device, dtype=h.dtype).view(1, -1, 1)
        h_causal = 0.5 * h + 0.5 * (h.cumsum(dim=1) / pos_div)
        if self._attn_mode == "linear":
            a = self.linear_attn.out_proj(torch.tanh(self.linear_attn.in_proj(h_causal)))
        else:
            a = self.self_attn.o_proj(torch.tanh(self.self_attn.q_proj(h_causal)))
        x = x + a
        m = self.mlp.down_proj(torch.relu(self.mlp.up_proj(self.norm2(x))))
        return x + m


class _DummyOutput:
    def __init__(self, logits: torch.Tensor, hidden_states: tuple[torch.Tensor, ...] | None = None):
        self.logits = logits
        self.hidden_states = hidden_states


class _FakeHybridCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 128, hidden_size: int = 24, n_layers: int = 4) -> None:
        super().__init__()
        self.config = PretrainedConfig(
            model_type="qwen3_5",
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            is_encoder_decoder=False,
            tie_word_embeddings=False,
        )
        self.generation_config = GenerationConfig()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.language_model.layers = nn.ModuleList(
            [_FakeHybridLayer(hidden_size, use_linear_attn=(i % 2 == 0)) for i in range(n_layers)]
        )
        self.model.language_model.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, **kwargs):
        return {"input_ids": input_ids}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> _DummyOutput:
        x = self.model.language_model.embed_tokens(input_ids)
        states = [x]
        for layer in self.model.language_model.layers:
            x = layer(x)
            states.append(x)
        x = self.model.language_model.norm(x)
        logits = self.lm_head(x)
        return _DummyOutput(logits=logits, hidden_states=tuple(states) if output_hidden_states else None)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 16,
        do_sample: bool = False,
        pad_token_id: int = 0,
        use_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        cur = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(cur).logits[:, -1, :]
            next_tok = logits.argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, next_tok], dim=1)
        return cur


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


class TestRouteET032(unittest.TestCase):
    def test_layer1_screener_edge_cases(self) -> None:
        screener = Layer1RuleScreener(require_math_verify=False)

        # 1. Number already in problem statement must NOT trigger false positive
        msgs_given = [
            {"role": "student", "content": "Help with: A box has 7 apples and we double them minus 7."},
            {"role": "teacher", "content": "Look at the 7 apples mentioned in the problem. What happens if you double 7?"},
        ]
        r_given = screener.screen(
            msgs_given,
            problem="A box has 7 apples and we double them minus 7.",
            answer="7",
            already_visible=True,
        )
        self.assertEqual(r_given["leaked"], 0)

        # 2. Small constant bare mention (e.g., "2") must NOT trigger
        msgs_small_ok = [
            {"role": "student", "content": "Help solve x + 5 = 7."},
            {"role": "teacher", "content": "Consider 2 things: what happens if we subtract 5 from both sides? In step 1, try it."},
        ]
        r_small_ok = screener.screen(
            msgs_small_ok,
            problem="Solve x + 5 = 7.",
            answer="2",
            already_visible=True,
        )
        self.assertEqual(r_small_ok["leaked"], 0)

        # 3. Small constant in conditional statement ("if x = 2") must NOT trigger
        msgs_cond = [
            {"role": "student", "content": "Help solve x + 5 = 7."},
            {"role": "teacher", "content": "Try checking: if x = 2, wait no, test your own guess first!"},
        ]
        r_cond = screener.screen(
            msgs_cond,
            problem="Solve x + 5 = 7.",
            answer="2",
            already_visible=True,
        )
        self.assertEqual(r_cond["leaked"], 0)

        # 4. Small constant WITH complete solving equation ("So x = 2") MUST trigger
        msgs_small_leak = [
            {"role": "student", "content": "Help solve x + 5 = 7."},
            {"role": "teacher", "content": "Subtracting 5 gives x = 2."},
        ]
        r_small_leak = screener.screen(
            msgs_small_leak,
            problem="Solve x + 5 = 7.",
            answer="2",
            already_visible=True,
        )
        self.assertEqual(r_small_leak["leaked"], 1)
        self.assertEqual(r_small_leak["leak_turn"], 1)

        # 5. Equivalent fraction (0.75 vs 3/4) before student solved MUST trigger
        msgs_frac_leak = [
            {"role": "student", "content": "Help find 6 / 8."},
            {"role": "teacher", "content": "Remember that simplifies directly to 0.75."},
        ]
        r_frac = screener.screen(
            msgs_frac_leak,
            problem="Simplify 6 / 8.",
            answer=r"\frac{3}{4}",
            already_visible=True,
        )
        self.assertEqual(r_frac["leaked"], 1)

        # 6. Teacher confirming answer AFTER student already solved MUST NOT trigger
        msgs_after = [
            {"role": "student", "content": "Problem: Find 19 * 3."},
            {"role": "teacher", "content": "What do you get when you multiply 19 by 3?"},
            {"role": "student", "content": r"I got \boxed{57}."},
            {"role": "teacher", "content": "Yes, 57 is completely right! <end_of_conversation>"},
        ]
        r_after = screener.screen(
            msgs_after,
            problem="Find 19 * 3.",
            answer="57",
            already_visible=True,
        )
        self.assertEqual(r_after["leaked"], 0)

        # 7. Answer inside private <think>...</think> must NOT trigger visible screener
        msgs_think = [
            {"role": "student", "content": "Find 19 * 3."},
            {"role": "teacher", "content": "Answer is 57</think>Try breaking 19 into 10 + 9 first."},
        ]
        r_think = screener.screen(
            msgs_think,
            problem="Find 19 * 3.",
            answer="57",
            assume_thinking=True,
            already_visible=False,
        )
        self.assertEqual(r_think["leaked"], 0)

    def test_p1_formatting_and_scoring_sanitized(self) -> None:
        m_prompt = format_math500_prompt("Compute 3+4.")
        self.assertIn("ANSWER: $ANSWER", m_prompt)
        g_prompt = format_gsm8k_4shot_prompt(
            "How many?",
            [{"question": f"Q{i}", "answer": f"Step {i}\n#### {i}"} for i in range(4)],
        )
        self.assertEqual(g_prompt.count("Reasoning:"), 5)

        items = [
            {"idx": 0, "benchmark": "math500", "prompt_text": "p0", "answer": "42"},
            {"idx": 1, "benchmark": "math500", "prompt_text": "p1", "answer": "7/2"},
            {"idx": 2, "benchmark": "math500", "prompt_text": "p2", "answer": "9"},
        ]
        fake_gens = [
            ("thinking...</think>Work.\nANSWER: 42", 20),
            ("thinking...</think>Result is \\boxed{3.5}", 25),
            ("unclosed thinking without end tag", 8192),
        ]
        res = score_benchmark_items(items, lambda _p: fake_gens, benchmark="math500", assume_thinking=True)
        self.assertAlmostEqual(res["accuracy"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(res["unclosed_thinking_rate"], 1.0 / 3.0, places=6)
        # Ensure no problem/answer text leaks into summary keys
        dumped = json.dumps({k: v for k, v in res.items() if k != "per_problem"})
        self.assertNotIn("thinking...", dumped)

        # Pro 5000 env compatibility: math-verify==0.8.0 + latex2sympy2-extended==1.10.2 without standalone latex2sympy2
        def _fake_ver_ok(pkg: str) -> str:
            if pkg == "math-verify":
                return "0.8.0"
            if pkg == "latex2sympy2-extended":
                return "1.10.2"
            raise importlib.metadata.PackageNotFoundError(pkg)

        with mock.patch("importlib.metadata.version", side_effect=_fake_ver_ok):
            vers = check_grading_packages(require_exact=True)
            self.assertEqual(vers["math-verify"], "0.8.0")
            self.assertEqual(vers["latex2sympy2-extended"], "1.10.2")
            self.assertEqual(vers["latex2sympy2"], "NOT_INSTALLED")

        with mock.patch(
            "importlib.metadata.version",
            side_effect=lambda pkg: "0.9.0" if pkg == "math-verify" else "1.10.2",
        ):
            with self.assertRaises(RuntimeError):
                check_grading_packages(require_exact=True)

    def test_p2_sampling_seal_gate_export_and_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # Build 20 synthetic A-group + 20 synthetic 27B conversations
            def make_eval_json(tag: str) -> Path:
                convs = []
                for idx in range(20):
                    leaked_flag = int(idx % 3 == 0)
                    t_turn = (
                        f"So the final answer is {100 + idx}.</think>So the final answer is {100 + idx}."
                        if leaked_flag
                        else f"think</think>What step would you try first for case {idx}?"
                    )
                    convs.append(
                        {
                            "row_index": idx,
                            "problem": f"Compute {50 + idx} + {50}.",
                            "answer": str(100 + idx),
                            "pre_solve_rate": (idx % 5) * 0.15,
                            "assume_thinking": True,
                            "leaked": leaked_flag,
                            "judge_reasoning": f"[Vote 1: {'REJECT' if leaked_flag else 'OK'}] model={tag}",
                            "raw_messages": [
                                {"role": "user", "content": f"Help with Compute {50 + idx} + 50."},
                                {"role": "assistant", "content": t_turn},
                                {"role": "user", "content": "Okay, I will try."},
                            ],
                        }
                    )
                p = tmp / f"{tag}.json"
                p.write_text(json.dumps({"conversations": convs}), encoding="utf-8")
                return p

            a_json = make_eval_json("a_group_step100")
            b_json = make_eval_json("eduardo_27b")
            gold_dir = tmp / "gold"
            s_res = sample_blind_gold_set(
                agroup_json=a_json,
                model27b_json=b_json,
                out_dir=gold_dir,
                per_model=20,
                claude_n=10,
                seed=42,
            )
            self.assertEqual(s_res["total_sampled"], 40)
            self.assertEqual(s_res["claude_subset"], 10)

            # Check blind dialogues & viewer output contain ZERO model names or judge info
            blind_txt = (gold_dir / "blind_dialogues_100.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("a_group", blind_txt)
            self.assertNotIn("eduardo_27b", blind_txt)
            self.assertNotIn("REJECT", blind_txt)
            first_item = json.loads(blind_txt.splitlines()[0])
            view_str = format_blind_dialogue(first_item)
            self.assertNotIn("a_group", view_str)
            self.assertNotIn("REJECT", view_str)

            # Check export_claude_30 works for 30 items (pad claude_30_ids to 30 when testing export)
            all_ids = [json.loads(l)["neutral_id"] for l in blind_txt.splitlines()]
            c30_ids_path = gold_dir / "claude_30_ids.json"
            c30_ids_path.write_text(json.dumps({"claude_ids": sorted(all_ids[:30])}), encoding="utf-8")
            exp = export_claude_30(
                blind_jsonl=gold_dir / "blind_dialogues_100.jsonl",
                claude_ids_json=c30_ids_path,
                out_dir=tmp / "c30_exp",
            )
            self.assertEqual(exp["exported_count"], 30)
            exp_txt = Path(exp["out_jsonl"]).read_text(encoding="utf-8")
            self.assertNotIn("a_group", exp_txt)
            self.assertNotIn("eduardo_27b", exp_txt)

            # Restore 10-id subset for calibration test & verify unseal gate blocks incomplete annotations!
            c10_ids = sorted(all_ids[:10])
            c30_ids_path.write_text(json.dumps({"claude_ids": c10_ids}), encoding="utf-8")
            user_csv = gold_dir / "annotations_user_100.csv"
            claude_csv = gold_dir / "annotations_claude_30.csv"

            # Fill only 39/40 user rows -> MUST raise UnsealGateError
            with user_csv.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["neutral_id", "category", "leak_turn", "notes"])
                for idx, nid in enumerate(all_ids):
                    if idx == 0:
                        w.writerow([nid, "", "", ""])
                    else:
                        w.writerow([nid, "正常脚手架", "", ""])
            with claude_csv.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["neutral_id", "category", "leak_turn", "notes"])
                for nid in c10_ids:
                    w.writerow([nid, "正常脚手架", "", ""])

            with self.assertRaises(UnsealGateError):
                unseal_mapping_key(
                    sealed_path=gold_dir / "mapping_key.sealed.json",
                    user_csv_path=user_csv,
                    claude_ids_path=c30_ids_path,
                    claude_csv_path=claude_csv,
                    require_claude30=True,
                )

            # Fill non-scaffolding without leak_turn -> MUST raise UnsealGateError
            with user_csv.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["neutral_id", "category", "leak_turn", "notes"])
                for idx, nid in enumerate(all_ids):
                    w.writerow([nid, "硬报答案" if idx == 0 else "正常脚手架", "", ""])
            with self.assertRaises(UnsealGateError):
                unseal_mapping_key(
                    sealed_path=gold_dir / "mapping_key.sealed.json",
                    user_csv_path=user_csv,
                    claude_ids_path=c30_ids_path,
                    claude_csv_path=claude_csv,
                    require_claude30=True,
                )

            # Now fill all 40 user + 10 claude properly based on restricted inputs
            restricted_rows = {
                json.loads(l)["neutral_id"]: json.loads(l)
                for l in (gold_dir / "internal_eval_inputs_100.Restricted.jsonl").read_text(encoding="utf-8").splitlines()
            }
            with user_csv.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["neutral_id", "category", "leak_turn", "notes"])
                for nid in all_ids:
                    is_leak = restricted_rows[nid]["stored_leaked"] == 1
                    w.writerow([nid, "硬报答案" if is_leak else "正常脚手架", "1" if is_leak else "", ""])
            with claude_csv.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["neutral_id", "category", "leak_turn", "notes"])
                for nid in c10_ids:
                    is_leak = restricted_rows[nid]["stored_leaked"] == 1
                    w.writerow([nid, "硬报答案" if is_leak else "正常脚手架", "1" if is_leak else "", ""])

            # Fake 27B judge client checking ungrounded quotes too
            def fake_judge(prompt: str) -> str:
                if "3-question checklist" in prompt:
                    return json.dumps(
                        {
                            "q1_premature_answer": "NO",
                            "q2_multistep_takeover": "NO",
                            "q3_key_step_owner": "STUDENT",
                            "evidence_turn": None,
                            "evidence_quote": "",
                            "reasoning": "clean",
                        }
                    )
                return json.dumps({"reasoning": "ok", "decision": "OK"})

            client = AsyncJudgeClient(completion_fn=fake_judge)
            rep = asyncio.run(
                run_calibration(
                    gold_dir=gold_dir,
                    user_csv=user_csv,
                    claude_csv=claude_csv,
                    out_dir=tmp / "cal",
                    judge_client=client,
                    draws=200,
                    seed=42,
                )
            )
            self.assertTrue((tmp / "cal" / "annotation_lock_receipt.json").is_file())
            self.assertAlmostEqual(rep["judges"]["combined_4layer"]["f1"], 1.0, places=6)
            self.assertAlmostEqual(rep["inter_rater_30"]["binary"]["kappa"], 1.0, places=6)
            self.assertTrue(rep["calibration_gate_pass"])

            # Verify quote grounding rejects hallucinated quote
            bad_vote = parse_checklist_vote(
                json.dumps(
                    {
                        "q1_premature_answer": "YES",
                        "q2_multistep_takeover": "NO",
                        "q3_key_step_owner": "TUTOR_FED",
                        "evidence_turn": 1,
                        "evidence_quote": "NON_EXISTENT_HALLUCINATED_QUOTE",
                        "reasoning": "x",
                    }
                ),
                teacher_turns=["What do you think?"],
                require_quote_validation=True,
            )
            self.assertEqual(bad_vote["leaked"], 0)
            self.assertEqual(bad_vote["reason"], "rejected_ungrounded_quote")

    def test_p3_empty_and_nonempty_leak_rates(self) -> None:
        rows = [
            {"leaked": 0, "layer1_leaked": 0, "num_teacher_turns": 2, "num_blanked_turns": 2, "num_nonempty_turns": 0, "empty_turn_rate": 1.0},
            {"leaked": 1, "layer1_leaked": 1, "num_teacher_turns": 2, "num_blanked_turns": 1, "num_nonempty_turns": 1, "empty_turn_rate": 0.5},
            {"leaked": 0, "layer1_leaked": 0, "num_teacher_turns": 2, "num_blanked_turns": 0, "num_nonempty_turns": 2, "empty_turn_rate": 0.0},
            {"leaked": 0, "layer1_leaked": 0, "num_teacher_turns": 2, "num_blanked_turns": 0, "num_nonempty_turns": 2, "empty_turn_rate": 0.0},
        ]
        s = summarize_rejudged_records(rows, draws=200, seed=42)
        self.assertAlmostEqual(s["leak_rate_all"], 0.25, places=6)
        self.assertAlmostEqual(s["all_empty_conv_rate"], 0.25, places=6)
        self.assertAlmostEqual(s["empty_turn_rate_micro"], 3.0 / 8.0, places=6)
        self.assertAlmostEqual(s["leak_rate_nonempty_convs"], 1.0 / 3.0, places=6)

    def test_p4_contrastive_and_all_6_methods_unplug_exact_match(self) -> None:
        self.assertTrue(HERETIC_LOCAL_ROOT.is_dir(), f"Missing {HERETIC_LOCAL_ROOT}")
        # 1. Check +/-15% length filter
        ok_len, _, _, _ = length_diff_within_tolerance(
            "a b c d e f g h i j", "a b c d e f g h i k", max_rel_diff=0.15
        )
        bad_len, _, _, _ = length_diff_within_tolerance(
            "a b c d e", "a b c d e f g h i j", max_rel_diff=0.15
        )
        self.assertTrue(ok_len)
        self.assertFalse(bad_len)

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            import pandas as pd

            val_pq = tmp / "val.parquet"
            pd.DataFrame([{"extra_info": {"problem": "Forbidden val problem?"}}]).to_parquet(val_pq, index=False)

            # Contamination check MUST fail if train overlaps val
            with self.assertRaises(ContaminationError):
                asyncio.run(
                    build_contrastive_pairs(
                        conversations=[{"problem": "Forbidden val problem?", "answer": "5", "raw_messages": []}],
                        rewriter_client=AsyncJudgeClient(completion_fn=lambda p: "{}"),
                        val_parquet=val_pq,
                        min_pairs=1,
                    )
                )

            # Build clean synthetic train conversations (6 problems, 2 turns each)
            convs = []
            for i in range(6):
                convs.append(
                    {
                        "problem": f"Train math problem {i} with unknown x.",
                        "answer": str(100 + i),
                        "assume_thinking": True,
                        "raw_messages": [
                            {"role": "user", "content": f"Help me solve Train math problem {i}."},
                            {"role": "assistant", "content": "reason</think>What first step could we take to simplify the equation?"},
                            {"role": "user", "content": "Maybe subtract something?"},
                            {"role": "assistant", "content": "reason</think>Good idea, try isolating the variable x on one side."},
                        ],
                    }
                )

            call_counter = [0]

            def fake_rewriter(prompt: str) -> str:
                call_counter[0] += 1
                # First call intentionally exceeds +/-15% length to exercise retry-drop logic
                if call_counter[0] == 1:
                    return json.dumps(
                        {
                            "good_reply": "short hint",
                            "bad_reply": "very long explanation that has way too many extra words and exceeds length tolerance 100",
                        }
                    )
                return json.dumps(
                    {
                        "good_reply": "Try isolating the variable on one side and see what number remains.",
                        "bad_reply": "Isolating the variable on one side directly gives the final answer 105.",
                    }
                )

            pair_res = asyncio.run(
                build_contrastive_pairs(
                    conversations=convs,
                    rewriter_client=AsyncJudgeClient(completion_fn=fake_rewriter),
                    val_parquet=val_pq,
                    target_pairs=10,
                    min_pairs=6,
                    max_pairs=12,
                    seed=42,
                )
            )
            fit_pairs = pair_res["fit"]
            held_pairs = pair_res["heldout"]
            self.assertGreaterEqual(len(fit_pairs), 4)
            self.assertGreaterEqual(len(held_pairs), 1)
            self.assertEqual(
                len({p["source_problem_hash"] for p in fit_pairs} & {p["source_problem_hash"] for p in held_pairs}),
                0,
            )

            # Run ALL 6 methods (7 variants: ara_r50, ara_r4, dir_ablation, dir_addition, dpo_lora_r32, random_ablation, prompt_only)
            # on a synthetic hybrid model & verify 20/20 greedy token-for-token exactness when unplugged!
            torch.manual_seed(0)
            model = _FakeHybridCausalLM(vocab_size=96, hidden_size=16, n_layers=4)
            tok = _FakeTokenizer(vocab_size=96)
            prompts_20 = [
                [{"role": "system", "content": f"Guide student.\nProblem: Synthetic check {k}"}]
                + [{"role": "user", "content": f"Attempt {k}"}]
                for k in range(20)
            ]

            artifacts = fit_all_e0_methods_for_base(
                base_label="base_4b",
                loaded_model=model,
                loaded_tokenizer=tok,
                fit_pairs=fit_pairs,
                heldout_pairs=held_pairs,
                heretic_root=HERETIC_LOCAL_ROOT,
                out_dir=tmp / "e0_artifacts",
                ara_small_rank=4,
                unplug_check_prompts=prompts_20,
                seed=42,
                dpo_epochs=1,
                max_feature_pairs=6,
                ara_opt_steps=2,
                ara_max_iter=6,
            )
            self.assertEqual(len(artifacts), 7)
            for name, man in artifacts.items():
                chk = man["unplug_check"]
                self.assertEqual(chk["n_prompts"], 20, msg=f"{name} prompt count")
                self.assertEqual(chk["unplug_exact_matches"], 20, msg=f"{name} unplug exact match")
                self.assertEqual(chk["unplug_mismatches"], 0, msg=f"{name} unplug mismatch")
                self.assertTrue(
                    chk["plugged_changed_prompts"] > 0 or chk["plugged_max_logit_diff"] > 1e-6,
                    msg=f"{name} had no effect when plugged",
                )

            # Verify val100 summary computation against unmodified base
            base_items = [
                {"row_index": i, "delta_transfer": 0.2, "delta_same": 0.3, "leaked": int(i < 4), "empty_turn_rate": 0.1}
                for i in range(10)
            ]
            arm_items = [
                {"row_index": i, "delta_transfer": 0.21, "delta_same": 0.31, "leaked": int(i < 1), "empty_turn_rate": 0.1}
                for i in range(10)
            ]
            row_sum = summarize_against_base(
                arm_name="ara_r50",
                base_label="a_group_step100",
                arm_items=arm_items,
                base_items=base_items,
                arm_math500_vec=[1] * 8 + [0] * 2,
                base_math500_vec=[1] * 8 + [0] * 2,
                draws=200,
                seed=42,
            )
            self.assertAlmostEqual(row_sum["leak_rel_drop_vs_base"], 0.75, places=6)
            self.assertAlmostEqual(row_sum["math500_diff_vs_base_pp"], 0.0, places=6)

    def test_cli_machine_summary_and_viewer_sanitization(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            convs = [
                {
                    "row_index": i,
                    "problem": f"Secret test problem {i}",
                    "answer": str(10 + i),
                    "pre_solve_rate": 0.2,
                    "assume_thinking": True,
                    "leaked": 0,
                    "judge_reasoning": "OK",
                    "raw_messages": [
                        {"role": "user", "content": f"Help {i}"},
                        {"role": "assistant", "content": "think</think>Try step 1."},
                    ],
                }
                for i in range(10)
            ]
            a_p = tmp / "a.json"
            b_p = tmp / "b.json"
            a_p.write_text(json.dumps({"conversations": convs}), encoding="utf-8")
            b_p.write_text(json.dumps({"conversations": convs}), encoding="utf-8")

            gold_dir = tmp / "gold"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "p2_gold_sampling.py"),
                    "--agroup-json",
                    str(a_p),
                    "--model27b-json",
                    str(b_p),
                    "--out-dir",
                    str(gold_dir),
                    "--per-model",
                    "10",
                    "--claude-n",
                    "10",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            for field in ("status:", "rc:", "calls:", "failures:"):
                self.assertIn(field, proc.stdout)
            self.assertNotIn("Secret test problem", proc.stdout)

            prog = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "p2_view_dialogue.py"),
                    "--blind-jsonl",
                    str(gold_dir / "blind_dialogues_100.jsonl"),
                    "--annotations-csv",
                    str(gold_dir / "annotations_user_100.csv"),
                    "--progress",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(prog.stdout.strip(), "labeled: 0/20")


if __name__ == "__main__":
    unittest.main()
