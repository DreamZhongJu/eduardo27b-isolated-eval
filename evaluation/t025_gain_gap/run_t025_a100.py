#!/usr/bin/env python3
"""Read-only A100 count & metadata collector for T-025 (Causes C, D, E1, G).

Strictly read-only on:
  - /workspace/results/own-test-499-v1 (+ overlay + rejudge)
  - /workspace/results/base4b-selfbuilt-full-v1 (T-019)
  - /workspace/data/math-transfer-test-499-49d3420/data/test.jsonl
  - /workspace/models/Meta-Llama-3.1-8B-Instruct/tokenizer.json

Never loads GPU models, never calls Agnes, never modifies existing directories,
and never emits problem text, answers, student responses, or source IDs.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import sys
import traceback
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BASE4B_DIR = REPO_ROOT / "evaluation" / "base4b_selfbuilt"
for _p in (str(HERE), str(BASE4B_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from attribution_core import (  # noqa: E402
    DIFFICULTY_ORDER,
    assert_output_sanitized,
    extract_author_boxed_pred,
    summarize_length_and_format_records,
)
from ref27b_manifest import (  # noqa: E402
    INVENTORY_SHA,
    SUMMARY_SHA,
    _default_module_paths,
    _import_checked,
    build_manifest,
)
from summarize_base4b import STAGE_BY_DIFF, extract_safe_failure_summary, normalized_gain, p_value  # noqa: E402

SAFE_RUN_CONFIG_KEYS = (
    "protocol",
    "transfer_difficulty",
    "teacher_dtype",
    "teacher_tp",
    "student_dtype",
    "engine",
    "teacher_max_model_len",
    "teacher_max_tokens",
    "teacher_timeout",
    "student_max_tokens",
    "student_samples_per_condition",
    "student_temperature",
    "grading",
    "installed_versions",
)


def _load_fast_tokenizer(tokenizer_dir: Path):
    """Load Llama-3.1-8B-Instruct tokenizer on CPU without loading model weights."""
    tok_json = tokenizer_dir / "tokenizer.json"
    if not tok_json.is_file():
        raise ValueError(f"Missing tokenizer.json in {tokenizer_dir}")
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(tok_json))

        def _batch_count_and_prefix800(texts: list[str]) -> list[tuple[int, str]]:
            encs = tok.encode_batch(texts, add_special_tokens=False)
            out: list[tuple[int, str]] = []
            for enc in encs:
                ids = enc.ids
                n_tok = len(ids)
                pref = tok.decode(ids[:800], skip_special_tokens=False) if n_tok > 800 else ""
                out.append((n_tok, pref))
            return out

        return _batch_count_and_prefix800, "rust_tokenizers"
    except Exception:
        from transformers import AutoTokenizer

        hf_tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, use_fast=True)

        def _batch_count_and_prefix800_hf(texts: list[str]) -> list[tuple[int, str]]:
            enc = hf_tok(texts, add_special_tokens=False, return_attention_mask=False)
            out: list[tuple[int, str]] = []
            for ids in enc["input_ids"]:
                n_tok = len(ids)
                pref = hf_tok.decode(ids[:800], skip_special_tokens=False) if n_tok > 800 else ""
                out.append((n_tok, pref))
            return out

        return _batch_count_and_prefix800_hf, "transformers_fast"


def _resolve_27b_per_sample_agnes_status(
    inv_mod: Any,
    s_mod: Any,
    ref_root: Path,
    overlay_root: Path,
    decisions_root: Path,
    stage: str,
    key: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Return (score_input_rows, {token: final_agnes_status}) matching ref27b_manifest.py."""
    src = ref_root / stage / "sources" / key
    score_raw, rows = s_mod.read_json(src / "SCORE_INPUT.json", ref_root)
    report_raw, report = s_mod.read_json(ref_root / "AGNES" / stage / "sources" / key / "RESULTS.json", ref_root)
    score_sha = hashlib.sha256(score_raw).hexdigest()
    report_sha = hashlib.sha256(report_raw).hexdigest()

    originals = {}
    for result in report["results"]:
        ident = inv_mod.identity(result.get("identity"))
        token = inv_mod.digest({"identity": ident})
        originals[token] = result

    recoveries: dict[str, bool] = {}
    if any(r["status"] == "unknown" for r in originals.values()):
        ov_path = overlay_root / stage / key / "OVERLAY.json"
        if ov_path.is_file():
            _, old = s_mod.read_json(ov_path, overlay_root)
            recoveries = {
                inv_mod.digest({"identity": x["identity"]}): x["decision"]
                for x in (old.get("recoveries") or [])
            }

    final_by_token: dict[str, str] = {}
    for token, original in originals.items():
        ident = original["identity"]
        status = original["status"]
        if status == "unknown":
            chosen = recoveries.get(token)
            cand = ("correct" if chosen else "incorrect") if type(chosen) is bool else None
            dec_path = decisions_root / "decisions" / stage / key / token / "DECISION.json"
            if dec_path.exists():
                try:
                    _, decision = s_mod.read_json(dec_path, decisions_root)
                    binding = {
                        "schema": s_mod.SCHEMA,
                        "stage": stage,
                        "source_key": key,
                        "identity": ident,
                        "request_sha256": original["request_sha256"],
                        "source_result_sha256": report_sha,
                        "input_sha256": score_sha,
                    }
                    if all(decision.get(k) == v for k, v in binding.items()) and decision.get("status") in (
                        "correct",
                        "incorrect",
                        "unknown",
                    ):
                        ds, method = decision["status"], decision.get("method")
                        if cand is None and ds != "unknown" and method == "agnes_rejudge":
                            rounds = decision.get("rounds")
                            if type(rounds) is int and 1 <= rounds <= 3:
                                cand = ds
                except Exception:
                    cand = None
            if cand is not None:
                status = cand
        final_by_token[token] = status
    return rows, final_by_token


def collect_27b_a100_data(
    manifest: dict[str, Any],
    *,
    repo_root: Path,
    ref_root: Path,
    overlay_root: Path,
    decisions_root: Path,
    tok_batch_fn: Any,
) -> dict[str, Any]:
    """Scan 27B RUN_CONFIGURATION, ROLLOUT_RESULT, and 142,944 SCORE_INPUT samples on A100."""
    app_dir = str(repo_root / "evaluation" / "app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    import protocol  # noqa: E402

    sum_p, inv_p = _default_module_paths(repo_root)
    s_mod = _import_checked(sum_p, SUMMARY_SHA, "t025_ref27b_summary")
    inv_mod = _import_checked(inv_p, INVENTORY_SHA, "t025_ref27b_inventory")

    # 1) RUN_CONFIGURATION.json per stage
    stage_configs: dict[str, Any] = {}
    for stage in ("PILOT", "EASY", "MEDIUM", "HARD"):
        cfg_p = ref_root / stage / "RUN_CONFIGURATION.json"
        if cfg_p.is_file():
            raw_cfg = json.loads(cfg_p.read_bytes())
            stage_configs[stage] = {k: raw_cfg.get(k) for k in SAFE_RUN_CONFIG_KEYS if k in raw_cfg}

    # Representative config check across stages
    rep_cfg = stage_configs.get("EASY") or next(iter(stage_configs.values()), {})

    # 2) Rollout diagnostics & per-sample records across 1,489 eligible sources
    roll_by_tier: dict[str, Counter] = {d: Counter() for d in DIFFICULTY_ORDER}
    end_reasons_total: Counter = Counter()
    assume_vals: set[bool] = set()
    all_sample_records: list[dict[str, Any]] = []
    tier_r1_verify: dict[str, list[float]] = {d: [] for d in DIFFICULTY_ORDER}

    entries = manifest["eligible"]
    for idx, entry in enumerate(entries, 1):
        diff = entry["difficulty"]
        stage = entry["ref_stage"]
        key = entry["source_key"]
        src_dir = ref_root / stage / "sources" / key
        dc = roll_by_tier[diff]
        dc["sources"] += 1

        roll_p = src_dir / "dialogue" / "ROLLOUT_RESULT.json"
        if roll_p.is_file():
            roll = json.loads(roll_p.read_bytes())
            dc["rollout_files"] += 1
            assume = bool(roll.get("assume_thinking", True))
            assume_vals.add(assume)
            er = str(roll.get("end_reason") or "unknown")
            dc[f"end_{er}"] += 1
            end_reasons_total[er] += 1
            if roll.get("eoc_used"):
                dc["eoc_used"] += 1
            for t in roll.get("thinking_tokens") or []:
                dc["thinking_tokens_sum"] += int(t)
                dc["thinking_tokens_count"] += 1
            for msg in roll.get("raw_messages") or []:
                if msg.get("role") == "teacher":
                    raw_c = msg.get("content") or ""
                    dc["teacher_turns"] += 1
                    if not protocol.strip_teacher_controls(raw_c, assume_thinking=assume).strip():
                        dc["blanked_turns"] += 1
                    if "</think>" not in raw_c:
                        dc["unterminated_think_turns"] += 1

        dc["student_sample_errors"] += len(list((src_dir / "samples").glob("*/*/*/ERROR.json")))

        rows, final_status = _resolve_27b_per_sample_agnes_status(
            inv_mod, s_mod, ref_root, overlay_root, decisions_root, stage, key
        )
        texts = [r.get("student_response") or "" for r in rows]
        tok_info = tok_batch_fn(texts)

        by_tp_cnt: dict[tuple[str, str], Counter] = {}
        for r, (n_tok, pref800) in zip(rows, tok_info):
            ident = inv_mod.identity(r)
            token = inv_mod.digest({"identity": ident})
            st = final_status[token]
            tgt = "orig" if r["target"] == "source" else "trans"
            phase = r["phase"]
            by_tp_cnt.setdefault((tgt, phase), Counter())[st] += 1
            pred_full, fmt_err = extract_author_boxed_pred(r.get("student_response") or "")
            pred_800, _ = extract_author_boxed_pred(pref800) if n_tok > 800 else (pred_full, fmt_err)
            all_sample_records.append(
                {
                    "difficulty": diff,
                    "target": tgt,
                    "phase": phase,
                    "tokens": int(n_tok),
                    "finish_reason": str(r.get("finish_reason") or ""),
                    "agnes_status": st,
                    "ar_incorrect_format": bool(fmt_err),
                    "boxed_pred_full": pred_full,
                    "boxed_pred_prefix800": pred_800,
                }
            )

        # Verify exact match with manifest["ref_counts"] for this source
        rc_trans_pre = entry["ref_counts"]["迁移题"]["pre"]
        rc_trans_post = entry["ref_counts"]["迁移题"]["post"]
        if dict(by_tp_cnt[("trans", "pre")]) != {k: v for k, v in rc_trans_pre.items() if v > 0}:
            raise ValueError("27B A100 sample-level Agnes status mismatch vs manifest ref_counts (pre)")
        if dict(by_tp_cnt[("trans", "post")]) != {k: v for k, v in rc_trans_post.items() if v > 0}:
            raise ValueError("27B A100 sample-level Agnes status mismatch vs manifest ref_counts (post)")
        tier_r1_verify[diff].append(
            normalized_gain(p_value(rc_trans_pre, True), p_value(rc_trans_post, True))
        )

        if idx % 300 == 0 or idx == len(entries):
            print(f"[t025_a100] 27b_scanned={idx}/{len(entries)}", flush=True)

    tot_sources = sum(roll_by_tier[d]["sources"] for d in DIFFICULTY_ORDER)
    tot_roll = sum(roll_by_tier[d]["rollout_files"] for d in DIFFICULTY_ORDER)
    tot_turns = sum(roll_by_tier[d]["teacher_turns"] for d in DIFFICULTY_ORDER)
    tot_blank = sum(roll_by_tier[d]["blanked_turns"] for d in DIFFICULTY_ORDER)
    tot_unterm = sum(roll_by_tier[d]["unterminated_think_turns"] for d in DIFFICULTY_ORDER)
    tot_eoc = sum(roll_by_tier[d]["eoc_used"] for d in DIFFICULTY_ORDER)
    tot_trunc = int(end_reasons_total.get("token_budget", 0))
    tot_max_turns = int(end_reasons_total.get("max_teacher_turns", 0))
    tot_think_sum = sum(roll_by_tier[d]["thinking_tokens_sum"] for d in DIFFICULTY_ORDER)
    tot_think_cnt = sum(roll_by_tier[d]["thinking_tokens_count"] for d in DIFFICULTY_ORDER)

    rollout_diag = {
        "eligible_sources": tot_sources,
        "rollout_files_found": tot_roll,
        "assume_thinking_values": sorted(assume_vals),
        "end_reasons": dict(end_reasons_total),
        "teacher_turns": tot_turns,
        "blanked_turns": tot_blank,
        "blanked_turn_rate": (tot_blank / tot_turns) if tot_turns > 0 else None,
        "unterminated_think_turns": tot_unterm,
        "unterminated_think_rate": (tot_unterm / tot_turns) if tot_turns > 0 else None,
        "token_budget_truncated_convs": tot_trunc,
        "token_budget_truncated_rate": (tot_trunc / tot_roll) if tot_roll > 0 else None,
        "eoc_convs": tot_eoc,
        "eoc_rate": (tot_eoc / tot_roll) if tot_roll > 0 else None,
        "max_teacher_turns_convs": tot_max_turns,
        "max_teacher_turns_rate": (tot_max_turns / tot_roll) if tot_roll > 0 else None,
        "thinking_tokens_avg_per_turn": (tot_think_sum / tot_think_cnt) if tot_think_cnt > 0 else None,
        "per_tier": {d: dict(roll_by_tier[d]) for d in DIFFICULTY_ORDER},
    }

    len_fmt_summary = summarize_length_and_format_records(all_sample_records, model_label="eduardo27b")
    # Strip boxed_pred strings from any retained object (summarize already only keeps integer counts!)
    return {
        "stage_run_configurations": stage_configs,
        "run_configuration": rep_cfg,
        "pre_post_identical_student_and_grader": True,
        "omit_teacher_turns_in_post": True,
        "rollout_diagnostics": rollout_diag,
        "r1_spot_check": {d: sum(v) / len(v) for d, v in tier_r1_verify.items()},
        "length_and_format": len_fmt_summary,
    }


def collect_base4b_a100_data(
    manifest: dict[str, Any],
    *,
    base4b_dir: Path,
    tok_batch_fn: Any,
) -> dict[str, Any]:
    """Scan T-019 base 4B post-test SCORE_INPUT.json + AGNES RESULTS.json on A100."""
    sample_records: list[dict[str, Any]] = []
    sources_scanned = 0
    held_skipped = 0

    for entry in manifest["eligible"]:
        diff = entry["difficulty"]
        stage = STAGE_BY_DIFF[diff]
        key = entry["source_key"]
        exp_tgt = entry["row"]["transfer_target"]
        src_dir = base4b_dir / stage / "sources" / key
        if (src_dir / "SOURCE_HELD.json").exists() or not (src_dir / "RAW_COMPLETE.json").is_file():
            held_skipped += 1
            continue
        score_p = src_dir / "SCORE_INPUT.json"
        agnes_p = base4b_dir / "AGNES" / stage / "sources" / key / "RESULTS.json"
        if not score_p.is_file() or not agnes_p.is_file():
            continue

        rows = json.loads(score_p.read_bytes())
        ag_doc = json.loads(agnes_p.read_bytes())
        st_by_coord: dict[tuple[str, str, int], str] = {}
        for item in ag_doc.get("results") or []:
            ident = item.get("identity") or item
            st_by_coord[(ident["target"], ident["phase"], int(ident["index"]))] = item["status"]

        texts = [r.get("student_response") or "" for r in rows]
        tok_info = tok_batch_fn(texts)
        for r, (n_tok, pref800) in zip(rows, tok_info):
            t_raw = r["target"]
            if t_raw != "source" and t_raw != exp_tgt:
                raise ValueError("Unexpected target in base4b SCORE_INPUT.json")
            tgt = "orig" if t_raw == "source" else "trans"
            st = st_by_coord[(t_raw, r["phase"], int(r["index"]))]
            pred_full, fmt_err = extract_author_boxed_pred(r.get("student_response") or "")
            pred_800, _ = extract_author_boxed_pred(pref800) if n_tok > 800 else (pred_full, fmt_err)
            sample_records.append(
                {
                    "difficulty": diff,
                    "target": tgt,
                    "phase": r["phase"],
                    "tokens": int(n_tok),
                    "finish_reason": str(r.get("finish_reason") or ""),
                    "agnes_status": st,
                    "ar_incorrect_format": bool(fmt_err),
                    "boxed_pred_full": pred_full,
                    "boxed_pred_prefix800": pred_800,
                }
            )
        sources_scanned += 1

    return {
        "sources_scanned": sources_scanned,
        "held_or_incomplete_skipped": held_skipped,
        "length_and_format": summarize_length_and_format_records(sample_records, model_label="base"),
    }


def format_a100_report(payload: dict[str, Any]) -> str:
    """Render concise human-readable A100 summary (zero raw text, zero IDs)."""
    lines: list[str] = []
    lines.append("=" * 88)
    lines.append("T-025 A100 Read-Only Attribution Counts (Causes C, D, E1, G)")
    lines.append(f"Tokenizer backend: {payload['tokenizer_backend']} | sha256={payload['tokenizer_json_sha256'][:16]}...")
    d27 = payload["eduardo27b"]
    rd = d27["rollout_diagnostics"]
    cfg = d27["run_configuration"]
    lines.append(
        f"27B Config: tutor_dtype={cfg.get('teacher_dtype')} student_dtype={cfg.get('student_dtype')} "
        f"tutor_max_tok={cfg.get('teacher_max_tokens')} stu_max_tok={cfg.get('student_max_tokens')} "
        f"stu_temp={cfg.get('student_temperature')} k={cfg.get('student_samples_per_condition')} "
        f"versions={cfg.get('installed_versions')}"
    )
    emp_s = f"{rd['blanked_turn_rate']*100:.2f}%" if rd["blanked_turn_rate"] is not None else "N/A"
    trc_s = f"{rd['token_budget_truncated_rate']*100:.2f}%" if rd["token_budget_truncated_rate"] is not None else "N/A"
    eoc_s = f"{rd['eoc_rate']*100:.2f}%" if rd["eoc_rate"] is not None else "N/A"
    thk_s = f"{rd['thinking_tokens_avg_per_turn']:.1f}" if rd["thinking_tokens_avg_per_turn"] is not None else "N/A"
    lines.append(
        f"27B Rollout (n={rd['rollout_files_found']}): assume_thinking={rd['assume_thinking_values']} "
        f"Empty%={emp_s} ({rd['blanked_turns']}/{rd['teacher_turns']}) "
        f"Seq.trunc%={trc_s} EoC%={eoc_s} Think_tok/turn={thk_s} end_reasons={rd['end_reasons']}"
    )
    lines.append("-" * 88)
    lines.append("Cause C Pre-check: 27B Author-Rule \\boxed{} Missing/Format-Error vs Agnes Unknown")
    for row in d27["length_and_format"]["by_tier_target_phase"]:
        if row["phase"] in ("pre", "post"):
            lines.append(
                f"  phase={row['phase']:<4} tgt={row['target']:<5} tier={row['difficulty']:<9} n={row['n']:>5} "
                f"ar_no_boxed={row['ar_incorrect_format']:>4} ({row['ar_incorrect_format_rate']*100:5.2f}%) "
                f"agnes_unk={row['unknown']:>3} ({row['agnes_unknown_rate']*100:5.2f}%) "
                f"agnes_acc={row['acc_unknown_as_wrong']*100:6.2f}%"
            )
    lines.append("-" * 88)
    for m_key, m_title in (("eduardo27b", "27B"), ("base4b", "4B-base (T-019)")):
        m_doc = payload.get(m_key)
        if not m_doc:
            continue
        lines.append(f"Cause D Post-test Length (>800 vs <=800 tokens) — {m_title}:")
        for r in m_doc["length_and_format"]["post_length_screening"]:
            le = r["le_800"]
            gt = r["gt_800"]
            bd = r["gt_800_boxed_diagnostics"]
            le_acc = f"{le['acc_unknown_as_wrong']*100:6.2f}%" if le["acc_unknown_as_wrong"] is not None else "   N/A"
            gt_acc = f"{gt['acc_unknown_as_wrong']*100:6.2f}%" if gt["acc_unknown_as_wrong"] is not None else "   N/A"
            le_unk = f"{le['agnes_unknown_rate']*100:5.2f}%" if le["agnes_unknown_rate"] is not None else "  N/A"
            gt_unk = f"{gt['agnes_unknown_rate']*100:5.2f}%" if gt["agnes_unknown_rate"] is not None else "  N/A"
            lines.append(
                f"  {r['group']:<18} N={r['n_total']:>5} >800={r['n_gt_800']:>4} ({r['gt_800_rate']*100:5.2f}%) | "
                f"<=800: acc={le_acc} unk={le_unk} | >800: acc={gt_acc} unk={gt_unk} "
                f"[box_only_801_1024={bd['boxed_only_in_801_1024_lost_if_trunc']}, box_overwritten={bd['boxed_overwritten_in_801_1024']}]"
            )
    lines.append("=" * 88)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only A100 count collector for T-025.")
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("/workspace/data/math-transfer-test-499-49d3420/data/test.jsonl"))
    ap.add_argument("--ref-root", type=Path, default=Path("/workspace/results/own-test-499-v1"))
    ap.add_argument("--overlay", type=Path, default=Path("/workspace/results/agnes-topbool-recovery-v1"))
    ap.add_argument("--decisions", type=Path, default=Path("/workspace/results/agnes-rejudge-v1"))
    ap.add_argument("--base4b-dir", type=Path, default=None)
    ap.add_argument("--student-tokenizer-dir", type=Path, default=Path("/workspace/models/Meta-Llama-3.1-8B-Instruct"))
    ap.add_argument("--json-out", type=Path, required=True)
    ap.add_argument("--allow-small-test", action="store_true")
    args = ap.parse_args()

    tok_fn, tok_backend = _load_fast_tokenizer(args.student_tokenizer_dir)
    tok_sha = hashlib.sha256((args.student_tokenizer_dir / "tokenizer.json").read_bytes()).hexdigest()

    manifest = build_manifest(
        args.data,
        args.ref_root,
        args.overlay,
        args.decisions,
        repo_root=args.repo,
        require_exact_totals=not args.allow_small_test,
        show_progress=True,
    )

    d27 = collect_27b_a100_data(
        manifest,
        repo_root=args.repo,
        ref_root=args.ref_root,
        overlay_root=args.overlay,
        decisions_root=args.decisions,
        tok_batch_fn=tok_fn,
    )
    dbase = None
    if args.base4b_dir is not None and args.base4b_dir.is_dir():
        dbase = collect_base4b_a100_data(manifest, base4b_dir=args.base4b_dir, tok_batch_fn=tok_fn)

    payload: dict[str, Any] = {
        "schema": "t025_a100_counts_v1",
        "eligible_count": manifest["eligible_count"],
        "skipped_count": manifest["skipped_count"],
        "per_tier_counts": manifest["per_tier_counts"],
        "tokenizer_backend": tok_backend,
        "tokenizer_json_sha256": tok_sha,
        "eduardo27b": d27,
        "base4b": dbase,
    }

    json_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    report_text = format_a100_report(payload)

    # Strict sanitization verification before writing or printing anything
    forbidden_ids = [e["source_id"] for e in manifest["eligible"]] + [s["source_id"] for s in manifest["skipped"]]
    forbidden_snippets = []
    for e in manifest["eligible"][:50]:
        r = e["row"]
        forbidden_snippets.extend([r.get("problem", ""), r.get("transfer_problem", ""), r.get("answer", "")])
    assert_output_sanitized(json_text, forbidden_exact_ids=forbidden_ids, forbidden_text_snippets=forbidden_snippets)
    assert_output_sanitized(report_text, forbidden_exact_ids=forbidden_ids, forbidden_text_snippets=forbidden_snippets)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json_text, encoding="utf-8")
    json_sha = hashlib.sha256(json_text.encode("utf-8")).hexdigest()
    compressed_b64 = base64.b64encode(zlib.compress(json_text.encode("utf-8"), level=9)).decode("ascii")

    print(report_text)
    print(f"T025_A100_JSON_SHA256={json_sha}")
    print("T025_A100_B64_BEGIN")
    print(compressed_b64)
    print("T025_A100_B64_END")
    print(f"T025_A100_DONE json_out={args.json_out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print(extract_safe_failure_summary(traceback.format_exc()), file=sys.stderr)
        raise SystemExit(1)
