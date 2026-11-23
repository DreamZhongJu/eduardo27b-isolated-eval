#!/usr/bin/env python3
"""Cross-machine stack check (Pro 5000 vs A100 T-018) before paired comparison.

Checks:
1. Student side (494 problems): Pro 5000 untutored control `ctl` vs A100 `ctl1`:
   - |mean(ctl_same) - mean(ctl1_same)| <= 0.02
   - |mean(ctl_transfer) - mean(ctl1_transfer)| <= 0.02
   - pearson_r(ctl_transfer, ctl1_transfer) >= 0.884 - 0.05 (0.834)
   - pearson_r(ctl_same, ctl1_same) >= 0.867 - 0.05 (0.817)
   - Also verifies pre_solve_rate consistency and fail rate <= 1%.
2. Tutor side (base 20-problem smoke): Pro 5000 base smoke vs A100 T-018 smoke:
   - assume_thinking is True (same as T-018)
   - blanked_teacher_turn_rate within 0.7014 +/- 0.15 ([0.5514, 0.8514])
   - n_student_calls_failed == 0
   - transfer_tested_rate == 1.0

Never prints row_index, problem text, answers, or dialogues.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from export_t018_table import load_t018_table  # noqa: E402
from summarize_author494_pro5000 import (  # noqa: E402
    recompute_author_summary,
    validate_results_doc,
    verify_author_summary_match,
)

T018_SMOKE_EMPTY_RATE = 0.7014
A100_RETEST_R_TRANSFER = 0.884
A100_RETEST_R_SAME = 0.867
MIN_R_TRANSFER = A100_RETEST_R_TRANSFER - 0.05  # 0.834
MIN_R_SAME = A100_RETEST_R_SAME - 0.05  # 0.817
MAX_MEAN_DIFF = 0.02
SMOKE_EMPTY_TOL = 0.15
_EPS = 1e-12


def pearson_r(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx = math.fsum(xs) / n
    my = math.fsum(ys) / n
    num = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.fsum((x - mx) ** 2 for x in xs)
    dy = math.fsum((y - my) ** 2 for y in ys)
    denom = math.sqrt(dx * dy)
    if denom <= _EPS:
        return None
    return max(-1.0, min(1.0, num / denom))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate_stack_equivalence(
    *,
    pro5000_ctl_doc: dict[str, Any],
    t018_by_row: dict[int, dict[str, Any]],
    pro5000_smoke_doc: dict[str, Any],
) -> dict[str, Any]:
    """Run Student-side and Tutor-side cross-machine checks."""
    ctl_metrics = pro5000_ctl_doc.get("metrics") or {}
    ctl_records = pro5000_ctl_doc.get("records") or []
    if not ctl_records:
        raise ValueError("Pro 5000 control JSON missing non-empty 'records'")
    if bool(ctl_metrics.get("exceeds_1pct")):
        raise ValueError("Pro 5000 control fail rate exceeded 1%")

    ctl_by_row: dict[int, dict[str, Any]] = {}
    for rec in ctl_records:
        ridx = int(rec["row_index"])
        if ridx in ctl_by_row:
            raise ValueError("Pro 5000 control JSON has duplicate row_index")
        ctl_by_row[ridx] = rec

    if set(ctl_by_row) != set(t018_by_row):
        raise ValueError(
            f"row_index sets differ between Pro 5000 ctl ({len(ctl_by_row)}) and T-018 table ({len(t018_by_row)})"
        )

    ctl_s_list: list[float] = []
    ctl1_s_list: list[float] = []
    ctl_t_list: list[float] = []
    ctl1_t_list: list[float] = []
    pre_mismatch = 0

    for ridx in sorted(t018_by_row):
        pc = ctl_by_row[ridx]
        at = t018_by_row[ridx]
        if not math.isclose(float(pc["pre_solve_rate"]), at["pre_solve_rate"], abs_tol=1e-9) or not math.isclose(
            float(pc["pre_solve_rate_transfer"]), at["pre_solve_rate_transfer"], abs_tol=1e-9
        ):
            pre_mismatch += 1
        ctl_s_list.append(float(pc["ctl_same"]))
        ctl_t_list.append(float(pc["ctl_transfer"]))
        ctl1_s_list.append(at["ctl1_same_correct"] / at["ctl1_same_total"])
        ctl1_t_list.append(at["ctl1_transfer_correct"] / at["ctl1_transfer_total"])

    if pre_mismatch:
        raise ValueError(f"pre_solve_rate mismatch on {pre_mismatch} rows between Pro 5000 ctl and T-018 table")

    pro_mean_s = statistics.fmean(ctl_s_list)
    a100_mean_s = statistics.fmean(ctl1_s_list)
    pro_mean_t = statistics.fmean(ctl_t_list)
    a100_mean_t = statistics.fmean(ctl1_t_list)
    diff_s = abs(pro_mean_s - a100_mean_s)
    diff_t = abs(pro_mean_t - a100_mean_t)
    r_same = pearson_r(ctl_s_list, ctl1_s_list)
    r_trans = pearson_r(ctl_t_list, ctl1_t_list)

    student_passed = (
        diff_s <= MAX_MEAN_DIFF + _EPS
        and diff_t <= MAX_MEAN_DIFF + _EPS
        and r_same is not None
        and r_same >= MIN_R_SAME - _EPS
        and r_trans is not None
        and r_trans >= MIN_R_TRANSFER - _EPS
    )

    # Tutor side: Pro 5000 base 20-problem smoke
    sm_metrics, sm_records, _ = validate_results_doc(pro5000_smoke_doc)
    sm_re = recompute_author_summary(sm_records)
    verify_author_summary_match(sm_re, sm_metrics)

    assume_thinking = bool(sm_re["assume_thinking"])
    empty_rate = float(sm_re["blanked_teacher_turn_rate"])
    empty_diff = abs(empty_rate - T018_SMOKE_EMPTY_RATE)
    smoke_failed = int(sm_re["n_student_calls_failed"])
    transfer_rate = float(sm_re["transfer_tested_rate"])
    n_smoke_probs = int(sm_re["num_problems"])

    tutor_passed = (
        assume_thinking is True
        and empty_diff <= SMOKE_EMPTY_TOL + _EPS
        and smoke_failed == 0
        and math.isclose(transfer_rate, 1.0, abs_tol=1e-9)
    )

    overall_passed = student_passed and tutor_passed
    return {
        "overall_passed": overall_passed,
        "verdict": "通过" if overall_passed else "不通过",
        "student_side": {
            "passed": student_passed,
            "n_problems": len(t018_by_row),
            "calls_failed": int(ctl_metrics.get("calls_failed", 0)),
            "fail_rate": float(ctl_metrics.get("fail_rate", 0.0)),
            "pro5000_ctl_same_mean": pro_mean_s,
            "a100_ctl1_same_mean": a100_mean_s,
            "abs_diff_same": diff_s,
            "pro5000_ctl_transfer_mean": pro_mean_t,
            "a100_ctl1_transfer_mean": a100_mean_t,
            "abs_diff_transfer": diff_t,
            "pearson_r_same": r_same,
            "min_r_same": MIN_R_SAME,
            "pearson_r_transfer": r_trans,
            "min_r_transfer": MIN_R_TRANSFER,
        },
        "tutor_side": {
            "passed": tutor_passed,
            "n_problems": n_smoke_probs,
            "assume_thinking": assume_thinking,
            "expected_assume_thinking": True,
            "blanked_teacher_turn_rate": empty_rate,
            "t018_smoke_empty_rate": T018_SMOKE_EMPTY_RATE,
            "abs_diff_empty_rate": empty_diff,
            "max_empty_diff": SMOKE_EMPTY_TOL,
            "student_calls_failed": smoke_failed,
            "transfer_tested_rate": transfer_rate,
        },
    }


def format_compare_stack_report(res: dict[str, Any]) -> str:
    st = res["student_side"]
    tt = res["tutor_side"]
    r_s = f"{st['pearson_r_same']:.4f}" if st["pearson_r_same"] is not None else "N/A"
    r_t = f"{st['pearson_r_transfer']:.4f}" if st["pearson_r_transfer"] is not None else "N/A"
    lines = [
        "=" * 84,
        f"换机器检查 (Pro 5000 vs A100 T-018) 结论: {res['verdict']}",
        "-" * 84,
        f"1. 学生侧 (Pro 5000 ctl vs A100 ctl1, n={st['n_problems']}, calls_failed={st['calls_failed']}): "
        f"{'通过' if st['passed'] else '不通过'}",
        f"   原题均值:   Pro5000={st['pro5000_ctl_same_mean']:.4f} vs A100={st['a100_ctl1_same_mean']:.4f} "
        f"| |diff|={st['abs_diff_same']:.4f} (门槛 <= 0.0200)",
        f"   迁移题均值: Pro5000={st['pro5000_ctl_transfer_mean']:.4f} vs A100={st['a100_ctl1_transfer_mean']:.4f} "
        f"| |diff|={st['abs_diff_transfer']:.4f} (门槛 <= 0.0200)",
        f"   原题逐题相关 r={r_s} (门槛 >= {st['min_r_same']:.3f})",
        f"   迁移逐题相关 r={r_t} (门槛 >= {st['min_r_transfer']:.3f})",
        f"2. Tutor 侧 (Pro 5000 base 冒烟 vs A100 T-018 冒烟, n={tt['n_problems']}): "
        f"{'通过' if tt['passed'] else '不通过'}",
        f"   assume_thinking={tt['assume_thinking']} (要求 True)",
        f"   Empty%={tt['blanked_teacher_turn_rate']*100:.2f}% vs T-018 冒烟 70.14% "
        f"(|diff|={tt['abs_diff_empty_rate']*100:.2f}pp, 门槛 <= 15.00pp)",
        f"   student_calls_failed={tt['student_calls_failed']} (要求 0), "
        f"transfer_tested_rate={tt['transfer_tested_rate']:.2%} (要求 100%)",
        "=" * 84,
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-machine stack check (Pro 5000 vs A100 T-018).")
    ap.add_argument("--pro5000-ctl-json", required=True, type=Path)
    ap.add_argument("--t018-table", required=True, type=Path)
    ap.add_argument("--pro5000-smoke-json", required=True, type=Path)
    ap.add_argument("--t018-manifest", type=Path, default=None)
    ap.add_argument("--pro5000-test-parquet", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    try:
        if args.t018_manifest is not None:
            manifest = json.loads(args.t018_manifest.read_text(encoding="utf-8"))
            table_sha = _sha256_file(args.t018_table)
            if table_sha != manifest.get("table_sha256"):
                raise ValueError("t018_per_problem.csv SHA-256 mismatch with t018_manifest.json")
            if args.pro5000_test_parquet is not None:
                pq_sha = _sha256_file(args.pro5000_test_parquet)
                if pq_sha != manifest.get("test_parquet_sha256"):
                    raise ValueError(
                        f"Pro 5000 test.parquet SHA-256 ({pq_sha}) != A100 T-018 ({manifest.get('test_parquet_sha256')})"
                    )
        ctl_doc = json.loads(args.pro5000_ctl_json.read_text(encoding="utf-8"))
        t018_rows = load_t018_table(args.t018_table)
        smoke_doc = json.loads(args.pro5000_smoke_json.read_text(encoding="utf-8"))
        res = evaluate_stack_equivalence(
            pro5000_ctl_doc=ctl_doc,
            t018_by_row=t018_rows,
            pro5000_smoke_doc=smoke_doc,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(format_compare_stack_report(res))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if res["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
