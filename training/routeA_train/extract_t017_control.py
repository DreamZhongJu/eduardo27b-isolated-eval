"""Extract step-by-step T-017 control metrics strictly from T-017 training records.

Read-only tool: inspects ONLY T-017 training run directory
(``stage3_formal_20261121_144311``), never test sets or T-020/T-021.
Emits purely numeric summaries (no dialogue, problem, or answer text).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

DEFAULT_T017_RUN_DIR = Path(
    "/data/users/zhujie/bzhang/a_group_pro5000/runs/stage3_formal_20261121_144311"
)

_CONSOLE_STEP_RE = re.compile(r"\bstep:(\d+)\b")
_CONSOLE_KV_RE = re.compile(r"([A-Za-z0-9_./-]+):([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
_BATCH_REWARD_RE = re.compile(
    r"reward/mean=([-+]?\d+(?:\.\d+)?)\s+reward/min=([-+]?\d+(?:\.\d+)?)\s+reward/max=([-+]?\d+(?:\.\d+)?)"
)
_BATCH_GATE_RE = re.compile(
    r"reward/gate_penalty=([-+]?\d+(?:\.\d+)?)\s+\(gated=([-+]?\d+(?:\.\d+)?)\)"
)


def _pick_metric(metrics: Dict[str, Any], candidates: tuple[str, ...]) -> Optional[float]:
    for key in candidates:
        val = metrics.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)):
            return float(val)
    return None


def _parse_trainer_log_numeric(log_path: Path) -> Dict[str, Any]:
    """Parse ONLY numeric console step lines and batch summary lines from trainer.log."""
    console_by_step: Dict[int, Dict[str, float]] = {}
    batch_summaries: List[Dict[str, float]] = []
    if not log_path.is_file():
        return {"console_by_step": console_by_step, "batch_summaries": batch_summaries}

    cur_batch: Dict[str, float] = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            if "step:" in raw_line and ("critic/score/mean" in raw_line or "timing_s/step" in raw_line):
                m_step = _CONSOLE_STEP_RE.search(raw_line)
                if m_step:
                    step_num = int(m_step.group(1))
                    parsed: Dict[str, float] = {}
                    for k, v in _CONSOLE_KV_RE.findall(raw_line):
                        if k == "step":
                            continue
                        try:
                            fv = float(v)
                        except ValueError:
                            continue
                        if math.isfinite(fv):
                            parsed[k] = fv
                    if parsed:
                        console_by_step[step_num] = parsed
            m_rew = _BATCH_REWARD_RE.search(raw_line)
            if m_rew:
                cur_batch = {
                    "reward_mean": float(m_rew.group(1)),
                    "reward_min": float(m_rew.group(2)),
                    "reward_max": float(m_rew.group(3)),
                }
            m_gate = _BATCH_GATE_RE.search(raw_line)
            if m_gate:
                cur_batch["gate_penalty"] = float(m_gate.group(1))
                cur_batch["gated_rate"] = float(m_gate.group(2))
                batch_summaries.append(cur_batch)
                cur_batch = {}

    return {"console_by_step": console_by_step, "batch_summaries": batch_summaries}


def _check_per_item_rewards(run_dir: Path) -> Dict[str, Any]:
    """Check whether per-item rollout reward JSONL exists (expected absent when CONV_DIR=/dev/null)."""
    conv_files = sorted(run_dir.glob("conversations_*.jsonl"))
    if not conv_files:
        return {"available": False, "reason": "EDUARDO_CONV_DIR=/dev/null (no per-rollout files)"}

    total_compliant = 0
    compliant_lt_neg02 = 0
    for cpath in conv_files:
        with cpath.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("split", "train") != "train":
                    continue
                if int(row.get("gated", 0)) == 0:
                    total_compliant += 1
                    if float(row.get("score", 0.0)) < -0.2:
                        compliant_lt_neg02 += 1
    return {
        "available": total_compliant > 0,
        "total_compliant": total_compliant,
        "compliant_lt_neg02": compliant_lt_neg02,
        "compliant_lt_neg02_rate": (
            round(compliant_lt_neg02 / total_compliant, 6) if total_compliant > 0 else None
        ),
    }


def extract_t017_control(run_dir: Path) -> Dict[str, Any]:
    """Extract per-step reward and gated-rate table from a T-017 training run directory."""
    steps_dir = run_dir / "step-statistics"
    if not steps_dir.is_dir():
        raise FileNotFoundError(f"Missing step-statistics dir: {steps_dir}")

    log_info = _parse_trainer_log_numeric(run_dir / "trainer.log")
    console_by_step = log_info["console_by_step"]
    per_item = _check_per_item_rewards(run_dir)

    numbered = sorted(
        (int(p.stem), p)
        for p in steps_dir.glob("*.json")
        if p.stem.isascii() and p.stem.isdigit()
    )
    if not numbered:
        raise ValueError(f"No numbered step JSON files in {steps_dir}")

    observed_key_union: set[str] = set()
    steps_out: Dict[int, Dict[str, Any]] = {}
    gated_history: List[Optional[float]] = []

    for step, spath in numbered:
        raw = json.loads(spath.read_text(encoding="utf-8"))
        metrics = raw.get("metrics", {}) or {}
        observed_key_union.update(metrics.keys())
        cmetrics = console_by_step.get(step, {})

        reward_mean = _pick_metric(
            metrics,
            ("reward/mean", "critic/score/mean", "critic/rewards/mean"),
        )
        if reward_mean is None:
            reward_mean = _pick_metric(cmetrics, ("reward/mean", "critic/score/mean", "critic/rewards/mean"))

        reward_std = _pick_metric(
            metrics,
            ("reward/std", "critic/score/std", "critic/rewards/std"),
        )
        if reward_std is None:
            reward_std = _pick_metric(cmetrics, ("reward/std", "critic/score/std", "critic/rewards/std"))

        reward_min = _pick_metric(
            metrics,
            ("reward/min", "critic/score/min", "critic/rewards/min"),
        )
        if reward_min is None:
            reward_min = _pick_metric(cmetrics, ("reward/min", "critic/score/min", "critic/rewards/min"))

        gated_rate = _pick_metric(
            metrics,
            ("reward/gated_rate", "eduardo/gated_rate", "eduardo/gated", "gated"),
        )
        if gated_rate is None:
            gated_rate = _pick_metric(
                cmetrics,
                ("reward/gated_rate", "eduardo/gated_rate", "eduardo/gated", "gated"),
            )

        gated_history.append(gated_rate)
        recent_valid_gated = [g for g in gated_history[-5:] if g is not None]
        gated_ma5 = (
            round(sum(recent_valid_gated) / len(recent_valid_gated), 6)
            if len(recent_valid_gated) == len(gated_history[-5:]) and recent_valid_gated
            else None
        )

        steps_out[step] = {
            "step": step,
            "reward_mean": round(reward_mean, 6) if reward_mean is not None else None,
            "reward_std": round(reward_std, 6) if reward_std is not None else None,
            "reward_min": round(reward_min, 6) if reward_min is not None else None,
            "gated_rate": round(gated_rate, 6) if gated_rate is not None else None,
            "gated_rate_ma5": gated_ma5,
            "step_seconds": _pick_metric(metrics, ("timing_s/step",)),
        }

    means = [v["reward_mean"] for v in steps_out.values() if v["reward_mean"] is not None]
    mins = [v["reward_min"] for v in steps_out.values() if v["reward_min"] is not None]
    stds = [v["reward_std"] for v in steps_out.values() if v["reward_std"] is not None]
    gateds = [v["gated_rate"] for v in steps_out.values() if v["gated_rate"] is not None]

    return {
        "source_run_dir": str(run_dir),
        "n_steps": len(steps_out),
        "available_reward_related_keys": sorted(
            k for k in observed_key_union if any(tok in k for tok in ("score", "reward", "gate", "judge", "eduardo"))
        ),
        "per_item_rewards": per_item,
        "batch_summaries_in_trainer_log": len(log_info["batch_summaries"]),
        "summary": {
            "reward_mean_first": steps_out[min(steps_out)]["reward_mean"],
            "reward_mean_last": steps_out[max(steps_out)]["reward_mean"],
            "reward_mean_overall": round(sum(means) / len(means), 6) if means else None,
            "reward_min_overall": min(mins) if mins else None,
            "reward_std_present_steps": len(stds),
            "gated_rate_present_steps": len(gateds),
            "gated_rate_overall": round(sum(gateds) / len(gateds), 6) if gateds else None,
        },
        "steps": {str(k): v for k, v in sorted(steps_out.items())},
    }


def format_t017_markdown_table(baseline: Dict[str, Any]) -> str:
    lines = [
        "# T-017 正式训练逐步奖励与门禁对照表（离线训练记录）",
        "",
        f"- 来源目录：`{baseline['source_run_dir']}`（仅用训练记录，未用测试集或 T-020/T-021）",
        f"- 步数：`{baseline['n_steps']}` 步；逐条奖励落盘：`{baseline['per_item_rewards'].get('available')}`"
        + (
            f"（`{baseline['per_item_rewards'].get('reason', '')}`）"
            if not baseline["per_item_rewards"].get("available")
            else f"（合规对话 `< -0.2` 比例=`{baseline['per_item_rewards'].get('compliant_lt_neg02_rate')}`）"
        ),
        f"- `step-statistics` 中奖励/门禁相关字段：`{', '.join(baseline['available_reward_related_keys']) or '无'}`",
        "",
        "| 步号 | 奖励均值 (`reward_mean`) | 奖励标准差 (`reward_std`) | 奖励最小值 (`reward_min`) | 门禁不过比例 (`gated_rate`) | 门禁 5 步滑均 (`gated_ma5`) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    def _fmt(v: Optional[float]) -> str:
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

    for step_key in sorted(baseline["steps"].keys(), key=int):
        row = baseline["steps"][step_key]
        lines.append(
            f"| {row['step']} | {_fmt(row['reward_mean'])} | {_fmt(row['reward_std'])} | "
            f"{_fmt(row['reward_min'])} | {_fmt(row['gated_rate'])} | {_fmt(row['gated_rate_ma5'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_T017_RUN_DIR)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    baseline = extract_t017_control(args.run_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.out_md.write_text(format_t017_markdown_table(baseline), encoding="utf-8")
    s = baseline["summary"]
    print(
        f"T017_CONTROL_OK steps={baseline['n_steps']} "
        f"reward_mean_first={s['reward_mean_first']} reward_mean_last={s['reward_mean_last']} "
        f"reward_std_steps={s['reward_std_present_steps']} gated_steps={s['gated_rate_present_steps']} "
        f"per_item={baseline['per_item_rewards']['available']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
