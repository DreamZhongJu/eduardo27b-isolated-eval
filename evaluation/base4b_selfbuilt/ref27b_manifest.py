#!/usr/bin/env python3
"""Read-only manifest and pre/control reuse validator against 27B evaluation records.

Matches 27B records strictly by (difficulty, source_id) and source content hash,
skips the incomplete (held/partial) sources in 27B so both arms remain paired,
and extracts verified pre/control counts using the frozen summary.py pipeline.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SUMMARY_SHA = "53879d59aa1ce8ae6cf85f4baea62e699c6ea0b21be43ed0a9bf870a5da069f8"
INVENTORY_SHA = "2e35edbc6105c99bd044347673ec4735061da1091149033a49077d803d090f78"

DIFFICULTY_ORDER = ("easy", "medium", "hard")
STAGE_BY_DIFFICULTY = {
    "easy": ("PILOT", "EASY"),
    "medium": ("MEDIUM",),
    "hard": ("HARD",),
}
GROUP_BY_DIFFICULTY = {"easy": "简单", "medium": "中等", "hard": "困难"}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stable_json(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _source_key(source_id: str) -> str:
    return hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:24]


def _import_checked(path: Path, expected_sha: str, mod_name: str):
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"Invalid verified module path: {mod_name}")
    if _sha256_bytes(path.read_bytes()) != expected_sha:
        raise ValueError(f"Module SHA-256 mismatch for {mod_name}")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module spec for {mod_name}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_offline_eval(repo_root: Path):
    app_dir = str(repo_root / "evaluation" / "app")
    off_dir = str(repo_root / "evaluation" / "official")
    for p in (off_dir, app_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    import offline_eval  # noqa: E402

    return offline_eval


def _default_module_paths(repo_root: Path) -> tuple[Path, Path]:
    return (
        repo_root / "evaluation" / "agnes_rejudge" / "summary.py",
        repo_root / "evaluation" / "diagnostics" / "unresolved_inventory.py",
    )


def _validate_ref_roots(ref_root: Path, overlay_root: Path, decisions_root: Path) -> None:
    for p in (ref_root, overlay_root, decisions_root):
        if not p.is_absolute() or p.is_symlink():
            raise ValueError("Reference paths must be non-symlink absolute paths")
    if not ref_root.is_dir():
        raise ValueError("27B reference root directory does not exist")
    if not overlay_root.is_dir():
        raise ValueError("27B overlay directory does not exist")
    if decisions_root.exists() and not decisions_root.is_dir():
        raise ValueError("27B decisions path is not a directory")


def _locate_stage_dir(ref_root: Path, difficulty: str, key: str) -> tuple[str, Path]:
    matches = []
    for stage in STAGE_BY_DIFFICULTY[difficulty]:
        cand = ref_root / stage / "sources" / key
        if cand.exists() or cand.is_symlink():
            if cand.is_symlink() or not cand.is_dir():
                raise ValueError("Invalid reference source directory")
            matches.append((stage, cand))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly 1 stage match in 27B ref for difficulty={difficulty}, got {len(matches)}")
    return matches[0]


def _verify_source_identity(src_dir: Path, row: dict[str, Any], *, require_source_json: bool) -> None:
    source_json = src_dir / "SOURCE.json"
    if not source_json.exists():
        if require_source_json:
            raise ValueError("Missing SOURCE.json in completed 27B source")
        return
    if source_json.is_symlink() or not source_json.is_file():
        raise ValueError("Invalid SOURCE.json in 27B source")
    doc = json.loads(source_json.read_bytes())
    if doc.get("source_id") != row["source_id"]:
        raise ValueError("SOURCE.json source_id mismatch with dataset row")
    expected_sha = _sha256_bytes(_stable_json(row))
    if doc.get("sha256") != expected_sha:
        raise ValueError("SOURCE.json sha256 mismatch with dataset row")
    if doc.get("transfer_target") != row["transfer_target"]:
        raise ValueError("SOURCE.json transfer_target mismatch")
    if doc.get("candidate_id") != row["transfer_candidate_id"]:
        raise ValueError("SOURCE.json candidate_id mismatch")


def _verify_score_input_alignment(src_dir: Path, row: dict[str, Any]) -> None:
    score_input = src_dir / "SCORE_INPUT.json"
    if score_input.is_symlink() or not score_input.is_file():
        raise ValueError("Missing SCORE_INPUT.json in completed 27B source")
    entries = json.loads(score_input.read_bytes())
    if not isinstance(entries, list) or len(entries) != 96:
        raise ValueError("Expected 96 entries in 27B SCORE_INPUT.json")
    for item in entries:
        if item.get("source_id") != row["source_id"]:
            raise ValueError("SCORE_INPUT source_id mismatch")
        if item.get("target") == "source":
            if item.get("problem") != row["problem"] or item.get("reference_answer") != row["answer"]:
                raise ValueError("SCORE_INPUT original problem/answer mismatch")
        elif item.get("target") == row["transfer_target"]:
            if (
                item.get("problem") != row["transfer_problem"]
                or item.get("reference_answer") != row["transfer_answer"]
                or item.get("candidate_id") != row["transfer_candidate_id"]
            ):
                raise ValueError("SCORE_INPUT transfer problem/answer/candidate mismatch")
        else:
            raise ValueError("SCORE_INPUT unexpected target")


def _extract_counts_dict(counts: dict[tuple[str, str], Counter]) -> dict[str, dict[str, dict[str, int]]]:
    out: dict[str, dict[str, dict[str, int]]] = {}
    for target in ("原题", "迁移题"):
        out[target] = {}
        for phase in ("pre", "control", "post"):
            c = counts[(target, phase)]
            item = {
                "correct": int(c.get("correct", 0)),
                "incorrect": int(c.get("incorrect", 0)),
                "unknown": int(c.get("unknown", 0)),
            }
            if sum(item.values()) != 16:
                raise ValueError("27B phase sample count does not sum to 16")
            out[target][phase] = item
    return out


def _fast_load_source_counts(
    inv_mod: Any,
    s_mod: Any,
    ref_root: Path,
    overlay_root: Path,
    decisions_root: Path,
    stage: str,
    key: str,
    row: dict[str, Any],
) -> tuple[str, dict[tuple[str, str], Counter], Counter]:
    """Verify source bindings, RESULTS.json, overlay & decisions without reading per-sample receipts."""
    src = ref_root / stage / "sources" / key
    score_raw, rows = s_mod.read_json(src / "SCORE_INPUT.json", ref_root)
    report_raw, report = s_mod.read_json(ref_root / "AGNES" / stage / "sources" / key / "RESULTS.json", ref_root)
    _, marker = s_mod.read_json(src / "RAW_COMPLETE.json", ref_root)
    _, source_record = s_mod.read_json(src / "SOURCE.json", ref_root)
    if not all(isinstance(doc, dict) for doc in (report, marker, source_record)):
        raise ValueError("source documents")

    sid = row["source_id"]
    score_sha = _sha256_bytes(score_raw)
    report_sha = _sha256_bytes(report_raw)
    if (
        report.get("source_id") != sid
        or marker.get("source_id") != sid
        or marker.get("samples") != 96
        or marker.get("grading_status") != "ungraded"
        or source_record.get("source_id") != sid
        or source_record.get("sha256") != marker.get("source_sha256")
        or (marker.get("score_input_sha256") is not None and marker["score_input_sha256"] != score_sha)
        or report.get("input_sha256") != score_sha
        or report.get("grader") != "agnes_full_response_v2"
        or report.get("model") != inv_mod.MODEL
        or not isinstance(rows, list)
        or len(rows) != 96
        or not isinstance(report.get("results"), list)
        or len(report["results"]) != 96
    ):
        raise ValueError("source binding")

    expected_target = "ours_" + ("easy" if stage in ("PILOT", "EASY") else stage.lower())
    if row["transfer_target"] != expected_target:
        raise ValueError("SCORE_INPUT unexpected target")
    expected_coords = {(t, p, n) for t in ("source", expected_target) for p in s_mod.PHASES for n in range(16)}

    entries: dict[str, dict[str, Any]] = {}
    for item in rows:
        ident = inv_mod.identity(item)
        token = inv_mod.digest({"identity": ident})
        coord = (ident["target"], ident["phase"], ident["index"])
        if ident["source_id"] != sid or coord not in expected_coords or token in entries:
            raise ValueError("input identity")
        if ident["target"] == "source":
            if item.get("problem") != row["problem"] or item.get("reference_answer") != row["answer"]:
                raise ValueError("SCORE_INPUT original problem/answer mismatch")
        else:
            if (
                item.get("problem") != row["transfer_problem"]
                or item.get("reference_answer") != row["transfer_answer"]
                or item.get("candidate_id") != row["transfer_candidate_id"]
            ):
                raise ValueError("SCORE_INPUT transfer problem/answer/candidate mismatch")
        entries[token] = item
    if len(entries) != 96:
        raise ValueError("missing sample")

    originals: dict[str, dict[str, Any]] = {}
    for result in report["results"]:
        ident = inv_mod.identity(result.get("identity"))
        token = inv_mod.digest({"identity": ident})
        if (
            token in originals
            or token not in entries
            or result.get("status") not in ("correct", "incorrect", "unknown")
            or result.get("request_sha256") != inv_mod.digest(inv_mod.payload(entries[token]))
            or result.get("model") != inv_mod.MODEL
            or result.get("grader") != "agnes_full_response_v2"
        ):
            raise ValueError("result identity")
        originals[token] = result
    if set(originals) != set(entries):
        raise ValueError("result coverage")

    recoveries: dict[str, bool] = {}
    unresolved = any(r["status"] == "unknown" for r in originals.values())
    errors: Counter = Counter()
    if unresolved:
        ov_path = overlay_root / stage / key / "OVERLAY.json"
        if ov_path.is_file():
            _, old = s_mod.read_json(ov_path, overlay_root)
            recoveries = {
                inv_mod.digest({"identity": x["identity"]}): x["decision"]
                for x in (old.get("recoveries") or [])
            }

    counts = {
        ("原题" if t == "source" else "迁移题", p): Counter()
        for t in ("source", expected_target)
        for p in s_mod.PHASES
    }
    for token, original in originals.items():
        ident = original["identity"]
        category = ("原题" if ident["target"] == "source" else "迁移题", ident["phase"])
        status = original["status"]
        if status == "unknown":
            chosen = recoveries.get(token)
            candidate_status = ("correct" if chosen else "incorrect") if type(chosen) is bool else None
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
                    if (
                        any(decision.get(k) != v for k, v in binding.items())
                        or decision.get("status") not in ("correct", "incorrect", "unknown")
                    ):
                        raise ValueError("decision binding")
                    ds, method = decision["status"], decision.get("method")
                    if candidate_status is not None:
                        if ds != candidate_status or method != "verified_topbool_overlay" or decision.get("rounds") != 0:
                            raise ValueError("conflicting decision")
                    elif ds != "unknown":
                        rounds = decision.get("rounds")
                        if method != "agnes_rejudge" or type(rounds) is not int or not 1 <= rounds <= 3:
                            raise ValueError("unverified decision")
                        candidate_status = ds
                    elif method != "agnes_rejudge":
                        raise ValueError("invalid unknown")
                except (OSError, ValueError, KeyError, TypeError):
                    errors["补判记录异常作答"] += 1
                    candidate_status = None
            if candidate_status is not None:
                status = candidate_status
        counts[category][status] += 1
    return sid, counts, errors


def _inspect_one_row(
    difficulty: str,
    idx: int,
    row: dict[str, Any],
    *,
    ref_root: Path,
    overlay_root: Path,
    decisions_root: Path,
    s_mod: Any,
    inv_mod: Any,
    verify_receipts: bool = False,
) -> tuple[str, dict[str, Any]]:
    sid = row["source_id"]
    key = _source_key(sid)
    stage, src_dir = _locate_stage_dir(ref_root, difficulty, key)
    raw_complete = src_dir / "RAW_COMPLETE.json"
    source_held = src_dir / "SOURCE_HELD.json"
    results_json = ref_root / "AGNES" / stage / "sources" / key / "RESULTS.json"

    if raw_complete.exists() and source_held.exists():
        raise ValueError("Conflicting RAW_COMPLETE and SOURCE_HELD in 27B source")

    if source_held.exists():
        _verify_source_identity(src_dir, row, require_source_json=False)
        return (
            "held",
            {
                "difficulty": difficulty,
                "tier_zh": GROUP_BY_DIFFICULTY[difficulty],
                "ref_stage": stage,
                "source_key": key,
                "source_id": sid,
                "row_index": idx,
                "reason": "held_in_27b",
            },
        )

    if not raw_complete.is_file() or not results_json.is_file():
        _verify_source_identity(src_dir, row, require_source_json=False)
        return (
            "partial",
            {
                "difficulty": difficulty,
                "tier_zh": GROUP_BY_DIFFICULTY[difficulty],
                "ref_stage": stage,
                "source_key": key,
                "source_id": sid,
                "row_index": idx,
                "reason": "partial_or_ungraded_in_27b",
            },
        )

    _verify_source_identity(src_dir, row, require_source_json=True)
    if verify_receipts:
        _verify_score_input_alignment(src_dir, row)
        loaded_sid, counts, errs = s_mod.load_source(
            inv_mod, ref_root, overlay_root, decisions_root, stage, key
        )
    else:
        loaded_sid, counts, errs = _fast_load_source_counts(
            inv_mod, s_mod, ref_root, overlay_root, decisions_root, stage, key, row
        )
    if errs:
        raise ValueError(f"27B source verification errors for difficulty={difficulty}: {dict(errs)}")
    if loaded_sid != sid:
        raise ValueError("Loaded 27B source_id mismatch")

    return (
        "eligible",
        {
            "difficulty": difficulty,
            "tier_zh": GROUP_BY_DIFFICULTY[difficulty],
            "ref_stage": stage,
            "source_key": key,
            "source_id": sid,
            "row_index": idx,
            "row": row,
            "ref_counts": _extract_counts_dict(counts),
        },
    )


def build_manifest(
    data_path: Path,
    ref_root: Path,
    overlay_root: Path,
    decisions_root: Path,
    *,
    repo_root: Path,
    summary_path: Path | None = None,
    inventory_path: Path | None = None,
    require_exact_totals: bool = True,
    workers: int = 12,
    show_progress: bool = False,
    verify_receipts: bool = False,
) -> dict[str, Any]:
    """Verify 27B reference records against data_path and return eligible + skipped manifest."""
    if not data_path.is_absolute() or not data_path.is_file():
        raise ValueError("Dataset path must be an existing absolute file")
    _validate_ref_roots(ref_root, overlay_root, decisions_root)

    default_sum, default_inv = _default_module_paths(repo_root)
    s_mod = _import_checked(summary_path or default_sum, SUMMARY_SHA, "ref27b_summary")
    inv_mod = _import_checked(inventory_path or default_inv, INVENTORY_SHA, "ref27b_inventory")
    offline_eval = _load_offline_eval(repo_root)

    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    per_tier_counts: dict[str, dict[str, int]] = {}

    for difficulty in DIFFICULTY_ORDER:
        rows = offline_eval.parse_sources(data_path, difficulty)
        expected_keys = {_source_key(r["source_id"]) for r in rows}
        found_keys: set[str] = set()
        for stage in STAGE_BY_DIFFICULTY[difficulty]:
            stage_sources = ref_root / stage / "sources"
            if not stage_sources.is_dir() or stage_sources.is_symlink():
                raise ValueError(f"Missing 27B stage sources dir: {stage}")
            for child in stage_sources.iterdir():
                if child.is_dir():
                    if child.name in found_keys:
                        raise ValueError(f"Duplicate source key across stages for difficulty={difficulty}")
                    found_keys.add(child.name)
        if found_keys != expected_keys:
            raise ValueError(
                f"27B source set mismatch for difficulty={difficulty}: "
                f"expected={len(expected_keys)} found={len(found_keys)}"
            )

        tier_eligible = 0
        tier_held = 0
        tier_partial = 0
        ordered_outcomes: list[tuple[str, dict[str, Any]] | None] = [None] * len(rows)
        max_w = max(1, min(workers, len(rows))) if rows else 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as pool:
            fut_to_idx = {
                pool.submit(
                    _inspect_one_row,
                    difficulty,
                    idx,
                    row,
                    ref_root=ref_root,
                    overlay_root=overlay_root,
                    decisions_root=decisions_root,
                    s_mod=s_mod,
                    inv_mod=inv_mod,
                    verify_receipts=verify_receipts,
                ): idx
                for idx, row in enumerate(rows)
            }
            done = 0
            for fut in concurrent.futures.as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                ordered_outcomes[idx] = fut.result()
                done += 1
                if show_progress and (done % 100 == 0 or done == len(rows)):
                    print(f"[manifest] difficulty={difficulty} verified={done}/{len(rows)}", flush=True)

        for outcome in ordered_outcomes:
            assert outcome is not None
            kind, item = outcome
            if kind == "held":
                tier_held += 1
                skipped.append(item)
            elif kind == "partial":
                tier_partial += 1
                skipped.append(item)
            else:
                tier_eligible += 1
                eligible.append(item)

        per_tier_counts[difficulty] = {
            "total": len(rows),
            "eligible": tier_eligible,
            "skipped": tier_held + tier_partial,
            "held": tier_held,
            "partial": tier_partial,
        }

    if require_exact_totals:
        if len(eligible) != 1489 or len(skipped) != 8:
            raise ValueError(
                f"Expected 1489 eligible and 8 skipped in 27B reference, got eligible={len(eligible)} skipped={len(skipped)}"
            )

    return {
        "data_sha256": _sha256_bytes(data_path.read_bytes()),
        "per_tier_counts": per_tier_counts,
        "eligible_count": len(eligible),
        "skipped_count": len(skipped),
        "eligible": eligible,
        "skipped": skipped,
    }


def select_pilot_entries(manifest: dict[str, Any], per_tier: int = 3) -> list[dict[str, Any]]:
    """Deterministically select the first per_tier eligible sources per difficulty."""
    if per_tier < 0:
        raise ValueError("per_tier must be >= 0")
    if per_tier == 0:
        return list(manifest["eligible"])
    picked: list[dict[str, Any]] = []
    counts = {d: 0 for d in DIFFICULTY_ORDER}
    for entry in manifest["eligible"]:
        d = entry["difficulty"]
        if counts[d] < per_tier:
            picked.append(entry)
            counts[d] += 1
    for d, n in counts.items():
        if n < per_tier:
            raise ValueError(f"Not enough eligible sources for pilot in {d}: {n} < {per_tier}")
    return picked
