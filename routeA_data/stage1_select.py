#!/usr/bin/env python3
"""Stage 1 pipeline for T-023: inventory, test exclusion, stratified sampling, packaging.

PRIVACY & DATA PARTITIONING GUARANTEE:
- Never prints or writes problem text, answers, or solutions to stdout or manifest files.
- Output JSONL containing problems and answers must be written ONLY to a private local
  directory (never inside any git repository or GitHub release bundle).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from routeA_data.normalize import problem_text_hash
from routeA_data.rasch_ladder import infer_generation_tier

EXPORT_GROUP_PRIORITY: Dict[str, int] = {
    "legacy_overlay": 3,
    "agnes": 2,
    "old": 1,
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_verified_export_tar(tar_path: Path, station: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load all COMPLETE_TRAIN.jsonl records from a station tar.gz after verifying MANIFEST.json."""
    manifest_member = f"{station}/MANIFEST.json"
    records: List[Dict[str, Any]] = []
    actual_hashes: Dict[str, str] = {}
    jsonl_count = 0

    with tarfile.open(tar_path, "r:gz") as tf:
        manifest_raw = tf.extractfile(manifest_member)
        if manifest_raw is None:
            raise FileNotFoundError(f"Missing {manifest_member} inside {tar_path}")
        manifest = json.loads(manifest_raw.read().decode("utf-8"))

        for member in tf.getmembers():
            if not member.isfile() or member.name == manifest_member:
                continue
            blob = tf.extractfile(member).read()
            actual_hashes[member.name] = hashlib.sha256(blob).hexdigest()
            if member.name.endswith(".jsonl"):
                jsonl_count += 1
                for line in blob.decode("utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    row["_station"] = station
                    row["_member"] = member.name
                    records.append(row)

    entries = manifest.get("files") or manifest.get("entries") or []
    verified = 0
    for entry in entries:
        rel = (
            entry.get("archive_path")
            or entry.get("path")
            or entry.get("relpath")
            or entry.get("member")
        )
        expected = entry.get("sha256")
        if rel not in actual_hashes or actual_hashes[rel] != expected:
            raise ValueError(f"Archive member SHA-256 mismatch in {tar_path}: {rel}")
        verified += 1

    if verified != jsonl_count:
        raise ValueError(
            f"Manifest entry count ({verified}) != JSONL file count ({jsonl_count}) in {tar_path}"
        )

    meta = {
        "station": station,
        "archive_name": tar_path.name,
        "archive_sha256": file_sha256(tar_path),
        "jsonl_files_verified": verified,
        "raw_rows": len(records),
    }
    return records, meta


def is_qualified_variant(transfer: Dict[str, Any]) -> bool:
    """Return True if a transfer variant passed blind-solve verification and is complete."""
    if not transfer.get("complete", False):
        return False
    if int(transfer.get("known_n", 0)) != 16:
        return False
    quality = transfer.get("quality") or {}
    if quality.get("quality_status") != "provisional_pass":
        return False
    mr = quality.get("model_review") or {}
    if mr.get("solvability") != "solvable":
        return False
    if mr.get("core_relation") != "related":
        return False
    if mr.get("direct_repeat") != "no":
        return False
    ae_status = (quality.get("answer_equivalence") or {}).get("status")
    se_decision = ((quality.get("semantic_equivalence_review") or {}).get("verdict") or {}).get(
        "decision"
    )
    if ae_status != "correct" and se_decision != "equivalent_complete":
        return False
    if not str(transfer.get("question", "")).strip():
        return False
    if not str(transfer.get("answer", "")).strip():
        return False
    return True


def deduplicate_sources(
    raw_records: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Deduplicate raw export rows by source_id and (source_id, candidate_id)."""
    by_source: Dict[str, Dict[str, Any]] = {}
    duplicate_source_rows = 0
    duplicate_candidate_rows = 0
    unqualified_variants_dropped = 0

    for row in raw_records:
        source = row["source"]
        sid = str(source["source_id"])
        member = str(row.get("_member", ""))
        parts = member.split("/")
        group_name = parts[1] if len(parts) >= 2 else ""
        prio = EXPORT_GROUP_PRIORITY.get(group_name, 0)

        if sid not in by_source:
            by_source[sid] = {
                "source_id": sid,
                "station": str(row.get("_station", "")),
                "direction": str(source.get("direction_band", "")),
                "source_problem": str(source["problem"]),
                "source_answer": str(source["answer"]),
                "source_text_sha256": problem_text_hash(str(source["problem"])),
                "source_pre_solve_rate_old": source.get("pre_solve_rate"),
                "source_llama8b_solve_rate_old": source.get("llama8b_solve_rate"),
                "cands": {},
            }
        else:
            duplicate_source_rows += 1
            if by_source[sid]["source_text_sha256"] != problem_text_hash(str(source["problem"])):
                raise ValueError(f"Conflicting source problem text for duplicate source_id={sid}")

        for transfer in row.get("transfers", []):
            if not is_qualified_variant(transfer):
                unqualified_variants_dropped += 1
                continue
            cid = str(transfer["candidate_id"])
            cands = by_source[sid]["cands"]
            if cid in cands:
                duplicate_candidate_rows += 1
                prev_prio, prev_t = cands[cid]
                if (
                    prev_t["question"] != transfer["question"]
                    or prev_t["answer"] != transfer["answer"]
                    or prev_t.get("correct") != transfer.get("correct")
                ):
                    raise ValueError(
                        f"Conflicting duplicate candidate_id={cid} for source_id={sid}"
                    )
                if prio > prev_prio:
                    cands[cid] = (prio, transfer)
            else:
                cands[cid] = (prio, transfer)

    groups: List[Dict[str, Any]] = []
    sources_without_variants = 0
    for sid in sorted(by_source.keys()):
        entry = by_source[sid]
        variants: List[Dict[str, Any]] = []
        for cid in sorted(entry["cands"].keys()):
            _, t = entry["cands"][cid]
            origin = str(t.get("origin", ""))
            raw_dir = t.get("direction")
            gen_tag = str(raw_dir) if raw_dir else ("paper_transfer" if origin == "paper_transfer" else "unknown")
            gen_tier = infer_generation_tier(origin, raw_dir)
            paper_meta = t.get("paper_metadata") or {}
            old_pre: Dict[str, Any] = {
                "correct": int(t["correct"]),
                "known_n": int(t["known_n"]),
                "pass_rate": float(t["pass_rate"]),
                "student_dtype": t.get("student_dtype"),
                "grade_policy": t.get("grade_policy"),
            }
            if paper_meta:
                old_pre["paper_rate"] = paper_meta.get("paper_rate")
                old_pre["paper_n_scored"] = paper_meta.get("paper_n_scored")
            variants.append(
                {
                    "variant_id": cid,
                    "origin": origin,
                    "problem": str(t["question"]),
                    "answer": str(t["answer"]),
                    "problem_text_sha256": problem_text_hash(str(t["question"])),
                    "generation_tag": gen_tag,
                    "generation_tier": gen_tier,
                    "old_pre_solve": old_pre,
                }
            )
        if not variants:
            sources_without_variants += 1
            continue
        groups.append(
            {
                "source_id": entry["source_id"],
                "station": entry["station"],
                "direction": entry["direction"],
                "source_problem": entry["source_problem"],
                "source_answer": entry["source_answer"],
                "source_text_sha256": entry["source_text_sha256"],
                "source_pre_solve_rate_old": entry["source_pre_solve_rate_old"],
                "source_llama8b_solve_rate_old": entry["source_llama8b_solve_rate_old"],
                "variants": variants,
            }
        )

    # Check uniqueness of normalized source text across distinct source_ids.
    norm_source_hashes = {g["source_text_sha256"] for g in groups}
    stats = {
        "raw_rows": len(raw_records),
        "unique_source_ids": len(by_source),
        "duplicate_source_rows": duplicate_source_rows,
        "duplicate_candidate_rows": duplicate_candidate_rows,
        "unqualified_variants_dropped": unqualified_variants_dropped,
        "sources_without_qualified_variants": sources_without_variants,
        "qualified_source_groups": len(groups),
        "unique_normalized_source_texts": len(norm_source_hashes),
        "qualified_variants_total": sum(len(g["variants"]) for g in groups),
    }
    return groups, stats


def summarize_inventory(groups: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute counts by variant count, direction, origin, and joint strata (no raw text)."""
    by_vcount = collections.Counter()
    by_vcount_3plus = collections.Counter()
    by_direction = collections.Counter()
    by_origin = collections.Counter()
    by_gen_tier = collections.Counter()
    by_strata = collections.Counter()
    by_station = collections.Counter()

    for g in groups:
        nv = len(g["variants"])
        direction = g["direction"]
        by_vcount[str(nv)] += 1
        by_vcount_3plus["3+" if nv >= 3 else str(nv)] += 1
        by_direction[direction] += 1
        by_strata[f"{nv}|{direction}"] += 1
        if g.get("station"):
            by_station[g["station"]] += 1
        for v in g["variants"]:
            by_origin[v["origin"]] += 1
            by_gen_tier[v["generation_tier"]] += 1

    return {
        "n_sources": len(groups),
        "n_variants": sum(len(g["variants"]) for g in groups),
        "by_variant_count": dict(sorted(by_vcount.items(), key=lambda kv: int(kv[0]))),
        "by_variant_count_3plus": dict(sorted(by_vcount_3plus.items())),
        "by_direction": dict(sorted(by_direction.items())),
        "by_origin": dict(sorted(by_origin.items())),
        "by_generation_tier": dict(sorted(by_gen_tier.items())),
        "by_stratum_vcount_direction": dict(sorted(by_strata.items())),
        "by_station": dict(sorted(by_station.items())),
    }


@dataclass(frozen=True)
class TestHashIndex:
    """Normalized problem hash sets for test-set leakage exclusion."""

    self499_source: Set[str]
    self499_variant: Set[str]
    author_test_source: Set[str]
    author_test_variant: Set[str]

    def summary(self) -> Dict[str, int]:
        return {
            "self499_source_hashes": len(self.self499_source),
            "self499_variant_hashes": len(self.self499_variant),
            "author_test_source_hashes": len(self.author_test_source),
            "author_test_variant_hashes": len(self.author_test_variant),
            "union_all_test_hashes": len(
                self.self499_source
                | self.self499_variant
                | self.author_test_source
                | self.author_test_variant
            ),
        }


def build_test_hash_index(
    self_test_rows: Iterable[Dict[str, Any]],
    author_test_rows: Iterable[Dict[str, Any]],
) -> TestHashIndex:
    self_src: Set[str] = set()
    self_var: Set[str] = set()
    for row in self_test_rows:
        sp = str(row.get("source_problem") or "")
        if sp.strip():
            self_src.add(problem_text_hash(sp))
        for tq in row.get("transfer_questions", []):
            vp = str(tq.get("problem") or "")
            if vp.strip():
                self_var.add(problem_text_hash(vp))

    auth_src: Set[str] = set()
    auth_var: Set[str] = set()
    for row in author_test_rows:
        sp = str(row.get("problem") or "")
        if sp.strip():
            auth_src.add(problem_text_hash(sp))
        vp = str(row.get("transfer_problem") or "")
        if vp.strip():
            auth_var.add(problem_text_hash(vp))

    return TestHashIndex(
        self499_source=self_src,
        self499_variant=self_var,
        author_test_source=auth_src,
        author_test_variant=auth_var,
    )


def exclude_test_overlaps(
    groups: Sequence[Dict[str, Any]],
    test_index: TestHashIndex,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Remove any source group whose source OR any variant matches any test hash."""
    clean: List[Dict[str, Any]] = []
    excluded_groups = 0
    excluded_variants = 0
    reason_combo_counts: collections.Counter[str] = collections.Counter()
    individual_reason_counts: collections.Counter[str] = collections.Counter()

    for g in groups:
        sh = g.get("source_text_sha256") or problem_text_hash(g["source_problem"])
        vhs = [
            v.get("problem_text_sha256") or problem_text_hash(v["problem"])
            for v in g["variants"]
        ]
        reasons: List[str] = []
        if sh in test_index.self499_source:
            reasons.append("train_source_hits_self499_source")
        if sh in test_index.self499_variant:
            reasons.append("train_source_hits_self499_variant")
        if sh in test_index.author_test_source:
            reasons.append("train_source_hits_author_test_source")
        if sh in test_index.author_test_variant:
            reasons.append("train_source_hits_author_test_variant")
        if any(vh in test_index.self499_source for vh in vhs):
            reasons.append("train_variant_hits_self499_source")
        if any(vh in test_index.self499_variant for vh in vhs):
            reasons.append("train_variant_hits_self499_variant")
        if any(vh in test_index.author_test_source for vh in vhs):
            reasons.append("train_variant_hits_author_test_source")
        if any(vh in test_index.author_test_variant for vh in vhs):
            reasons.append("train_variant_hits_author_test_variant")

        if reasons:
            excluded_groups += 1
            excluded_variants += len(g["variants"])
            reason_combo_counts["+".join(reasons)] += 1
            for r in reasons:
                individual_reason_counts[r] += 1
        else:
            clean.append(g)

    summary = {
        "input_groups": len(groups),
        "excluded_groups": excluded_groups,
        "excluded_variants": excluded_variants,
        "remaining_clean_groups": len(clean),
        "remaining_clean_variants": sum(len(g["variants"]) for g in clean),
        "reason_combo_counts": dict(sorted(reason_combo_counts.items())),
        "individual_reason_counts": dict(sorted(individual_reason_counts.items())),
    }
    return clean, summary


def allocate_stratified_counts(
    stratum_sizes: Dict[str, int],
    target_total: int,
) -> Dict[str, int]:
    """Proportional largest-remainder (Hamilton) allocation capped by stratum capacities."""
    total_avail = sum(stratum_sizes.values())
    if target_total < 0 or target_total > total_avail:
        raise ValueError(
            f"target_total ({target_total}) out of bounds for available pool ({total_avail})"
        )
    if target_total == 0:
        return {k: 0 for k in sorted(stratum_sizes)}
    if target_total == total_avail:
        return {k: stratum_sizes[k] for k in sorted(stratum_sizes)}

    quotas: Dict[str, float] = {
        k: (stratum_sizes[k] * target_total) / total_avail for k in sorted(stratum_sizes)
    }
    alloc: Dict[str, int] = {k: min(stratum_sizes[k], int(quotas[k])) for k in sorted(stratum_sizes)}
    remainder = target_total - sum(alloc.values())

    # Distribute remaining seats to strata with largest fractional remainder and spare capacity.
    while remainder > 0:
        candidates = [k for k in sorted(stratum_sizes) if alloc[k] < stratum_sizes[k]]
        candidates.sort(key=lambda k: (-(quotas[k] - alloc[k]), k))
        for k in candidates:
            if remainder == 0:
                break
            alloc[k] += 1
            remainder -= 1

    assert sum(alloc.values()) == target_total
    return alloc


def stratified_train_dev_split(
    clean_groups: Sequence[Dict[str, Any]],
    *,
    train_size: int = 2000,
    dev_size: int = 100,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Stratified train/dev sampling by (variant_count, direction) with zero overlap."""
    if train_size + dev_size > len(clean_groups):
        raise ValueError(
            f"Requested train ({train_size}) + dev ({dev_size}) exceeds pool ({len(clean_groups)})"
        )

    strata: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for g in sorted(clean_groups, key=lambda x: x["source_id"]):
        key = f"{len(g['variants'])}|{g['direction']}"
        strata[key].append(g)

    rng = random.Random(seed)

    # 1. Stratified draw for train
    train_alloc = allocate_stratified_counts(
        {k: len(v) for k, v in strata.items()}, train_size
    )
    train_groups: List[Dict[str, Any]] = []
    remaining_strata: Dict[str, List[Dict[str, Any]]] = {}

    for key in sorted(strata.keys()):
        bucket = list(strata[key])
        rng.shuffle(bucket)
        k_train = train_alloc[key]
        train_groups.extend(bucket[:k_train])
        remaining_strata[key] = sorted(bucket[k_train:], key=lambda x: x["source_id"])

    # 2. Stratified draw for dev from the remaining pool
    dev_alloc = allocate_stratified_counts(
        {k: len(v) for k, v in remaining_strata.items()}, dev_size
    )
    dev_groups: List[Dict[str, Any]] = []
    for key in sorted(remaining_strata.keys()):
        bucket = list(remaining_strata[key])
        rng.shuffle(bucket)
        k_dev = dev_alloc[key]
        dev_groups.extend(bucket[:k_dev])

    train_groups.sort(key=lambda x: x["source_id"])
    dev_groups.sort(key=lambda x: x["source_id"])

    # Strict disjointness assertions
    train_ids = {g["source_id"] for g in train_groups}
    dev_ids = {g["source_id"] for g in dev_groups}
    if not train_ids.isdisjoint(dev_ids):
        raise AssertionError("Train and dev source_ids overlap!")

    train_hashes = {
        g.get("source_text_sha256") or problem_text_hash(g["source_problem"])
        for g in train_groups
    }
    dev_hashes = {
        g.get("source_text_sha256") or problem_text_hash(g["source_problem"])
        for g in dev_groups
    }
    if not train_hashes.isdisjoint(dev_hashes):
        raise AssertionError("Train and dev normalized source hashes overlap!")

    split_meta = {
        "seed": seed,
        "stratification_key": "variant_count|direction",
        "allocation_method": "largest_remainder_hamilton",
        "train_target": train_size,
        "dev_target": dev_size,
        "train_allocation_by_stratum": train_alloc,
        "dev_allocation_by_stratum": dev_alloc,
        "train_summary": summarize_inventory(train_groups),
        "dev_summary": summarize_inventory(dev_groups),
        "unselected_clean_sources": len(clean_groups) - len(train_groups) - len(dev_groups),
    }
    return train_groups, dev_groups, split_meta


def format_output_record(group: Dict[str, Any], split: str) -> Dict[str, Any]:
    """Format one source group into the Stage 1 JSONL contract."""
    return {
        "source_id": group["source_id"],
        "split": split,
        "direction": group["direction"],
        "source_problem": group["source_problem"],
        "source_answer": group["source_answer"],
        "source_text_sha256": group["source_text_sha256"],
        "source_pre_solve_rate_old": group.get("source_pre_solve_rate_old"),
        "source_llama8b_solve_rate_old": group.get("source_llama8b_solve_rate_old"),
        "station": group.get("station", ""),
        "variants": [
            {
                "variant_id": v["variant_id"],
                "origin": v["origin"],
                "problem": v["problem"],
                "answer": v["answer"],
                "problem_text_sha256": v["problem_text_sha256"],
                "generation_tag": v["generation_tag"],
                "generation_tier": v["generation_tier"],
                "old_pre_solve": v["old_pre_solve"],
            }
            for v in group["variants"]
        ],
    }


def inspect_incomplete_sources_zip(
    zip_path: Path, completed_source_ids: Set[str]
) -> Dict[str, Any]:
    """Read-only inventory of data/训练来源/训练集未完成来源_7246_入站包.zip (counts only)."""
    with zipfile.ZipFile(zip_path) as zf:
        inner_manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        self_gen_status: collections.Counter[str] = collections.Counter()
        pt_quality: collections.Counter[str] = collections.Counter()
        dir_band: collections.Counter[str] = collections.Counter()
        n_rows = 0
        overlap_completed = 0
        with zf.open("sources.jsonl") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                n_rows += 1
                self_gen_status[str(row.get("self_generated_status"))] += 1
                dir_band[str(row.get("direction_band"))] += 1
                if str(row.get("source_id")) in completed_source_ids:
                    overlap_completed += 1
                for pt in row.get("paper_transfers", []):
                    pt_quality[str(pt.get("quality_status"))] += 1

    return {
        "zip_name": zip_path.name,
        "zip_sha256": file_sha256(zip_path),
        "n_sources": n_rows,
        "overlap_with_completed_source_ids": overlap_completed,
        "self_generated_status_counts": dict(sorted(self_gen_status.items())),
        "direction_band_counts": dict(sorted(dir_band.items())),
        "paper_transfer_quality_status_counts": dict(sorted(pt_quality.items())),
        "qualified_variants_count": 0,
        "inner_manifest_snapshot_source_count": inner_manifest.get("snapshot_source_count"),
    }


def _load_self_test_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    meta: Dict[str, Any] = {
        "file_name": path.name,
        "sha256": file_sha256(path),
        "n_groups": len(rows),
        "n_variants": sum(len(r.get("transfer_questions", [])) for r in rows),
    }
    manifest_path = path.parent.parent / "export_manifest.json"
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = m.get("test_jsonl_sha256")
        if expected and expected != meta["sha256"]:
            raise ValueError(
                f"Self-test SHA-256 mismatch: expected {expected}, got {meta['sha256']}"
            )
        meta["verified_against_export_manifest"] = True
    return rows, meta


def _load_parquet_rows(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import pyarrow.parquet as pq  # local import so pure unit tests don't require pyarrow

    rows = pq.read_table(path).to_pylist()
    nonempty_prob = sum(1 for r in rows if str(r.get("problem") or "").strip())
    nonempty_tprob = sum(1 for r in rows if str(r.get("transfer_problem") or "").strip())
    meta = {
        "file_name": path.name,
        "sha256": file_sha256(path),
        "n_rows": len(rows),
        "nonempty_problem_rows": nonempty_prob,
        "nonempty_transfer_problem_rows": nonempty_tprob,
    }
    return rows, meta


def run_stage1(
    *,
    export_dir: Path,
    self_test_jsonl: Path,
    author_test_parquet: Path,
    output_dir: Path,
    incomplete_sources_zip: Optional[Path] = None,
    author_all_test_parquet: Optional[Path] = None,
    train_size: int = 2000,
    dev_size: int = 100,
    seed: int = 42,
    private_repo_commit: Optional[str] = None,
    author_dataset_revision: str = "5d7529f06359f31b692c92634f5cc11c8b7953af",
) -> Dict[str, Any]:
    """Execute full Stage 1 pipeline and write private JSONL + safe manifest."""
    one_rows, one_meta = load_verified_export_tar(export_dir / "一号站.tar.gz", "one")
    two_rows, two_meta = load_verified_export_tar(export_dir / "二号站.tar.gz", "two")
    all_raw = one_rows + two_rows

    groups, dedup_stats = deduplicate_sources(all_raw)
    inventory_before_exclusion = summarize_inventory(groups)

    incomplete_audit = None
    if incomplete_sources_zip is not None and incomplete_sources_zip.exists():
        incomplete_audit = inspect_incomplete_sources_zip(
            incomplete_sources_zip, {g["source_id"] for g in groups}
        )

    self_test_rows, self_test_meta = _load_self_test_jsonl(self_test_jsonl)
    author_test_rows, author_test_meta = _load_parquet_rows(author_test_parquet)

    test_index = build_test_hash_index(self_test_rows, author_test_rows)
    clean_groups, exclusion_summary = exclude_test_overlaps(groups, test_index)
    inventory_after_exclusion = summarize_inventory(clean_groups)

    # Optional read-only diagnostic check against author's broader all_test split.
    all_test_diagnostic = None
    if author_all_test_parquet is not None and author_all_test_parquet.exists():
        all_test_rows, all_test_meta = _load_parquet_rows(author_all_test_parquet)
        at_index = build_test_hash_index([], all_test_rows)
        _, at_excl = exclude_test_overlaps(clean_groups, at_index)
        all_test_diagnostic = {
            "note": "Diagnostic only; NOT excluded by T-023 Stage 1 scope (which specifies self499 + author test 494).",
            "all_test_file": all_test_meta,
            "clean_pool_hits_on_all_test": at_excl["excluded_groups"],
            "reason_combo_counts": at_excl["reason_combo_counts"],
        }

    train_groups, dev_groups, split_meta = stratified_train_dev_split(
        clean_groups, train_size=train_size, dev_size=dev_size, seed=seed
    )

    if all_test_diagnostic is not None:
        all_test_rows, _ = _load_parquet_rows(author_all_test_parquet)
        at_index = build_test_hash_index([], all_test_rows)
        _, at_train = exclude_test_overlaps(train_groups, at_index)
        _, at_dev = exclude_test_overlaps(dev_groups, at_index)
        all_test_diagnostic["selected_train_hits_on_all_test"] = at_train["excluded_groups"]
        all_test_diagnostic["selected_dev_hits_on_all_test"] = at_dev["excluded_groups"]

    # Write private dataset JSONL and safe manifest.
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "routeA_stage1_dataset.jsonl"
    manifest_path = output_dir / "stage1_manifest.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for g in train_groups:
            f.write(json.dumps(format_output_record(g, "train"), ensure_ascii=False) + "\n")
        for g in dev_groups:
            f.write(json.dumps(format_output_record(g, "dev"), ensure_ascii=False) + "\n")

    jsonl_sha256 = file_sha256(jsonl_path)
    jsonl_bytes = jsonl_path.stat().st_size

    manifest: Dict[str, Any] = {
        "task": "T-023",
        "stage": "stage1",
        "card_version": 2,
        "inbox_read_up_to": 2,
        "sampling_seed": seed,
        "private_repo_commit": private_repo_commit,
        "author_dataset_revision": author_dataset_revision,
        "inputs": {
            "station_one": one_meta,
            "station_two": two_meta,
            "incomplete_sources_audit": incomplete_audit,
            "self_built_test_499": self_test_meta,
            "author_test": author_test_meta,
            "test_hash_index_counts": test_index.summary(),
        },
        "deduplication": dedup_stats,
        "inventory_before_exclusion": inventory_before_exclusion,
        "test_exclusion": exclusion_summary,
        "inventory_after_exclusion": inventory_after_exclusion,
        "split": split_meta,
        "all_test_diagnostic": all_test_diagnostic,
        "outputs": {
            "routeA_stage1_dataset.jsonl": {
                "sha256": jsonl_sha256,
                "bytes": jsonl_bytes,
                "lines": len(train_groups) + len(dev_groups),
                "train_lines": len(train_groups),
                "dev_lines": len(dev_groups),
                "train_variants": sum(len(g["variants"]) for g in train_groups),
                "dev_variants": sum(len(g["variants"]) for g in dev_groups),
                "total_variants": sum(len(g["variants"]) for g in train_groups)
                + sum(len(g["variants"]) for g in dev_groups),
            }
        },
    }

    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    manifest["manifest_sha256"] = file_sha256(manifest_path)
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="T-023 Stage 1: inventory, test exclusion, stratified sampling, packaging."
    )
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--self-test-jsonl", type=Path, required=True)
    parser.add_argument("--author-test-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--incomplete-sources-zip", type=Path, default=None)
    parser.add_argument("--author-all-test-parquet", type=Path, default=None)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--dev-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--private-repo-commit", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest = run_stage1(
        export_dir=args.export_dir,
        self_test_jsonl=args.self_test_jsonl,
        author_test_parquet=args.author_test_parquet,
        output_dir=args.output_dir,
        incomplete_sources_zip=args.incomplete_sources_zip,
        author_all_test_parquet=args.author_all_test_parquet,
        train_size=args.train_size,
        dev_size=args.dev_size,
        seed=args.seed,
        private_repo_commit=args.private_repo_commit,
    )
    # Safe summary only (contains zero problem/answer text).
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
