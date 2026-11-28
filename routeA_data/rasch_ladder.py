"""Rasch difficulty estimation and ladder sorting for Route A variants (T-023)."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

# Tag ordering from easiest (0) to hardest (3) for tie-breaking when pass counts
# differ by less than 1 rollout.
GENERATION_TIER_ORDER: Dict[str, int] = {
    "easier": 0,
    "nearby": 1,
    "paper_transfer": 2,
    "harder": 3,
}

# Mapping from self_generated variant `direction` strings to coarse generation tier.
DIRECTION_STRING_TO_TIER: Dict[str, str] = {
    "reduce non-core preparation": "easier",
    "make a necessary intermediate explicit without giving the core answer": "easier",
    "remove a non-core case distinction": "easier",
    "nearby variation with a different necessary reasoning structure": "nearby",
    "require recovery of a necessary relationship within the same core knowledge": "harder",
    "combine two necessary uses of the same core concept": "harder",
    "add one necessary dependency within the same core knowledge": "harder",
}


def infer_generation_tier(origin: str, direction_label: str | None) -> str:
    """Infer coarse difficulty tier from variant origin and generation direction label."""
    if origin == "paper_transfer":
        return "paper_transfer"
    if not direction_label:
        return "nearby"
    label = direction_label.strip().lower()
    if label in GENERATION_TIER_ORDER:
        return label
    if label in DIRECTION_STRING_TO_TIER:
        return DIRECTION_STRING_TO_TIER[label]
    if any(tok in label for tok in ("easier", "simplif", "reduce", "remove", "explicit")):
        return "easier"
    if any(tok in label for tok in ("harder", "add", "combine", "recover", "dependency")):
        return "harder"
    return "nearby"


def compute_smoothed_p_and_b(
    n_correct_hybrid: int,
    n_scored: int = 16,
) -> Tuple[float, float]:
    """Compute Jeffreys/Laplace-like smoothed pass rate p and Rasch difficulty b.

    Per T-023 Stage 2 step 4:
        p = (n_correct_hybrid + 0.5) / (n_scored + 1)   # = (n_correct_hybrid + 0.5) / 17 for 16 rollouts
        b = ln((1 - p) / p)
    Failed rollouts (n_failed) are excluded from the denominator n_scored.
    """
    if n_scored <= 0:
        raise ValueError(f"n_scored must be positive, got {n_scored}")
    if n_correct_hybrid < 0 or n_correct_hybrid > n_scored:
        raise ValueError(
            f"n_correct_hybrid ({n_correct_hybrid}) must be in [0, n_scored={n_scored}]"
        )
    p = (n_correct_hybrid + 0.5) / (n_scored + 1.0)
    b = math.log((1.0 - p) / p)
    return p, b


def _effective_correct_16(variant: Dict[str, Any]) -> float:
    """Return effective correct count on a 16-rollout scale for tie detection."""
    if "n_scored" in variant and variant["n_scored"] is not None:
        n_scored = int(variant["n_scored"])
    elif "n_failed" in variant and variant["n_failed"] is not None:
        n_scored = 16 - int(variant["n_failed"])
    else:
        n_scored = 16
    if "n_correct_hybrid" in variant and variant["n_correct_hybrid"] is not None:
        n_correct = float(variant["n_correct_hybrid"])
        if n_scored > 0 and n_scored != 16:
            return n_correct * (16.0 / n_scored)
        return n_correct
    if "p_new" in variant and variant["p_new"] is not None:
        return float(variant["p_new"]) * 17.0 - 0.5
    raise ValueError("Variant must contain n_correct_hybrid or p_new")


def _tier_sort_key(variant: Dict[str, Any]) -> Tuple[int, str]:
    tier = variant.get("generation_tier")
    if not tier:
        tier = infer_generation_tier(
            str(variant.get("origin", "")),
            variant.get("generation_tag"),
        )
    rank = GENERATION_TIER_ORDER.get(str(tier), 99)
    return (rank, str(variant.get("variant_id", "")))


def sort_ladder_variants(
    variants: Sequence[Dict[str, Any]],
    tie_delta_correct: float = 1.0,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Sort variants for one source problem from easiest (rank 1) to hardest.

    Per T-023 Stage 2 step 5:
    - Primary order: p_new descending (easiest first).
    - Fallback: when two variants differ by less than 1 correct rollout
      (abs(eff_correct_a - eff_correct_b) < tie_delta_correct), order them by
      generation tier (easier < nearby < paper_transfer < harder) and record
      `tie_broken_by_tag = True`.

    Returns:
        (sorted_variants_with_ladder_rank, tie_broken_by_tag)
    """
    if not variants:
        return [], False

    # First sort descending by p_new (or effective correct count), deterministic by variant_id.
    items = [dict(v) for v in variants]
    items.sort(
        key=lambda v: (
            -(float(v["p_new"]) if v.get("p_new") is not None else _effective_correct_16(v)),
            str(v.get("variant_id", "")),
        )
    )

    # Group contiguous clusters where consecutive items differ by < tie_delta_correct.
    clusters: List[List[Dict[str, Any]]] = [[items[0]]]
    for item in items[1:]:
        prev_eff = _effective_correct_16(clusters[-1][-1])
        cur_eff = _effective_correct_16(item)
        if abs(prev_eff - cur_eff) < tie_delta_correct:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    tie_broken_by_tag = False
    ordered: List[Dict[str, Any]] = []
    for cluster in clusters:
        if len(cluster) > 1:
            tie_broken_by_tag = True
            cluster.sort(key=_tier_sort_key)
        ordered.extend(cluster)

    for idx, item in enumerate(ordered, start=1):
        item["ladder_rank"] = idx
        item["ladder_tie_broken_by_tag"] = tie_broken_by_tag

    return ordered, tie_broken_by_tag
