"""Route A reward package: ladder post-test + Rasch MAP ability gain + hybrid grader.

Self-contained module that reuses author EduardoRL functions without modifying any
upstream author files. Designed for veRL ``custom_reward_function`` wiring via
``path: pkg://training.routeA_reward.reward_function`` (or file path) and
``name: route_a_reward_func``.
"""

from __future__ import annotations

from .hybrid_grader import (
    AGNES_JUDGE_REFERENCE_SHA256,
    EQUIV_JUDGE_SYSTEM_PROMPT,
    AsyncAdaptiveWindow,
    GradeResult,
    HybridGrader,
    build_equiv_messages,
    check_technical_failure_rate,
    file_sha256,
    parse_equiv_verdict,
    verify_hybrid_grader_sha256,
)
from .ladder import (
    ClimbOutcome,
    LadderBatchResult,
    LadderVariant,
    climb_ladder_batch,
    extract_ladder_variants,
    select_middle_variant,
)
from .rasch import (
    RASCH_NEWTON_MAX_ITER,
    RASCH_NEWTON_TOL,
    RaschEstimate,
    difficulty_from_rate,
    difficulty_from_solve_count,
    rasch_grid_search,
    rasch_log_posterior,
    rasch_map_estimate,
    sigmoid,
)
from .reward_function import (
    VALID_REWARD_MODES,
    resolve_reward_mode,
    route_a_reward_func,
    summarize_step_metrics,
)

__all__ = [
    "AGNES_JUDGE_REFERENCE_SHA256",
    "AsyncAdaptiveWindow",
    "ClimbOutcome",
    "EQUIV_JUDGE_SYSTEM_PROMPT",
    "GradeResult",
    "HybridGrader",
    "LadderBatchResult",
    "LadderVariant",
    "RASCH_NEWTON_MAX_ITER",
    "RASCH_NEWTON_TOL",
    "RaschEstimate",
    "VALID_REWARD_MODES",
    "build_equiv_messages",
    "check_technical_failure_rate",
    "climb_ladder_batch",
    "difficulty_from_rate",
    "difficulty_from_solve_count",
    "extract_ladder_variants",
    "file_sha256",
    "parse_equiv_verdict",
    "rasch_grid_search",
    "rasch_log_posterior",
    "rasch_map_estimate",
    "resolve_reward_mode",
    "route_a_reward_func",
    "select_middle_variant",
    "sigmoid",
    "summarize_step_metrics",
    "verify_hybrid_grader_sha256",
]
