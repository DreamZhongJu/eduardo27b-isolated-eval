# Route A Reward (`training/routeA_reward`)

Independent reward module for T-024 (Route A: ladder post-test + Rasch MAP ability gain + hybrid grader).
Does **not** modify any upstream author files in `eduardo/` (`e8c89b7`).

## Modules

- `rasch.py`: Numerically stable Newton-Raphson Rasch MAP estimator with `N(0, 1)` prior (`RASCH_NEWTON_TOL = 1e-10`), difficulty helper `b = ln((1 - p) / p)`, and reference grid search.
- `hybrid_grader.py`: Rule-first (`eduardo.utils.grade_math_answer`, math-verify 0.8.0) + local Qwen3.6-27B fallback (prompt from `evaluation/agnes_eval_v3/agnes_judge.py`, sha256 `2ffe7464...`) with async AIMD congestion window (1..16) and SHA-256 verification helper. **Before Stage 2, replace with T-023's canonical `hybrid_grader.py` and verify identical SHA-256.**
- `ladder.py`: Level-by-level concurrent ladder runner (`k = 8` default, 1–3 variants ordered by `ladder_rank`), directly calling `eduardo.metrics.correctness._build_final_attempt_messages` for masking and final prompt rendering.
- `reward_function.py`: veRL async reward entry point `route_a_reward_func` supporting `reward.mode` in `{"ladder_rasch", "author", "author_hybrid"}` and sanitized numeric-only step summary `summarize_step_metrics`.

## veRL Custom Reward Wiring

With both `EduardoRL` (`e8c89b7`) and this repository root on `PYTHONPATH`:

```bash
custom_reward_function.path=pkg://training.routeA_reward.reward_function
custom_reward_function.name=route_a_reward_func
custom_reward_function.reward_kwargs.mode=ladder_rasch
custom_reward_function.reward_kwargs.ladder_num_climbs=8
custom_reward_function.reward_kwargs.gate_penalty=0.2
custom_reward_function.reward_kwargs.max_tokens_per_turn=150
custom_reward_function.reward_kwargs.max_thinking_tokens_per_turn=512
```

> Note: When running under veRL's `migrate_legacy_reward_impl`, preserve
> `config.custom_reward_function.reward_kwargs.{max_tokens_per_turn,max_thinking_tokens_per_turn}`
> (`150` / `512`) so `EduardoAgentLoop` keeps the exact `150 + 512 = 662` per-turn cap.
