# T-032 Route E Preflight Kit (`routeE_t032`)

- **Purpose**: Stage 1 CPU-validated toolkit for Route E preflight checks (P1–P4) on Pro 5000.
- **Precision & Grading**: `bfloat16` throughout; `math-verify==0.8.0`, `latex2sympy2-extended==1.10.2`, `latex2sympy2==1.5.4`; Agnes disabled (0 concurrency).
- **External Code**: `p-e-w/heretic@d89c7a2d5a970f7ec634cd300f2bf893dd8048d4` (downloaded separately via `gh-proxy.org` and SHA-256 verified on Pro 5000; not bundled in this ZIP).

## Modules
- `common_utils.py`: File/version lock checks, `<think>` stripping, math grading, 10,000-draw paired bootstrap, Pro 5000 idle GPU detection.
- `p1_math_checkup.py`: MATH500 (0-shot) & GSM8K test (4-shot) per PedagogicalRL (EMNLP 2025) Table 3 (`temperature=0.6`, `top_p=0.95`, `max_new_tokens=8192`, `bfloat16`).
- `p2_gold_sampling.py`, `p2_view_dialogue.py`, `p2_export_claude30.py`: Stratified 50+50 blind sampling, hard-gated sealed mapping key (`UnsealGateError` until 100+30 annotations complete), user-only viewer, and 30-item blind export for Claude.
- `leakage_judges.py`, `p2_calibrate_consistency.py`: Layer 1 deterministic screener + Layer 2 3-question checklist 27B judge + single-sentence & author baselines + F1/Recall/Cohen's $\kappa$ calibration.
- `p3_rejudge_494.py`: Re-judges A-group (T-020) and Eduardo-27B (T-022) 494 conversations and splits overall / empty / non-empty leakage rates.
- `p4_contrastive_data.py`, `p4_e0_methods.py`, `p4_eval_val100.py`: Train-only contrastive rewriting (`±15%` length filter, 80/20 problem-grouped split), 6 E0 methods × 2 bases (`ara_r50`, `ara_r16`, `dir_ablation`, `dir_addition`, `dpo_lora_r32`, `random_ablation`, `prompt_only`) with 20-prompt greedy unplug exact-match check, and `val.parquet` (100 problems) evaluation.
