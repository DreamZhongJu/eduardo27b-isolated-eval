# T-033: Route A Step-100 Evaluation on Pro 5000

Unified evaluation package for T-033 on `8 x RTX PRO 5000 72GB`:
- `preflight_t033.py`: Package (`math-verify==0.8.0`, `latex2sympy2-extended==1.10.2`), Step-100 HF bf16 SHA-256 (7/7), frozen module SHA-256 (`agnes_judge.py`, `hybrid_grader.py`, `paired_gain.py`), dataset SHA-256, GPU, disk, and unit-test preflight.
- `check_overlap_t033.py`: Zero-leakage overlap check between T-026 `train`/`val` parquets (source & variant problems) and Author 494 / Self-built 3-tier test sets (prints counts only).
- `precision_compare_selfbuilt.py`: 90-source deterministic student `fp16` vs `bf16` untutored precision comparison (`30/30/30`, 32 calls/source each).
- `run_eval_routea_author494.py` / `run_t033_author494.sh`: Pipeline ① (Author 494, `bf16`, Agnes AIMD `1..16`, `allow_abbrev=False`).
- `run_post_eval_routea.py` / `run_t033_selfbuilt.sh`: Pipeline ② (Self-built 3-tier, Tutor & Student `bf16`, Agnes AIMD `1..16`, `--smoke-total 16` or `--mode full`).
- `run_t033_dev_sim.py` / `run_t033_dev_sim.sh`: Pipeline ③ (Dev 100 Route A reward & unconditional `theta_hat`, `bf16`, 27B gate + `hybrid_grader.py`).
- `run_t033_stage1.sh`: Single Stage 1 entrypoint (`--preflight-only`, background GPU run with `<= 4` GPUs, `--resume`, `--status`).
