# Route C Training Stage 1 (`training/routeC_train`)

T-034 路线 C 正式训练（阶段 1：代码、单测与 Pro 5000 无卡预检）模块。相对路线 A（T-026）仅改变每道原题的“教题与后测梯子路由”，其余题池、奖励公式（`ladder_rasch`, `k=8`, `G = θ̂ / 4`, 门禁 `-0.2`）、超参、精度（`bf16`）与卡布局均保持一致。

## 字段核对（T-034 阶段 1 第 2 条）

已核对作者上游 `EduardoRL@e8c89b7`：
- Tutor 提示词（`eduardo/configs/prompts/teacher_prompt.txt`）、学生首轮开场（`eduardo/process_dataset.py`）与学生模拟器（`eduardo/configs/prompts/student_prompt.txt`、`eduardo/eduardo_interaction.py`）**仅使用 `{problem}`**。
- 两道门禁裁判（`eduardo/configs/prompts/judge_leakage.txt`、`eduardo/configs/prompts/judge_quality.txt`、`eduardo/metrics/judge_quality.py`）**仅使用 `{problem}`、`{answer}`、`{conversation}`**，不用参考解答或其它字段。
- T-023 重测包中每道变体均校验非空 `problem` 和 `answer`，缺字段立即报错停下。

## 组件

- `prepare_parquet.py`：在 Pro 5000 本机读取 T-023 重测包（`9ff8449b…`）、T-031 新原题前测（`a031276f…`）与 T-026 `train.parquet` / `val.parquet`（`1e85fd0f…` / `9a6808ea…`），按 `τ = 0.4, m = 1.0` 生成 `routeC_t034/{train.parquet,val.parquet}`：
  - 简单支（`p_原 ≥ 0.4` 且存在 `Δb ≥ 1.0`）：教原题，后测仅保留 `Δb > -1.0` 的变体；
  - 难题支（`p_原 < 0.4` 且存在 `Δb ≤ -1.0`）：教 `Δb ≤ -1.0` 中 `b` 最大的变体（平局取编号小的），后测为原题 + 其余未教变体；
  - 兜底支（`fallback`）：逐字段与 T-026 一致，仅新增 `route_branch` 与 `taught_id` 字段。
- `preflight_stage1.py` & `run_stage1_preflight.sh`：Pro 5000 无卡预检（`CUDA_VISIBLE_DEVICES=""`），不占卡、兼容 T-033 在跑状态；核对包版本、SHA-256、`/dev/shm` 权重、生成路线 C parquet 并校验各支计数（`289/801/805` 与 `15/46/39`）、运行 Hydra `--check-only` 及全套 6 个单测。
- `test_routeC_train.py`：仅用假数据的路由边界、平局、T-026 兜底一致、奖励逐位一致、按支与按门禁类型日志及脱敏校验单测。
