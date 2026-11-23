# Author-494 Evaluation Kit v2 (T-020: Pro 5000 A-group step_100 vs A100 T-018 base)

用于在 **A100**（抽取 T-018 逐题表）与 **RTX PRO 5000**（换机器检查、A 组第 100 步全量、配对 bootstrap 与汇总）上按作者上游 `e8c89b7` 评测作者 494 题。

## v2 核心流程

1. **A100 侧（`export_t018_table.py`）**：从 T-018 阶段 2/3 两个结果 JSON 抽出脱敏逐题表 `t018_per_problem.csv`（494 行，无题面/答案/对话/裁判理由）与 `t018_manifest.json`（含 T-018 `test.parquet` 的 SHA-256），复算核对 `< 1e-9` 后打包为 `t018_table_bundle.tar.gz`。
2. **Pro 5000 侧换机器检查（`run_control_author494_pro5000.sh` + `compare_stack.py`）**：
   - 先核对 Pro 5000 `test.parquet` 与 A100 T-018 的 `test.parquet` SHA-256 逐字节一致；
   - 跑一轮 494 题无辅导对照 `ctl`（调用作者 `_solve_rate`，约 15,808 次，不调 Agnes），与 A100 `ctl1` 比均值差 `<= 0.02` 与逐题相关（迁移 `>= 0.834`、原题 `>= 0.817`）；
   - 核对 base 20 题冒烟（`assume_thinking=True`、`Empty%` 在 `70.14% ± 15pp`、`student_calls_failed=0`）。
3. **Pro 5000 侧阶段 2（A 组第 100 步全量 → `make_paired_csv.py` → `summarize_author494_pro5000.py`）**：
   - 只跑 A 组第 100 步全量（Agnes 并发 `2 + 1 = 3` 路），不再重跑 base 全量；
   - `make_paired_csv.py` 读 A 组结果 JSON 与 A100 `t018_per_problem.csv`，核对行号对齐、前测一致、CSV 均值与两边原始结果差 `< 1e-9`；
   - `summarize_author494_pro5000.py` 给出判定 A（对比论文 Eduardo-4B）、判定 B（`paired_gain.py` 逐题配对）、换机器检查表及次要指标并排表（含 `ctl1` 换基线、R2、R3、均值差）。

## cu04 本地单元测试

```bash
cd ~/项目/数学迁移/代码与工作树/eduardo-author494-eval-pro5000-20261123
~/.local/envs/mhpo/bin/python -B -m unittest discover -s evaluation/author494_pro5000 -v
SOURCE_GAIN_PY="$PWD/evaluation/agnes_rejudge/source_gain.py" \
  ~/.local/envs/mhpo/bin/python -B -m unittest discover -s evaluation/author494_pro5000/paired_gain -v
```
