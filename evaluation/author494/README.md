# Author-494 Base Evaluation on A100 (`Qwen3.5-4B` base)

本目录用于在 A100 上以 4 张卡（Tutor 1 卡 + 学生 3 卡 DP=3）复跑作者上游 `e8c89b7` 的 `eval/run_env_test_eval.py`（阶段 1–2），并运行无辅导对照（3 卡学生、`eduardo.create_transfer_dataset._solve_rate`）与扩展统计汇总（阶段 3，v5 包）。

## 文件说明

- `preflight_author494.sh`：只读预检（不占卡、不调 Agnes），检查卡和进程归属、端口、磁盘、作者上游 `e8c89b7`（22 个文件，含 `eduardo/create_transfer_dataset.py`）SHA-256、包版本、`Qwen3.5-4B` 与 `Llama-3.1-8B-Instruct` 的 SHA-256、官方数据集 SHA-256 与环境变量状态。
- `agnes_ping_author494.py`：用 1 条合成假对话走作者 `make_judge_client` 路径调 1 次 Agnes，只打印 HTTP 状态、返回模型名和是否解析成作者要求的 JSON。
- `wait_gpus_author494.py`：只读等卡器（等待期间不占显存、不动他人进程），支持 `--num-gpus 4`（阶段 1–2）与 `--num-gpus 3`（阶段 3 对照），以及 `--hold` / `--find-replacement`（阶段 1–2 保护 Tutor 卡交接窗口）。
- `run_base_author494.sh`：阶段 1–2 启动脚本（3 卡 DP=3 学生 `vllm serve` + 1 卡 Tutor `eval.run_env_test_eval`），支持 `--auto-wait`、`--status` 与自动汇总。
- `run_control_author494.py`：阶段 3 无辅导对照 Python 驱动，直接调用未改动的 `eduardo.create_transfer_dataset._solve_rate`（`temperature=0.6`, `max_tokens=1024`, `k=16`）在 494 题原题和迁移题上各跑两轮（`ctl1` / `ctl2`），失败调用超过 1% 自动停下报警，屏幕与日志不打印任何题号、题面、答案或作答原文。
- `run_control_author494.sh`：阶段 3 无辅导对照 bash 启动器，只占 3 张空闲整卡，拉起与阶段 2 同参数的学生 `vllm serve`，跑完后仅关闭本任务启动的进程；支持 `--auto-wait`、`--status` 与可选 `--stage2-json` 自动联动汇总。
- `summarize_author494.py`：汇总脚本 v2，除阶段 1–2 校验、Agnes 诊断、bootstrap 区间与三档判定外，新增 R2 Hake g、R3 Cohen's d、配对 `transfer_gap` 区间、过程-结果相关（排除重试失败对话，Pearson/Spearman 含并列秩处理与按题聚类 bootstrap），以及 `--control-json` 下的 R4 无辅导基线（`pre -> ctl1`）、`ctl1` 换基线 R1（`ctl1 -> post`）、重测信度、按 `ctl2_transfer` 切点 `1/3` 与 `2/3` 划分的难度三档，与脱敏异常定位 `extract_safe_failure_summary`（只输出调用栈与异常类名）。
- `test_summarize_author494.py`：标准库 `unittest` 测试（纯假数据 / 假客户端）。

## cu04 单元测试

```bash
cd ~/项目/数学迁移/代码与工作树/eduardo-author494-eval-20261121
~/.local/envs/mhpo/bin/python -B -m unittest discover -s evaluation/author494 -v
```
