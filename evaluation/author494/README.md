# Author-494 Base Evaluation on A100 (`Qwen3.5-4B` base)

本目录用于在 A100 上以 4 张卡（Tutor 1 卡 + 学生 3 卡 DP=3）复跑作者上游 `e8c89b7` 的 `eval/run_env_test_eval.py`，并将评测端 `--gemini` 接口指向 Agnes，对齐论文 Table 1、7、8（n=494）。

## 文件说明

- `preflight_author494.sh`：只读预检（不占卡、不调 Agnes），检查卡和进程归属、端口、磁盘、作者上游 `e8c89b7` 文件 SHA-256、包版本、Qwen3.5-4B 与 Llama-3.1-8B-Instruct 的 SHA-256、官方数据集 SHA-256 与环境变量状态。
- `agnes_ping_author494.py`：用 1 条完全合成的假对话走作者 `make_judge_client` 路径调 1 次 Agnes，只打印 HTTP 状态、返回模型名和是否解析成作者要求的 JSON。
- `run_base_author494.sh`：先离线通过未改动的 `eduardo.process_dataset` 物化并校验 494 行 `test.parquet`；检查选定的 4 张空闲整卡，启动 3 卡 DP=3 学生 `vllm serve`（参数按 `launch_student_server.sh`），每 60 秒记录 4 张卡的进程归属，在 1 张卡上跑作者 `eval/run_env_test_eval.py`，结束时仅清理本任务启动的进程。支持 `--status <output_dir>` 只读看进度。
- `summarize_author494.py`：严格校验结果 JSON 结构，复算作者汇总指标并核对一致；分开统计 Agnes 重试耗尽技术失败票与解析失败票，若技术失败票 > 1% 则将泄题率/教学门槛/Ped./Fact. 标为“不可用”；按题做 10,000 次 bootstrap（种子 42）；按事先定好的 3 档规则给出对照结论；不打印任何题号、题面、答案、对话或裁判理由。
- `test_summarize_author494.py`：标准库 `unittest` 测试（纯假数据）。

## cu04 单元测试

```bash
cd ~/项目/数学迁移/代码与工作树/eduardo-author494-eval-20261121
~/.local/envs/mhpo/bin/python -B -m unittest discover -s evaluation/author494 -v
```

## A100 执行顺序

以下命令均在容器 `edu-contain` 内运行（先从宿主机将打包文件与 `.sha256` 复制进容器并校验后再解包到新目录 `/workspace/author494_eval_code`）。

### 1. 预检（不占卡、不调 Agnes）

```bash
export AGNES_API_KEY="$(tr -d '\r\n' < /workspace/.secrets/agnes_api_key)"
unset GEMINI_API_KEY HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
bash /workspace/author494_eval_code/evaluation/author494/preflight_author494.sh
```

### 2. 调 1 次 Agnes（测试接口与 `reasoning_effort=high` 兼容性）

```bash
/workspace/A_group_repro_20261117/envs/env-candidate-vllm0191/bin/python -B \
  /workspace/author494_eval_code/evaluation/author494/agnes_ping_author494.py \
  --api-base https://api.agnes-ai.cn/v1 \
  --model agnes-3.0-flash \
  --api-key-env AGNES_API_KEY
```

### 3. 准备 494 题 `test.parquet` 并冒烟（前 20 题，替换 `<TUTOR_GPU>` 和 `<S1,S2,S3>` 为预检确认的 4 张空闲整卡；需与第 1–2 步在同一终端中保持 `AGNES_API_KEY` 已导出）

```bash
nohup bash /workspace/author494_eval_code/evaluation/author494/run_base_author494.sh \
  --tutor-gpu <TUTOR_GPU> \
  --student-gpus <S1,S2,S3> \
  --num-problems 20 \
  --tag base_qwen35_4b_author494_smoke20 \
  --output-dir /workspace/A_group_repro_20261117/eval_author494_smoke20 \
  > /workspace/A_group_repro_20261117/eval_author494_smoke20_launcher.log 2>&1 < /dev/null &
```

看进度（只数 `Round N` 和 `Scoring` 状态行，不打印正文）：

```bash
bash /workspace/author494_eval_code/evaluation/author494/run_base_author494.sh \
  --status /workspace/A_group_repro_20261117/eval_author494_smoke20
```

### 4. 阶段 2 全量 494 题（阶段 1 审核通过后再运行）

```bash
nohup bash /workspace/author494_eval_code/evaluation/author494/run_base_author494.sh \
  --tutor-gpu <TUTOR_GPU> \
  --student-gpus <S1,S2,S3> \
  --num-problems -1 \
  --tag base_qwen35_4b_author494 \
  --output-dir /workspace/A_group_repro_20261117/eval_author494_full494 \
  > /workspace/A_group_repro_20261117/eval_author494_full494_launcher.log 2>&1 < /dev/null &
```

跑完后汇总（如启动器结束输出未完整贴回，可单独重跑只读汇总）：

```bash
/workspace/A_group_repro_20261117/envs/env-candidate-vllm0191/bin/python -B \
  /workspace/author494_eval_code/evaluation/author494/summarize_author494.py \
  /workspace/A_group_repro_20261117/eval_author494_full494/transfer_eval_base_qwen35_4b_author494_*.json
```
